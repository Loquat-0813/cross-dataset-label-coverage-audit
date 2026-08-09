"""Evaluate an E3 coverage guard against its matched flat adapter output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1data.external import RasterPair, discover_external_pairs
from p1eval.coverage_evaluation import CoverageAwareAccumulator
from p1eval.external_transfer import ExternalTransferAccumulator
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.selective_hierarchy import build_coverage_node_lookup
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.coverage_guard import CoverageGuard
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.loveda_e1 import base_leaf_logits, source_supervised_leaf_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("openearthmap", "landcoverai"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, default=ROOT / "configs" / "taxonomy_prompts_v0.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-prompt-diagnostic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_e3_checkpoint(path: Path, prompts: tuple[str, ...], allow_prompt_diagnostic: bool) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    run_config = checkpoint.get("run_config")
    conformal = checkpoint.get("conformal_calibration")
    if not isinstance(run_config, dict) or run_config.get("experiment") != "e3_coverage_guard":
        raise ValueError("--adapter-checkpoint must be produced by train_loveda_coverage_guard.py")
    if not isinstance(conformal, dict) or "probability_threshold" not in conformal:
        raise ValueError("E3 checkpoint is missing conformal_calibration.probability_threshold")
    threshold = float(conformal["probability_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("E3 conformal probability threshold must lie in [0, 1]")
    trained_prompts = tuple(run_config.get("prompts", ()))
    if trained_prompts != prompts and not allow_prompt_diagnostic:
        raise ValueError(
            "evaluation prompts differ from checkpoint prompts; use --allow-prompt-diagnostic only for a clearly labelled exploratory run"
        )
    return checkpoint, conformal


@torch.inference_mode()
def predict_pair(
    pair: RasterPair,
    backbone: FrozenCLIPPatchEncoder,
    adapter: LowRankSemanticCalibration,
    guard: CoverageGuard,
    text_features: torch.Tensor,
    leaf_node_ids: torch.Tensor,
    probability_threshold: float,
    tile_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw labels plus matched flat and guarded node-level predictions."""
    with Image.open(pair.image_path) as image_file, Image.open(pair.mask_path) as mask_file:
        image = image_file.convert("RGB")
        raw_mask = np.asarray(mask_file)
        if raw_mask.ndim != 2:
            raise ValueError(f"{pair.identifier}: expected one-channel mask, got {raw_mask.shape}")
        if image.size != mask_file.size:
            raise ValueError(f"{pair.identifier}: image size {image.size} does not match mask size {mask_file.size}")
        width, height = image.size
        flat_node_prediction = np.empty((height, width), dtype=np.int64)
        guarded_node_prediction = np.empty((height, width), dtype=np.int64)
        routed_to_base = np.empty((height, width), dtype=bool)
        conformal_set_size = np.empty((height, width), dtype=np.int16)
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                rgb = np.asarray(image.crop((left, top, right, bottom)), dtype=np.uint8)
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[: bottom - top, : right - left] = rgb
                tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device)
                features = backbone.encode_image(tensor)
                base_logits = F.interpolate(
                    base_leaf_logits(features, text_features),
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )
                adapted_logits = F.interpolate(
                    adapter(features, text_features),
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )
                output = guard(base_logits, adapted_logits, probability_threshold)
                flat_nodes = leaf_node_ids[adapted_logits.argmax(dim=1)]
                rows = slice(top, bottom)
                columns = slice(left, right)
                tile_height = bottom - top
                tile_width = right - left
                flat_node_prediction[rows, columns] = flat_nodes[0, :tile_height, :tile_width].cpu().numpy()
                guarded_node_prediction[rows, columns] = output.node_prediction[0, :tile_height, :tile_width].cpu().numpy()
                routed_to_base[rows, columns] = output.routed_to_base[0, :tile_height, :tile_width].cpu().numpy()
                conformal_set_size[rows, columns] = output.conformal_membership[0, :, :tile_height, :tile_width].sum(dim=0).cpu().numpy()
    return (
        np.array(raw_mask, copy=True),
        flat_node_prediction,
        guarded_node_prediction,
        routed_to_base,
        conformal_set_size,
    )


def main() -> int:
    args = parse_args()
    if args.tile_size < 1 or args.tile_size % 16:
        raise SystemExit("--tile-size must be a positive multiple of CLIP's 16-pixel patch size")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive when set")
    taxonomy, levels = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    target_mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", args.dataset, taxonomy)
    source_mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
    if not target_mapping.ready_for_scoring:
        raise SystemExit(f"{args.dataset}: mapping is not ready for scoring")
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(args.prompt_config, leaves.leaf_names)
    checkpoint, conformal = _load_e3_checkpoint(args.adapter_checkpoint, prompts, args.allow_prompt_diagnostic)
    run_config = checkpoint["run_config"]
    coverage_mask = source_supervised_leaf_mask(source_mapping, leaves.leaf_names).numpy().astype(np.bool_, copy=False)
    expected_covered = tuple(name for name, covered in zip(leaves.leaf_names, coverage_mask) if covered)
    expected_uncovered = tuple(name for name, covered in zip(leaves.leaf_names, coverage_mask) if not covered)
    if tuple(run_config.get("covered_leaf_names", ())) != expected_covered or tuple(run_config.get("uncovered_leaf_names", ())) != expected_uncovered:
        raise SystemExit("E3 checkpoint source-coverage declaration disagrees with the current audited LoveDA mapping")
    data_root = args.data_root or ROOT / "data" / "raw" / args.dataset
    discovery = discover_external_pairs(args.dataset, data_root)
    pairs = discovery.pairs if args.max_samples is None else discovery.pairs[: args.max_samples]
    report_config = {
        "dataset": args.dataset,
        "experiment": "e3_coverage_guard",
        "data_root": str(data_root),
        "model_id": args.model_id,
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "taxonomy": taxonomy.version,
        "mapping": "dataset-label-mappings-v0",
        "tile_size": args.tile_size,
        "prompt_config": str(args.prompt_config),
        "prompts": list(prompts),
        "prompt_policy": "diagnostic_override_not_eligible_for_headline_claim"
        if args.allow_prompt_diagnostic
        else "checkpoint_matched_protocol_prompt",
        "probability_threshold": float(conformal["probability_threshold"]),
        "alpha": float(conformal.get("alpha", run_config.get("alpha"))),
        "covered_leaf_names": list(expected_covered),
        "uncovered_leaf_names": list(expected_uncovered),
        "audited_mask_count": discovery.audited_mask_count,
        "paired_raster_count": discovery.paired_raster_count,
        "unpaired_label_count": len(discovery.unpaired_mask_identifiers),
        "unpaired_label_examples": list(discovery.unpaired_mask_identifiers[:5]),
        "selected_raster_count": len(pairs),
        "matched_flat_control": "the same E3 adapter before coverage routing",
    }
    if args.dry_run:
        print("coverage_guard_external_dry_run_valid:", json.dumps(report_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("coverage guard external evaluation is restricted to CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory already contains files: {args.output_dir}; use --overwrite only intentionally")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(report_config, indent=2), encoding="utf-8")
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    adapter = LowRankSemanticCalibration(backbone.embedding_dim, rank=int(run_config["rank"])).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    leaf_node_ids = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], dtype=torch.long, device=device)
    coverage_lookup = build_coverage_node_lookup(taxonomy, levels, node_to_id, leaves.leaf_names, coverage_mask)
    guard = CoverageGuard(
        torch.from_numpy(coverage_mask),
        leaf_node_ids.cpu(),
        torch.from_numpy(coverage_lookup),
    ).to(device)
    flat_external = ExternalTransferAccumulator(
        target_mapping,
        taxonomy,
        levels,
        node_to_id,
        leaf_node_ids.cpu().numpy(),
        leaves.leaf_names,
    )
    node_to_leaf_id = np.full(max(node_to_id.values()) + 1, -1, dtype=np.int64)
    node_to_leaf_id[leaf_node_ids.cpu().numpy()] = np.arange(len(leaves.leaf_names), dtype=np.int64)
    flat_coverage = CoverageAwareAccumulator(target_mapping, taxonomy, levels, node_to_id, leaves.leaf_names, coverage_mask)
    guarded_coverage = CoverageAwareAccumulator(target_mapping, taxonomy, levels, node_to_id, leaves.leaf_names, coverage_mask)
    for index, pair in enumerate(pairs, start=1):
        raw_mask, flat_nodes, guarded_nodes, routed, set_size = predict_pair(
            pair,
            backbone,
            adapter,
            guard,
            text_features,
            leaf_node_ids,
            float(conformal["probability_threshold"]),
            args.tile_size,
            device,
        )
        flat_leaf_ids = node_to_leaf_id[flat_nodes]
        flat_external.update(raw_mask, flat_leaf_ids)
        flat_coverage.update(raw_mask, flat_nodes)
        guarded_coverage.update(raw_mask, guarded_nodes, routed, set_size)
        print(f"evaluated={index}/{len(pairs)} identifier={pair.identifier}", flush=True)
    flat_result = flat_coverage.result()
    guarded_result = guarded_coverage.result()
    report = {
        "run_config": report_config,
        "metrics": {
            "flat_adapter": {
                "external_leaf_and_coarse_diagnostic": flat_external.result().as_dict(),
                "coverage_aware_suite": flat_result.as_dict(),
            },
            "coverage_guard": guarded_result.as_dict(),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "coverage_guard_external_complete:",
        f"dataset={args.dataset}",
        f"flat_covered_miou={flat_result.covered_leaf.macro_iou}",
        f"guarded_covered_miou={guarded_result.covered_leaf.macro_iou}",
        f"guarded_vlr={guarded_result.selective_hierarchy.violation_leaf_rate}",
        f"guarded_route_rate={guarded_result.route_rate}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
