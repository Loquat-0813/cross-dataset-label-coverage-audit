"""Run the preregistered E0 herbaceous prompt robustness audit on one target set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1data.external import RasterPair, discover_external_pairs
from p1eval.external_transfer import ExternalTransferAccumulator
from p1eval.metrics import stable_node_index
from p1eval.prompt_audit import PromptAuditPlan, load_prompt_audit_plan
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.taxonomy import build_taxonomy_leaf_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("openearthmap",), default="openearthmap")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        help="Optional original E1 checkpoint. When omitted, run the frozen E0 audit.",
    )
    parser.add_argument("--prompt-audit-config", type=Path, default=ROOT / "configs" / "herbaceous_prompt_audit_v1.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_base_prompts(leaf_names: tuple[str, ...]) -> tuple[str, ...]:
    document = yaml.safe_load((ROOT / "configs" / "taxonomy_prompts_v0.yaml").read_text(encoding="utf-8"))
    configured = document["leaf_prompts"]
    if set(configured) != set(leaf_names):
        raise ValueError("prompt configuration must contain every and only taxonomy leaf")
    return tuple(configured[name] for name in leaf_names)


def _run_pair_all_variants(
    pair: RasterPair,
    backbone: Any,
    adapter: Any | None,
    text_features_by_variant: dict[str, Any],
    tile_size: int,
    device: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode each image tile once, then score every preregistered text variant."""
    import torch
    import torch.nn.functional as F

    from p1train.loveda_e1 import base_leaf_logits

    with Image.open(pair.image_path) as image_file, Image.open(pair.mask_path) as mask_file:
        image = image_file.convert("RGB")
        raw_mask = np.asarray(mask_file)
        if raw_mask.ndim != 2:
            raise ValueError(f"{pair.identifier}: expected one-channel mask, got {raw_mask.shape}")
        if image.size != mask_file.size:
            raise ValueError(f"{pair.identifier}: image size {image.size} does not match mask size {mask_file.size}")
        width, height = image.size
        predictions = {name: np.empty((height, width), dtype=np.int64) for name in text_features_by_variant}
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                rgb = np.asarray(image.crop((left, top, right, bottom)), dtype=np.uint8)
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[: bottom - top, : right - left] = rgb
                tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device)
                features = backbone.encode_image(tensor)
                for name, text_features in text_features_by_variant.items():
                    logits = base_leaf_logits(features, text_features) if adapter is None else adapter(features, text_features)
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                    predictions[name][top:bottom, left:right] = logits.argmax(dim=1)[0, : bottom - top, : right - left].cpu().numpy()
    return np.array(raw_mask, copy=True), predictions


def _audit_config(plan: PromptAuditPlan, args: argparse.Namespace, discovery) -> dict[str, Any]:
    return {
        "experiment": "e0_herbaceous_prompt_audit",
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "model_id": args.model_id,
        "adapter_checkpoint": None if args.adapter_checkpoint is None else str(args.adapter_checkpoint),
        "prompt_audit_config": str(args.prompt_audit_config),
        "prompt_audit_version": plan.version,
        "target_leaf": plan.target_leaf,
        "tile_size": args.tile_size,
        "audited_mask_count": discovery.audited_mask_count,
        "paired_raster_count": discovery.paired_raster_count,
        "selected_raster_count": len(discovery.pairs),
        "unpaired_label_count": len(discovery.unpaired_mask_identifiers),
        "unpaired_label_examples": list(discovery.unpaired_mask_identifiers[:5]),
        "individual_variants": {name: plan.prompts_for_candidate(name) for name in plan.individual_variant_names},
        "ensemble_variant": plan.ensemble_variant_name,
        "ensemble_members": list(plan.ensemble_members),
    }


def main() -> int:
    args = parse_args()
    if args.tile_size < 1 or args.tile_size % 16:
        raise SystemExit("--tile-size must be a positive multiple of CLIP's 16-pixel patch size")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive when set")
    data_root = args.data_root or ROOT / "data" / "raw" / args.dataset
    args.data_root = data_root
    taxonomy, levels = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", args.dataset, taxonomy)
    if not mapping.ready_for_scoring:
        raise SystemExit(f"{args.dataset}: mapping is not ready for scoring")
    discovery = discover_external_pairs(args.dataset, data_root)
    pairs = discovery.pairs if args.max_samples is None else discovery.pairs[: args.max_samples]
    discovery = type(discovery)(
        dataset=discovery.dataset,
        pairs=pairs,
        audited_mask_count=discovery.audited_mask_count,
        unpaired_mask_identifiers=discovery.unpaired_mask_identifiers,
    )
    leaf_index = build_taxonomy_leaf_index(taxonomy)
    plan = load_prompt_audit_plan(args.prompt_audit_config, leaf_index.leaf_names, load_base_prompts(leaf_index.leaf_names))
    run_config = _audit_config(plan, args, discovery)
    if args.adapter_checkpoint is not None:
        run_config["experiment"] = "e1_herbaceous_prompt_audit"
    if args.dry_run:
        print("herbaceous_prompt_audit_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")

    import torch
    import torch.nn.functional as F

    from p1backbone.hf_clip import FrozenCLIPPatchEncoder
    from p1model.calibration import LowRankSemanticCalibration

    if not torch.cuda.is_available():
        raise SystemExit("prompt audit is restricted to CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    adapter = None
    if args.adapter_checkpoint is not None:
        if not args.adapter_checkpoint.is_file():
            raise FileNotFoundError(f"adapter checkpoint is missing: {args.adapter_checkpoint}")
        # This audit is for the original all-leaf E1 adapter. Gated variants have
        # a different non-persistent coverage mask and must be reconstructed by
        # their dedicated evaluator rather than silently treated as E1.
        adapter = LowRankSemanticCalibration(backbone.embedding_dim).to(device)
        checkpoint = torch.load(args.adapter_checkpoint, map_location=device, weights_only=True)
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        adapter.eval()
        print(f"loaded_adapter_checkpoint={args.adapter_checkpoint}", flush=True)
    text_features_by_variant = {
        name: backbone.encode_text(plan.prompts_for_candidate(name)) for name in plan.individual_variant_names
    }
    target_index = plan.leaf_names.index(plan.target_leaf)
    ensemble_features = text_features_by_variant[plan.ensemble_members[0]].clone()
    ensemble_target = torch.stack(
        [text_features_by_variant[name][target_index] for name in plan.ensemble_members]
    ).mean(dim=0)
    ensemble_features[target_index] = F.normalize(ensemble_target, dim=0)
    text_features_by_variant[plan.ensemble_variant_name] = ensemble_features

    node_to_id = stable_node_index(taxonomy.nodes)
    accumulators = {
        name: ExternalTransferAccumulator(
            mapping,
            taxonomy,
            levels,
            node_to_id,
            np.asarray([leaf_index.node_to_id[leaf] for leaf in leaf_index.leaf_names], dtype=np.int64),
            leaf_index.leaf_names,
        )
        for name in plan.variant_names
    }
    for index, pair in enumerate(pairs, start=1):
        raw_mask, predictions = _run_pair_all_variants(
            pair, backbone, adapter, text_features_by_variant, args.tile_size, device
        )
        for name, prediction in predictions.items():
            accumulators[name].update(raw_mask, prediction)
        print(f"evaluated={index}/{len(pairs)} identifier={pair.identifier}", flush=True)
    report = {
        "run_config": run_config,
        "results": {name: accumulator.result().as_dict() for name, accumulator in accumulators.items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "herbaceous_prompt_audit_complete:",
        f"variants={len(plan.variant_names)}",
        f"paired_rasters={len(pairs)}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
