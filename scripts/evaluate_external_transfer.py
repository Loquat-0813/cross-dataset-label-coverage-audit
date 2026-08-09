"""Evaluate frozen CLIP E0/E1 on audited OpenEarthMap or LandCoverAI rasters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1data.external import RasterPair, discover_external_pairs
from p1eval.external_transfer import ExternalTransferAccumulator
from p1eval.metrics import stable_node_index
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.loveda_e1 import base_leaf_logits, source_supervised_leaf_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("openearthmap", "landcoverai"), required=True)
    parser.add_argument("--experiment", choices=("e0", "e1", "e1_gated", "e2_fixedscale_gated"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_leaf_prompts(leaf_names: tuple[str, ...]) -> tuple[str, ...]:
    document = yaml.safe_load((ROOT / "configs" / "taxonomy_prompts_v0.yaml").read_text(encoding="utf-8"))
    configured = document["leaf_prompts"]
    if set(configured) != set(leaf_names):
        raise ValueError("prompt configuration must contain every and only taxonomy leaf")
    return tuple(configured[name] for name in leaf_names)


def predict_pair(
    pair: RasterPair,
    backbone: FrozenCLIPPatchEncoder,
    adapter: LowRankSemanticCalibration | None,
    text_features: torch.Tensor,
    tile_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run non-overlapping, padded RGB tiles and return raw mask plus leaf predictions."""
    with Image.open(pair.image_path) as image_file, Image.open(pair.mask_path) as mask_file:
        image = image_file.convert("RGB")
        raw_mask = np.asarray(mask_file)
        if raw_mask.ndim != 2:
            raise ValueError(f"{pair.identifier}: expected one-channel mask, got {raw_mask.shape}")
        if image.size != mask_file.size:
            raise ValueError(f"{pair.identifier}: image size {image.size} does not match mask size {mask_file.size}")
        width, height = image.size
        prediction = np.empty((height, width), dtype=np.int64)
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                rgb = np.asarray(image.crop((left, top, right, bottom)), dtype=np.uint8)
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[: bottom - top, : right - left] = rgb
                tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device)
                features = backbone.encode_image(tensor)
                logits = base_leaf_logits(features, text_features) if adapter is None else adapter(features, text_features)
                logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                prediction[top:bottom, left:right] = logits.argmax(dim=1)[0, : bottom - top, : right - left].cpu().numpy()
    return np.array(raw_mask, copy=True), prediction


def main() -> int:
    args = parse_args()
    if args.tile_size < 1 or args.tile_size % 16:
        raise SystemExit("--tile-size must be a positive multiple of CLIP's 16-pixel patch size")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive when set")
    if args.experiment in {"e1", "e1_gated", "e2_fixedscale_gated"} and args.adapter_checkpoint is None:
        raise SystemExit("E1, E1-gated, and E2-fixedscale-gated external evaluation require --adapter-checkpoint")
    if args.experiment == "e0" and args.adapter_checkpoint is not None:
        raise SystemExit("--adapter-checkpoint is only valid for E1 or E1-gated")

    data_root = args.data_root or ROOT / "data" / "raw" / args.dataset
    taxonomy, levels = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", args.dataset, taxonomy)
    if not mapping.ready_for_scoring:
        raise SystemExit(f"{args.dataset}: mapping is not ready for scoring")
    discovery = discover_external_pairs(args.dataset, data_root)
    pairs = discovery.pairs
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    node_to_id = stable_node_index(taxonomy.nodes)
    leaf_index = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(leaf_index.leaf_names)
    gated_mask = None
    if args.experiment in {"e1_gated", "e2_fixedscale_gated"}:
        source_mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
        gated_mask = source_supervised_leaf_mask(source_mapping, leaf_index.leaf_names)
    run_config = {
        "dataset": args.dataset,
        "experiment": args.experiment,
        "data_root": str(data_root),
        "model_id": args.model_id,
        "adapter_checkpoint": None if args.adapter_checkpoint is None else str(args.adapter_checkpoint),
        "taxonomy": taxonomy.version,
        "mapping": "dataset-label-mappings-v0",
        "tile_size": args.tile_size,
        "audited_mask_count": discovery.audited_mask_count,
        "paired_raster_count": discovery.paired_raster_count,
        "unpaired_label_count": len(discovery.unpaired_mask_identifiers),
        "unpaired_label_examples": list(discovery.unpaired_mask_identifiers[:5]),
        "selected_raster_count": len(pairs),
        "leaf_names": list(leaf_index.leaf_names),
        "prompts": list(prompts),
        "calibration_leaf_mode": (
            "source_exact_leaves_fixed_temperature"
            if args.experiment == "e2_fixedscale_gated"
            else "source_exact_leaves_only"
            if gated_mask is not None
            else "all_leaves"
        ),
        "freeze_logit_scale": args.experiment == "e2_fixedscale_gated",
        "adapted_leaf_names": list(leaf_index.leaf_names)
        if gated_mask is None
        else [name for name, adapted in zip(leaf_index.leaf_names, gated_mask.tolist()) if adapted],
        "frozen_leaf_names": []
        if gated_mask is None
        else [name for name, adapted in zip(leaf_index.leaf_names, gated_mask.tolist()) if not adapted],
    }
    if args.dry_run:
        print("external_transfer_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("external transfer evaluation is restricted to CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory already contains files: {args.output_dir}; use --overwrite only intentionally")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    adapter = None
    if args.experiment in {"e1", "e1_gated", "e2_fixedscale_gated"}:
        adapter = LowRankSemanticCalibration(
            backbone.embedding_dim,
            adapted_leaf_mask=gated_mask,
            freeze_logit_scale=args.experiment == "e2_fixedscale_gated",
        ).to(device)
        checkpoint = torch.load(args.adapter_checkpoint, map_location=device, weights_only=True)
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        adapter.eval()
        print(f"loaded_adapter_checkpoint={args.adapter_checkpoint}", flush=True)

    accumulator = ExternalTransferAccumulator(
        mapping,
        taxonomy,
        levels,
        node_to_id,
        np.asarray([leaf_index.node_to_id[name] for name in leaf_index.leaf_names], dtype=np.int64),
        leaf_index.leaf_names,
    )
    for index, pair in enumerate(pairs, start=1):
        raw_mask, prediction = predict_pair(pair, backbone, adapter, text_features, args.tile_size, device)
        accumulator.update(raw_mask, prediction)
        print(f"evaluated={index}/{len(pairs)} identifier={pair.identifier}", flush=True)
    result = accumulator.result()
    report = {"run_config": run_config, "metrics": result.as_dict()}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "external_transfer_complete:",
        f"dataset={args.dataset}",
        f"experiment={args.experiment}",
        f"exact_macro_iou={result.exact_leaf.macro_iou}",
        f"exact_valid_pixels={result.exact_leaf.valid_pixels}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
