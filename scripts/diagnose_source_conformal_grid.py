"""Diagnose coverage-guard set sizes on LoveDA only, without target labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1data.loveda import discover_loveda_pairs
from p1data.splits import identifier_set_sha256, split_identifiers_for_calibration
from p1data.torch_dataset import LoveDATaxonomyDataset
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.selective_hierarchy import build_coverage_node_lookup
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.coverage_guard import split_conformal_probability_threshold_from_scores
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.conformal_calibration import (
    collect_frozen_base_nonconformity_scores,
    evaluate_frozen_base_coverage_routing_grid,
)
from p1train.loveda_e1 import source_supervised_leaf_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "loveda")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--prompt-config", type=Path, default=ROOT / "configs" / "taxonomy_prompts_v0.yaml")
    parser.add_argument("--grid-config", type=Path, default=ROOT / "configs" / "source_conformal_grid_v1.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_grid_config(path: Path) -> tuple[str, tuple[float, ...], int, int]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    version = str(document["version"])
    alphas = tuple(float(value) for value in document["alpha_candidates"])
    calibration_percent = int(document["calibration_percent"])
    pixels_per_tile = int(document["calibration_pixels_per_tile"])
    if version != "source-conformal-grid-v1":
        raise ValueError(f"unsupported grid configuration version: {version}")
    if not alphas or len(set(alphas)) != len(alphas) or any(not 0.0 < value < 1.0 for value in alphas):
        raise ValueError("alpha_candidates must be unique values strictly between zero and one")
    if not 1 <= calibration_percent < 100 or pixels_per_tile < 1:
        raise ValueError("grid calibration settings are invalid")
    return version, alphas, calibration_percent, pixels_per_tile


def main() -> int:
    args = parse_args()
    if args.tile_size < 1 or args.tile_size % 16:
        raise SystemExit("--tile-size must be a positive multiple of CLIP's 16-pixel patch size")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")
    version, alphas, calibration_percent, pixels_per_tile = load_grid_config(args.grid_config)
    taxonomy, levels = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(args.prompt_config, leaves.leaf_names)
    pairs = discover_loveda_pairs(args.data_root, "train")
    if args.max_train_samples is not None:
        if args.max_train_samples < 1:
            raise SystemExit("--max-train-samples must be positive when set")
        pairs = pairs[: args.max_train_samples]
    _, calibration_ids = split_identifiers_for_calibration((pair.identifier for pair in pairs), calibration_percent)
    calibration_dataset = LoveDATaxonomyDataset(
        args.data_root, "train", mapping, node_to_id, None, args.seed, include_identifiers=calibration_ids
    )
    validation_dataset = LoveDATaxonomyDataset(
        args.data_root, "val", mapping, node_to_id, None, args.seed, args.max_val_samples
    )
    run_config = {
        "experiment": "source_conformal_grid_diagnostic",
        "grid_config": str(args.grid_config),
        "grid_version": version,
        "seed": args.seed,
        "model_id": args.model_id,
        "prompt_config": str(args.prompt_config),
        "prompts": list(prompts),
        "taxonomy": taxonomy.version,
        "mapping": "dataset-label-mappings-v0",
        "tile_size": args.tile_size,
        "alpha_candidates": list(alphas),
        "calibration_percent": calibration_percent,
        "calibration_pixels_per_tile": pixels_per_tile,
        "source_calibration_pairs": len(calibration_dataset),
        "source_calibration_identifier_set_sha256": identifier_set_sha256(calibration_ids),
        "source_validation_pairs": len(validation_dataset),
        "target_labels_read": False,
    }
    if args.dry_run:
        print("source_conformal_grid_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("source conformal grid diagnostic is restricted to CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    leaf_node_ids = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], dtype=torch.long, device=device)
    scores, sampled_tiles = collect_frozen_base_nonconformity_scores(
        calibration_dataset,
        backbone,
        text_features,
        leaf_node_ids,
        device,
        args.tile_size,
        pixels_per_tile,
        args.seed,
    )
    threshold_by_key = {
        f"alpha={alpha:.2f}": split_conformal_probability_threshold_from_scores(scores, alpha) for alpha in alphas
    }
    covered_mask = source_supervised_leaf_mask(mapping, leaves.leaf_names).to(device)
    lookup = build_coverage_node_lookup(
        taxonomy,
        levels,
        node_to_id,
        leaves.leaf_names,
        covered_mask.cpu().numpy().astype(np.bool_, copy=False),
    )
    node_depth_by_id = torch.tensor([levels[name] for name in taxonomy.nodes], dtype=torch.long, device=device)
    grid = evaluate_frozen_base_coverage_routing_grid(
        validation_dataset,
        backbone,
        text_features,
        leaf_node_ids,
        device,
        args.tile_size,
        threshold_by_key,
        covered_mask,
        torch.from_numpy(lookup).to(device),
        node_depth_by_id,
    )
    report = {
        "run_config": run_config,
        "calibration_score_count": int(scores.numel()),
        "sampled_tiles": sampled_tiles,
        "probability_thresholds": threshold_by_key,
        "source_validation": {key: result.as_dict() for key, result in grid.items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, result in grid.items():
        metrics = result.as_dict()
        print(
            "source_conformal_grid_complete:",
            key,
            f"threshold={threshold_by_key[key]:.6f}",
            f"coverage={metrics['coverage']:.6f}",
            f"mean_set_size={metrics['mean_set_size']:.6f}",
            f"route_rate={metrics['route_rate']:.6f}",
            f"root_route_rate={metrics['root_route_rate']:.6f}",
            flush=True,
        )
    print(f"source_conformal_grid_output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
