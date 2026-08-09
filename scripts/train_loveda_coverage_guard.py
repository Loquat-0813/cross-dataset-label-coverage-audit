"""Train a LoveDA adapter with a source-only conformal calibration partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1data.loveda import discover_loveda_pairs
from p1data.splits import identifier_set_sha256, split_identifiers_for_calibration
from p1data.torch_dataset import LoveDATaxonomyDataset
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.conformal_calibration import calibrate_frozen_base_probability_threshold, evaluate_frozen_base_prediction_sets
from p1train.loveda_e1 import evaluate_loveda_tiled, source_supervised_leaf_mask, train_calibration_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "loveda")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--prompt-config", type=Path, default=ROOT / "configs" / "taxonomy_prompts_v0.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--calibration-percent", type=int, default=15)
    parser.add_argument("--calibration-pixels-per-tile", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric_dict(result) -> dict:
    return {
        "mean_iou": result.mean_iou,
        "per_class_iou": result.per_class_iou,
        "valid_pixels": result.valid_pixels,
    }


def main() -> int:
    args = parse_args()
    if min(args.crop_size, args.tile_size, args.batch_size, args.epochs, args.calibration_pixels_per_tile) < 1:
        raise SystemExit("crop, tile, batch, epoch, and calibration pixel counts must be positive")
    if not 0 < args.alpha < 1:
        raise SystemExit("--alpha must lie strictly between zero and one")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(args.prompt_config, leaves.leaf_names)
    pairs = discover_loveda_pairs(args.data_root, "train")
    if args.max_train_samples is not None:
        pairs = pairs[: args.max_train_samples]
    optimization_ids, calibration_ids = split_identifiers_for_calibration(
        (pair.identifier for pair in pairs), args.calibration_percent
    )
    train_dataset = LoveDATaxonomyDataset(
        args.data_root, "train", mapping, node_to_id, args.crop_size, args.seed, include_identifiers=optimization_ids
    )
    calibration_dataset = LoveDATaxonomyDataset(
        args.data_root, "train", mapping, node_to_id, None, args.seed, include_identifiers=calibration_ids
    )
    validation_dataset = LoveDATaxonomyDataset(
        args.data_root, "val", mapping, node_to_id, None, args.seed, args.max_val_samples
    )
    run_config = {
        "experiment": "e3_coverage_guard",
        "seed": args.seed,
        "model_id": args.model_id,
        "prompt_config": str(args.prompt_config),
        "prompts": list(prompts),
        "taxonomy": taxonomy.version,
        "mapping": "dataset-label-mappings-v0",
        "optimization_train_pairs": len(train_dataset),
        "source_calibration_pairs": len(calibration_dataset),
        "optimization_identifier_set_sha256": identifier_set_sha256(optimization_ids),
        "source_calibration_identifier_set_sha256": identifier_set_sha256(calibration_ids),
        "validation_pairs": len(validation_dataset),
        "crop_size": args.crop_size,
        "tile_size": args.tile_size,
        "rank": args.rank,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "calibration_percent": args.calibration_percent,
        "calibration_pixels_per_tile": args.calibration_pixels_per_tile,
        "alpha": args.alpha,
        "covered_leaf_names": [
            name for name, covered in zip(leaves.leaf_names, source_supervised_leaf_mask(mapping, leaves.leaf_names).tolist()) if covered
        ],
        "uncovered_leaf_names": [
            name for name, covered in zip(leaves.leaf_names, source_supervised_leaf_mask(mapping, leaves.leaf_names).tolist()) if not covered
        ],
    }
    if args.dry_run:
        print("coverage_guard_train_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("coverage guard training is restricted to CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    seed_everything(args.seed)
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    leaf_node_ids = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], device=device)
    descendant_mask = torch.from_numpy(leaves.descendant_leaf_mask).to(device=device)
    adapter = LowRankSemanticCalibration(backbone.embedding_dim, args.rank).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    history: list[dict] = []
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        loss = train_calibration_epoch(loader, backbone, adapter, text_features, descendant_mask, optimizer, device)
        history.append({"epoch": epoch + 1, "train_loss": loss})
        print(f"epoch={epoch + 1}/{args.epochs} train_loss={loss:.6f}", flush=True)
    conformal = calibrate_frozen_base_probability_threshold(
        calibration_dataset,
        backbone,
        text_features,
        leaf_node_ids,
        device,
        args.tile_size,
        args.calibration_pixels_per_tile,
        args.alpha,
        args.seed,
    )
    source_calibration_diagnostics = evaluate_frozen_base_prediction_sets(
        calibration_dataset,
        backbone,
        text_features,
        leaf_node_ids,
        device,
        args.tile_size,
        conformal.probability_threshold,
    )
    source_validation_diagnostics = evaluate_frozen_base_prediction_sets(
        validation_dataset,
        backbone,
        text_features,
        leaf_node_ids,
        device,
        args.tile_size,
        conformal.probability_threshold,
    )
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "run_config": run_config,
            "conformal_calibration": conformal.as_dict(),
            "source_conformal_diagnostics": {
                "calibration_partition": source_calibration_diagnostics.as_dict(),
                "loveda_validation": source_validation_diagnostics.as_dict(),
            },
            "trainable_parameters": adapter.trainable_parameter_count,
        },
        args.output_dir / "adapter_final.pt",
    )
    (args.output_dir / "conformal_calibration.json").write_text(json.dumps(conformal.as_dict(), indent=2), encoding="utf-8")
    validation = evaluate_loveda_tiled(
        validation_dataset,
        backbone,
        adapter,
        text_features,
        descendant_mask,
        leaf_node_ids,
        leaves.leaf_names,
        device,
        args.tile_size,
    )
    report = {
        "run_config": run_config,
        "history": history,
        "conformal_calibration": conformal.as_dict(),
        "source_conformal_diagnostics": {
            "calibration_partition": source_calibration_diagnostics.as_dict(),
            "loveda_validation": source_validation_diagnostics.as_dict(),
        },
        "validation": metric_dict(validation),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "coverage_guard_train_complete:",
        f"mean_iou={validation.mean_iou:.6f}",
        f"threshold={conformal.probability_threshold:.6f}",
        f"calibration_scores={conformal.score_count}",
        f"source_validation_coverage={source_validation_diagnostics.coverage:.6f}",
        f"source_validation_mean_set_size={source_validation_diagnostics.mean_set_size:.6f}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
