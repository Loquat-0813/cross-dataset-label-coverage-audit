"""Run the OpenEarthMap city-disjoint B2 supervision control."""

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
from p1data.external import discover_external_pairs
from p1data.openearthmap_city_split import split_openearthmap_pairs_by_city
from p1data.torch_dataset import TaxonomyRasterDataset
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.loveda_e1 import evaluate_taxonomy_tiled, train_calibration_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e0", "e1"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "openearthmap")
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
    parser.add_argument("--train-percent", type=int, default=70)
    parser.add_argument("--validation-percent", type=int, default=15)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def city_manifest(split) -> dict:
    return {
        "summary": split.as_dict(),
        "train_cities": list(split.train_cities),
        "validation_cities": list(split.validation_cities),
        "test_cities": list(split.test_cities),
    }


def main() -> int:
    args = parse_args()
    if min(args.crop_size, args.tile_size, args.rank, args.epochs, args.batch_size) < 1:
        raise SystemExit("crop, tile, rank, epoch, and batch sizes must be positive")
    if args.tile_size % 16:
        raise SystemExit("--tile-size must be a multiple of CLIP's 16-pixel patch size")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory already contains files: {args.output_dir}; use --overwrite only intentionally")
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "openearthmap", taxonomy)
    if not mapping.ready_for_scoring:
        raise SystemExit("OpenEarthMap mapping is not ready for scoring")
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(args.prompt_config, leaves.leaf_names)
    discovery = discover_external_pairs("openearthmap", args.data_root)
    split = split_openearthmap_pairs_by_city(discovery.pairs, args.train_percent, args.validation_percent)
    train_dataset = TaxonomyRasterDataset(
        split.train_pairs, mapping, node_to_id, args.crop_size, args.seed, args.max_train_samples
    )
    validation_dataset = TaxonomyRasterDataset(
        split.validation_pairs, mapping, node_to_id, None, args.seed, args.max_validation_samples
    )
    test_dataset = TaxonomyRasterDataset(split.test_pairs, mapping, node_to_id, None, args.seed, args.max_test_samples)
    manifest = city_manifest(split)
    run_config = {
        "experiment": f"openearthmap_b2_{args.experiment}",
        "seed": args.seed,
        "model_id": args.model_id,
        "prompt_config": str(args.prompt_config),
        "prompts": list(prompts),
        "taxonomy": taxonomy.version,
        "mapping": "dataset-label-mappings-v0",
        "data_root": str(args.data_root),
        "crop_size": args.crop_size,
        "tile_size": args.tile_size,
        "rank": args.rank,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_percent": args.train_percent,
        "validation_percent": args.validation_percent,
        "city_split": manifest["summary"],
        "selected_train_pairs": len(train_dataset),
        "selected_validation_pairs": len(validation_dataset),
        "selected_test_pairs": len(test_dataset),
        "audited_mask_count": discovery.audited_mask_count,
        "paired_raster_count": discovery.paired_raster_count,
        "unpaired_label_count": len(discovery.unpaired_mask_identifiers),
        "unpaired_label_examples": list(discovery.unpaired_mask_identifiers[:5]),
        "test_data_policy": "city_disjoint_from_optimization_and_validation",
    }
    if args.dry_run:
        print("openearthmap_b2_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("OpenEarthMap B2 training is restricted to CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.output_dir / "city_split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    seed_everything(args.seed)
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    descendant_mask = torch.from_numpy(leaves.descendant_leaf_mask).to(device=device)
    leaf_node_ids = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], device=device)
    adapter = None
    history: list[dict] = []
    if args.experiment == "e1":
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
        for epoch in range(args.epochs):
            train_dataset.set_epoch(epoch)
            loss = train_calibration_epoch(loader, backbone, adapter, text_features, descendant_mask, optimizer, device)
            history.append({"epoch": epoch + 1, "train_loss": loss})
            print(f"epoch={epoch + 1}/{args.epochs} train_loss={loss:.6f}", flush=True)
        torch.save(
            {
                "adapter_state_dict": adapter.state_dict(),
                "run_config": run_config,
                "trainable_parameters": adapter.trainable_parameter_count,
            },
            args.output_dir / "adapter_final.pt",
        )
    validation = evaluate_taxonomy_tiled(
        validation_dataset, backbone, adapter, text_features, descendant_mask, leaf_node_ids, leaves.leaf_names, device, args.tile_size
    )
    test = evaluate_taxonomy_tiled(
        test_dataset, backbone, adapter, text_features, descendant_mask, leaf_node_ids, leaves.leaf_names, device, args.tile_size
    )
    report = {"run_config": run_config, "history": history, "validation": metric_dict(validation), "test": metric_dict(test)}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "openearthmap_b2_complete:",
        f"experiment={args.experiment}",
        f"test_mean_iou={test.mean_iou:.6f}",
        f"test_rangeland_iou={test.per_class_iou['herbaceous_vegetation']}",
        f"test_valid_pixels={test.valid_pixels}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
