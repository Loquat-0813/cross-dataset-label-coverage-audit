"""Replay a completed OpenEarthMap B2 checkpoint for city-block bootstrap inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1data.external import discover_external_pairs
from p1data.openearthmap_city_split import split_openearthmap_pairs_by_city
from p1data.torch_dataset import TaxonomyRasterDataset
from p1eval.b2_bootstrap import city_block_bootstrap, city_confusions_from_identifier_confusions
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.loveda_e1 import evaluate_taxonomy_tiled_by_identifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e0", "e1"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "openearthmap")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--prompt-config", type=Path, default=ROOT / "configs" / "taxonomy_prompts_v0.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_dict(result) -> dict:
    return {
        "mean_iou": result.mean_iou,
        "per_class_iou": result.per_class_iou,
        "valid_pixels": result.valid_pixels,
    }


def _load_checkpoint_config(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config")
    if not isinstance(config, dict) or config.get("experiment") != "openearthmap_b2_e1":
        raise ValueError("--adapter-checkpoint must come from a completed OpenEarthMap B2 E1 run")
    if "adapter_state_dict" not in checkpoint:
        raise ValueError("B2 checkpoint has no adapter state dictionary")
    return config


def _validate_checkpoint_context(checkpoint_config: dict, split_summary: dict, prompts: tuple[str, ...]) -> None:
    expected_split = checkpoint_config.get("city_split")
    if not isinstance(expected_split, dict):
        raise ValueError("B2 checkpoint has no recorded city split")
    required = (
        "train_pair_identifier_set_sha256",
        "validation_pair_identifier_set_sha256",
        "test_pair_identifier_set_sha256",
    )
    for key in required:
        if expected_split.get(key) != split_summary.get(key):
            raise ValueError(f"B2 checkpoint city split mismatch for {key}")
    if tuple(checkpoint_config.get("prompts", ())) != prompts:
        raise ValueError("B2 checkpoint prompt list does not match --prompt-config")


def main() -> int:
    args = parse_args()
    if args.tile_size < 1 or args.tile_size % 16:
        raise SystemExit("--tile-size must be a positive multiple of CLIP's 16-pixel patch size")
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if args.max_test_samples is not None and args.max_test_samples < 1:
        raise SystemExit("--max-test-samples must be positive when set")
    if args.experiment == "e1" and args.adapter_checkpoint is None:
        raise SystemExit("E1 replay requires --adapter-checkpoint")
    if args.experiment == "e0" and args.adapter_checkpoint is not None:
        raise SystemExit("E0 replay does not accept --adapter-checkpoint")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory already contains files: {args.output_dir}; use --overwrite only intentionally")

    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "openearthmap", taxonomy)
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(args.prompt_config, leaves.leaf_names)
    discovery = discover_external_pairs("openearthmap", args.data_root)
    split = split_openearthmap_pairs_by_city(discovery.pairs)
    full_split_summary = split.as_dict()
    checkpoint_config = None
    if args.experiment == "e1":
        checkpoint_config = _load_checkpoint_config(args.adapter_checkpoint)
        _validate_checkpoint_context(checkpoint_config, full_split_summary, prompts)
    test_pairs = split.test_pairs if args.max_test_samples is None else split.test_pairs[: args.max_test_samples]
    evaluated_pair_hash = hashlib.sha256("\n".join(sorted(pair.identifier for pair in test_pairs)).encode("utf-8")).hexdigest()
    run_config = {
        "experiment": f"openearthmap_b2_{args.experiment}_checkpoint_replay",
        "data_root": str(args.data_root),
        "model_id": args.model_id,
        "adapter_checkpoint": None if args.adapter_checkpoint is None else str(args.adapter_checkpoint),
        "adapter_checkpoint_sha256": None if args.adapter_checkpoint is None else _sha256(args.adapter_checkpoint),
        "taxonomy": taxonomy.version,
        "mapping": "dataset-label-mappings-v0",
        "prompt_config": str(args.prompt_config),
        "prompts": list(prompts),
        "tile_size": args.tile_size,
        "full_city_split": full_split_summary,
        "selected_test_pairs": len(test_pairs),
        "evaluated_test_pair_identifier_set_sha256": evaluated_pair_hash,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "source_checkpoint_city_split": None if checkpoint_config is None else checkpoint_config["city_split"],
    }
    if args.dry_run:
        print("openearthmap_b2_replay_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("OpenEarthMap B2 replay is restricted to CUDA")

    test_dataset = TaxonomyRasterDataset(test_pairs, mapping, node_to_id, crop_size=None, seed=19)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    descendant_mask = torch.from_numpy(leaves.descendant_leaf_mask).to(device=device)
    leaf_node_ids = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], device=device)
    adapter = None
    if args.experiment == "e1":
        adapter = LowRankSemanticCalibration(backbone.embedding_dim, rank=int(checkpoint_config["rank"])).to(device)
        checkpoint = torch.load(args.adapter_checkpoint, map_location=device, weights_only=True)
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        adapter.eval()
        print(f"loaded_adapter_checkpoint={args.adapter_checkpoint}", flush=True)
    print(f"replay_started: experiment={args.experiment} test_pairs={len(test_dataset)}", flush=True)
    result, per_identifier_confusion = evaluate_taxonomy_tiled_by_identifier(
        test_dataset,
        backbone,
        adapter,
        text_features,
        descendant_mask,
        leaf_node_ids,
        leaves.leaf_names,
        device,
        args.tile_size,
    )
    identifier_confusions = {identifier: matrix.tolist() for identifier, matrix in per_identifier_confusion.items()}
    city_confusions = city_confusions_from_identifier_confusions(identifier_confusions, leaves.leaf_names)
    bootstrap = city_block_bootstrap(
        city_confusions,
        leaves.leaf_names,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    point = bootstrap["point"]
    if not np.isclose(result.mean_iou, point["mean_iou"], rtol=0.0, atol=1e-12):
        raise RuntimeError("city confusion aggregation does not reproduce the tiled test mIoU")
    report = {"run_config": run_config, "test": _result_dict(result), "city_bootstrap": bootstrap}
    (args.output_dir / "per_raster_confusions.json").write_text(
        json.dumps({"class_names": list(leaves.leaf_names), "confusions": identifier_confusions}), encoding="utf-8"
    )
    (args.output_dir / "city_confusions.json").write_text(
        json.dumps(
            {
                "class_names": list(leaves.leaf_names),
                "city_confusions": {city: matrix.tolist() for city, matrix in city_confusions.items()},
            }
        ),
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "openearthmap_b2_replay_complete:",
        f"experiment={args.experiment}",
        f"test_mean_iou={result.mean_iou:.6f}",
        f"test_rangeland_iou={result.per_class_iou['herbaceous_vegetation']}",
        f"city_count={bootstrap['city_count']}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
