"""Evaluate one fixed B0/B1 checkpoint on all paired OpenEarthMap rasters."""

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
from p1data.openearthmap_city_split import (
    city_from_openearthmap_identifier,
    exclude_openearthmap_cities,
    load_openearthmap_test_city_manifest,
    select_openearthmap_cities,
)
from p1data.splits import identifier_set_sha256
from p1data.torch_dataset import TaxonomyRasterDataset
from p1eval.b2_bootstrap import city_block_bootstrap, city_confusions_from_identifier_confusions
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.loveda_e1 import evaluate_taxonomy_tiled_by_identifier


REQUIRED_EXCLUDED_CITIES = ("paris",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("b0", "b0_half", "b0_half_batch2", "b1", "b1_shuffle"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "openearthmap")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, default=ROOT / "configs" / "taxonomy_prompts_v0.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--exclude-city", action="append", default=[])
    parser.add_argument(
        "--include-city-manifest",
        type=Path,
        help="Frozen B2 city_split.json used for an inference-only aligned target replay.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_config(path: Path, arm: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config")
    if not isinstance(config, dict) or config.get("experiment") != f"flair_coverage_{arm}":
        raise ValueError(f"--adapter-checkpoint must be a completed FLAIR {arm.upper()} checkpoint")
    allowed_modes = {
        "b0": "fixed_budget_primary",
        "b0_half": "post_primary_loveda_exposure_control",
        "b0_half_batch2": "post_primary_loveda_exposure_matched_batch_size_control",
        "b1": "fixed_budget_primary",
        "b1_shuffle": "post_primary_reviewer_driven_shuffled_pairing_robustness_control",
    }
    if config.get("protocol_mode") != allowed_modes[arm]:
        raise ValueError("checkpoint does not have the required full-budget protocol status for this replay arm")
    if "adapter_state_dict" not in checkpoint:
        raise ValueError("checkpoint has no adapter state dictionary")
    return config


def _ready_audit(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint source audit is missing: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("source_mapping_status") != "source_audit_ready":
        raise ValueError("checkpoint source audit is not approved for B1 evaluation")
    return audit


def main() -> int:
    args = parse_args()
    if args.tile_size < 1 or args.tile_size % 16:
        raise SystemExit("--tile-size must be a positive multiple of CLIP's 16-pixel patch size")
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive when set")
    excluded_cities = tuple(sorted(set(args.exclude_city)))
    if excluded_cities != REQUIRED_EXCLUDED_CITIES:
        raise SystemExit(f"FLAIR B1 evaluation requires exactly --exclude-city {REQUIRED_EXCLUDED_CITIES[0]}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory already contains files: {args.output_dir}")
    config = _checkpoint_config(args.adapter_checkpoint, args.arm)
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "openearthmap", taxonomy)
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    prompts = load_leaf_prompts(args.prompt_config, leaves.leaf_names)
    if tuple(config.get("prompts", ())) != prompts:
        raise ValueError("checkpoint prompts do not match the frozen prompt configuration")
    source_audit = _ready_audit(Path(config["source_audit"]))
    discovery = discover_external_pairs("openearthmap", args.data_root)
    geographic_eligible_pairs = exclude_openearthmap_cities(discovery.pairs, excluded_cities)
    eligible_hash = identifier_set_sha256(pair.identifier for pair in geographic_eligible_pairs)
    expected_hash = source_audit.get("geographic_overlap_review", {}).get("eligible_pair_identifier_set_sha256")
    if eligible_hash != expected_hash:
        raise ValueError("OpenEarthMap eligible-pair fingerprint disagrees with the approved source overlap audit")
    target_selection: dict[str, object] = {
        "mode": "all_geographic_guard_eligible_pairs",
        "geographic_guard_eligible_pairs": len(geographic_eligible_pairs),
        "geographic_guard_eligible_pair_identifier_set_sha256": eligible_hash,
    }
    selected_before_geographic_guard = discovery.pairs
    if args.include_city_manifest is not None:
        city_manifest = load_openearthmap_test_city_manifest(args.include_city_manifest)
        selected_before_geographic_guard = select_openearthmap_cities(discovery.pairs, city_manifest.test_cities)
        selected_before_guard_hash = identifier_set_sha256(pair.identifier for pair in selected_before_geographic_guard)
        if len(selected_before_geographic_guard) != city_manifest.declared_test_pairs:
            raise ValueError("B2 test-city manifest test-pair count disagrees with this OpenEarthMap release")
        if selected_before_guard_hash != city_manifest.declared_test_pair_identifier_set_sha256:
            raise ValueError("B2 test-city manifest test-pair fingerprint disagrees with this OpenEarthMap release")
        target_selection = {
            "mode": "b2_frozen_test_city_manifest",
            "source_city_manifest": str(city_manifest.path),
            "source_city_manifest_sha256": _sha256(city_manifest.path),
            "b2_test_city_set_sha256": city_manifest.test_city_set_sha256,
            "b2_declared_test_pairs": city_manifest.declared_test_pairs,
            "b2_declared_test_pair_identifier_set_sha256": city_manifest.declared_test_pair_identifier_set_sha256,
            "requested_cities": list(city_manifest.test_cities),
            "selected_pairs_before_geographic_guard": len(selected_before_geographic_guard),
            "selected_pair_identifier_set_sha256_before_geographic_guard": selected_before_guard_hash,
        }
    selected_pairs = exclude_openearthmap_cities(selected_before_geographic_guard, excluded_cities)
    excluded_from_selection = len(selected_before_geographic_guard) - len(selected_pairs)
    if not selected_pairs:
        raise ValueError("geographic guard removed every target pair selected for replay")
    target_selection.update(
        {
            "geographic_guard_excluded_pairs_within_selection": excluded_from_selection,
            "selected_cities_after_geographic_guard": sorted(
                {city_from_openearthmap_identifier(pair.identifier) for pair in selected_pairs}
            ),
            "selected_pairs_after_geographic_guard": len(selected_pairs),
            "selected_pair_identifier_set_sha256_after_geographic_guard": identifier_set_sha256(
                pair.identifier for pair in selected_pairs
            ),
        }
    )
    pairs = selected_pairs if args.max_samples is None else selected_pairs[: args.max_samples]
    run_config = {
        "experiment": f"flair_b1_openearthmap_{args.arm}_replay",
        "arm": args.arm,
        "model_id": args.model_id,
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "adapter_checkpoint_sha256": _sha256(args.adapter_checkpoint),
        "source_checkpoint_run_config": config,
        "data_root": str(args.data_root),
        "paired_raster_count": len(discovery.pairs),
        "geographic_overlap_excluded_cities": list(excluded_cities),
        "eligible_pairs_before_sample_cap": len(selected_pairs),
        "eligible_pair_identifier_set_sha256": eligible_hash,
        "target_selection": target_selection,
        "evaluated_pairs": len(pairs),
        "evaluated_pair_identifier_set_sha256": identifier_set_sha256(pair.identifier for pair in pairs),
        "unpaired_label_count": len(discovery.unpaired_mask_identifiers),
        "tile_size": args.tile_size,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "prompts": list(prompts),
    }
    if args.dry_run:
        print("flair_b1_openearthmap_replay_dry_run_valid:", json.dumps(run_config, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("OpenEarthMap B1 replay is restricted to CUDA")
    dataset = TaxonomyRasterDataset(pairs, mapping, node_to_id, crop_size=None, seed=19)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    descendant_mask = torch.from_numpy(leaves.descendant_leaf_mask).to(device=device)
    leaf_node_ids = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], device=device)
    adapter = LowRankSemanticCalibration(backbone.embedding_dim, rank=int(config["rank"])).to(device)
    checkpoint = torch.load(args.adapter_checkpoint, map_location=device, weights_only=True)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    print(f"loaded_adapter_checkpoint={args.adapter_checkpoint}", flush=True)
    result, per_identifier = evaluate_taxonomy_tiled_by_identifier(
        dataset,
        backbone,
        adapter,
        text_features,
        descendant_mask,
        leaf_node_ids,
        leaves.leaf_names,
        device,
        args.tile_size,
    )
    identifier_confusions = {identifier: matrix.tolist() for identifier, matrix in per_identifier.items()}
    city_confusions = city_confusions_from_identifier_confusions(identifier_confusions, leaves.leaf_names)
    bootstrap = city_block_bootstrap(city_confusions, leaves.leaf_names, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed)
    if not np.isclose(result.mean_iou, bootstrap["point"]["mean_iou"], rtol=0.0, atol=1e-12):
        raise RuntimeError("city aggregation does not reproduce all-raster mIoU")
    report = {
        "run_config": run_config,
        "exact_leaf": {
            "mean_iou": result.mean_iou,
            "per_class_iou": result.per_class_iou,
            "valid_pixels": result.valid_pixels,
        },
        "city_bootstrap": bootstrap,
    }
    (args.output_dir / "per_raster_confusions.json").write_text(
        json.dumps({"class_names": list(leaves.leaf_names), "confusions": identifier_confusions}), encoding="utf-8"
    )
    (args.output_dir / "city_confusions.json").write_text(
        json.dumps({"class_names": list(leaves.leaf_names), "city_confusions": {city: matrix.tolist() for city, matrix in city_confusions.items()}}),
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "flair_b1_openearthmap_replay_complete:",
        f"arm={args.arm}",
        f"mean_iou={result.mean_iou:.6f}",
        f"rangeland_iou={result.per_class_iou['herbaceous_vegetation']}",
        f"city_count={bootstrap['city_count']}",
        f"output={args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
