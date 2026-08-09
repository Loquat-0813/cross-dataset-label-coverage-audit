"""Independent artifact audit for the shuffled-pairing and Semantic Drone gates."""

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

from p1eval.b2_bootstrap import paired_city_bootstrap_mean_difference
from p1eval.semantic_drone_bootstrap import (
    aggregate_grass_counts,
    paired_raster_bootstrap_mean_difference,
)
from p1eval.supported_leaf_bootstrap import paired_city_bootstrap_mean_supported_leaf_difference


SHUFFLE_MODE = "post_primary_reviewer_driven_shuffled_pairing_robustness_control"
FIXED_MODE = "fixed_budget_primary"
EXPOSURE_MODE = "post_primary_loveda_exposure_control"
SEEDS = (19, 37, 73)
EXPECTED_SOURCE_SHA256 = {
    "p1train/flair_shuffle.py": "a43669d0b02f9fc50809448e430b5571ee791c2019538b44303bc9cd23d09397",
    "p1train/fixed_budget.py": "f25e84861dadfa9c72bdb759314fcc88e4455e1a11cfa21b1092bbc6357ade8a",
    "p1data/torch_dataset.py": "45d0fa6502eb96c8a7e99045ff8164583b87c758c93fefc20eafe86759b8f3d3",
    "p1eval/semantic_drone_bootstrap.py": "cbbd42e7a4ac4f898983aab78fbcc98e04e479ffa85a213d20434aa3587b5383",
    "scripts/train_loveda_flair_b1.py": "e5b471514c02f87a6dba11444db75ddb2bdcb80bca40b4777884bd97619ee286",
    "scripts/evaluate_openearthmap_flair_b1_replay.py": "2d9c06fd28e2595b8fac632ba6185ba8ffc97147a025a51d5915d7ef46140784",
    "scripts/summarize_flair_shuffle_bootstrap.py": "a8f48e71b8fb56df0793562d84d3b2e406f63baabe388e09aadafe9caa293158",
    "scripts/summarize_semantic_drone_confirmation_bootstrap.py": "1a4b0946786c7a08f02242ac837bd1bf535d2983c88a355179ea12da34c287dd",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _close(left: float, right: float, label: str) -> None:
    if not np.isclose(left, right, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label}: {left} != {right}")


def _audit_source_code() -> dict[str, str]:
    """Verify the exact code set used for the new control against released hashes."""
    observed: dict[str, str] = {}
    for relative_path, expected in EXPECTED_SOURCE_SHA256.items():
        actual = _sha256(ROOT / relative_path)
        if actual != expected:
            raise ValueError(f"source SHA-256 mismatch for {relative_path}: {actual} != {expected}")
        observed[relative_path] = actual
    return observed


def _checkpoint_config(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config")
    if not isinstance(config, dict) or "adapter_state_dict" not in checkpoint:
        raise ValueError(f"{path}: not a completed adapter checkpoint")
    return config


def _audit_shuffle_training(directory: Path, seed: int) -> dict:
    run_config = _json(directory / "run_config.json")
    metrics = _json(directory / "metrics.json")
    checkpoint_path = directory / "adapter_final.pt"
    checkpoint_config = _checkpoint_config(checkpoint_path)
    if run_config != metrics.get("run_config") or run_config != checkpoint_config:
        raise ValueError(f"{directory}: run configuration differs across output files")
    if run_config.get("experiment") != "flair_coverage_b1_shuffle" or run_config.get("arm") != "b1_shuffle":
        raise ValueError(f"{directory}: not a B1-shuffle training output")
    if run_config.get("protocol_mode") != SHUFFLE_MODE or run_config.get("seed") != seed:
        raise ValueError(f"{directory}: protocol mode or seed mismatch")
    if run_config.get("updates") != 25_220 or run_config.get("loveda_crop_exposures_total") != 25_220:
        raise ValueError(f"{directory}: fixed update/exposure budget mismatch")
    schedule_info = run_config.get("flair_shuffle_schedule")
    if not isinstance(schedule_info, dict):
        raise ValueError(f"{directory}: missing shuffle schedule metadata")
    schedule_path = directory / str(schedule_info.get("path", ""))
    if _sha256(schedule_path) != schedule_info.get("sha256"):
        raise ValueError(f"{directory}: shuffle schedule SHA-256 mismatch")
    schedule = _json(schedule_path)
    records = schedule.get("records")
    permutation = schedule.get("permutation")
    if not isinstance(records, list) or not isinstance(permutation, list) or len(records) != 25_220:
        raise ValueError(f"{directory}: invalid shuffle schedule cardinality")
    if sorted(permutation) != list(range(len(records))):
        raise ValueError(f"{directory}: mask schedule is not a permutation")
    differences: list[float] = []
    for image_position, mask_position in enumerate(permutation):
        image = records[image_position]
        mask = records[mask_position]
        if image["source_identifier"] == mask["source_identifier"]:
            raise ValueError(f"{directory}: native image-mask pairing at update {image_position}")
        if image["domain"] != mask["domain"]:
            raise ValueError(f"{directory}: cross-domain mask pairing at update {image_position}")
        differences.append(abs(float(image["id10_density"]) - float(mask["id10_density"])))
    if sorted(record["id10_pixels"] for record in records) != sorted(records[position]["id10_pixels"] for position in permutation):
        raise ValueError(f"{directory}: ID-10 pixel-count multiset changed")
    _close(float(schedule["max_abs_density_difference"]), max(differences), f"{directory}: max density difference")
    _close(float(schedule["mean_abs_density_difference"]), float(np.mean(differences)), f"{directory}: mean density difference")
    _close(float(schedule_info["max_abs_density_difference"]), max(differences), f"{directory}: config max density difference")
    _close(float(schedule_info["mean_abs_density_difference"]), float(np.mean(differences)), f"{directory}: config mean density difference")
    return {
        "seed": seed,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "schedule_sha256": _sha256(schedule_path),
        "schedule_records": len(records),
        "merged_singleton_bins": len(schedule.get("merged_singleton_bins", [])),
        "max_abs_density_difference": max(differences),
        "mean_abs_density_difference": float(np.mean(differences)),
    }


def _load_oem_replay(directory: Path, arm: str, expected_mode: str, seed: int, checkpoint_sha256: str) -> tuple[dict, tuple[str, ...], dict]:
    report = _json(directory / "metrics.json")
    config = report.get("run_config")
    if not isinstance(config, dict) or config.get("arm") != arm:
        raise ValueError(f"{directory}: replay arm mismatch")
    source = config.get("source_checkpoint_run_config")
    if not isinstance(source, dict) or source.get("protocol_mode") != expected_mode or source.get("seed") != seed:
        raise ValueError(f"{directory}: source checkpoint protocol/seed mismatch")
    if config.get("adapter_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{directory}: replay checkpoint SHA-256 mismatch")
    payload = _json(directory / "city_confusions.json")
    names = tuple(payload.get("class_names", ()))
    cities = payload.get("city_confusions")
    if not names or not isinstance(cities, dict) or report.get("city_bootstrap", {}).get("city_count") != 74:
        raise ValueError(f"{directory}: invalid OEM city-confusion payload")
    if tuple(report["city_bootstrap"]["city_names"]) != tuple(sorted(cities)):
        raise ValueError(f"{directory}: city manifest does not match confusion payload")
    return report, names, cities


def _audit_shuffle(root: Path) -> dict:
    training = [_audit_shuffle_training(root / f"flair_b1_b1_shuffle_seed{seed}", seed) for seed in SEEDS]
    shuffled = [
        _load_oem_replay(
            root / f"flair_b1_b1_shuffle_oem_no_paris_seed{item['seed']}", "b1_shuffle", SHUFFLE_MODE, item["seed"], item["checkpoint_sha256"]
        )
        for item in training
    ]
    b1 = [_load_oem_replay(root / f"flair_b1_b1_oem_no_paris_seed{seed}", "b1", FIXED_MODE, seed, _sha256(root / f"flair_b1_b1_seed{seed}" / "adapter_final.pt")) for seed in SEEDS]
    b0_half = [_load_oem_replay(root / f"flair_b1_b0_half_oem_no_paris_seed{seed}", "b0_half", EXPOSURE_MODE, seed, _sha256(root / f"flair_b1_b0_half_seed{seed}" / "adapter_final.pt")) for seed in SEEDS]
    _, names, reference_cities = b1[0]
    reference_hash = b1[0][0]["run_config"]["evaluated_pair_identifier_set_sha256"]
    for report, candidate_names, cities in (*b1, *b0_half, *shuffled):
        if candidate_names != names or tuple(sorted(cities)) != tuple(sorted(reference_cities)):
            raise ValueError("OEM arms have different class names or city populations")
        if report["run_config"].get("evaluated_pair_identifier_set_sha256") != reference_hash:
            raise ValueError("OEM arms have different target fingerprints")
    summary = _json(root / "flair_shuffled_pairing_bootstrap_summary.json")
    if summary.get("evaluated_pair_identifier_set_sha256") != reference_hash or summary.get("city_count") != 74:
        raise ValueError("shuffle summary target population mismatch")
    supported = tuple(name for name in names if name != "herbaceous_vegetation")
    b1_cities = [item[2] for item in b1]
    shuffled_cities = [item[2] for item in shuffled]
    expected_overall = paired_city_bootstrap_mean_difference(shuffled_cities, b1_cities, names, replicates=2000, seed=20260806)
    expected_supported = paired_city_bootstrap_mean_supported_leaf_difference(
        shuffled_cities, b1_cities, names, supported, replicates=2000, seed=20260806
    )
    reported = summary.get("contrasts", {}).get("b1_minus_b1_shuffle", {})
    reported_overall = reported.get("overall", {})
    reported_supported = reported.get("supported_leaf", {})
    herbaceous = "herbaceous_vegetation"
    _close(
        expected_overall["point_difference"]["per_class_iou"][herbaceous],
        reported_overall["point_difference"]["per_class_iou"][herbaceous],
        "shuffle rangeland point difference",
    )
    for bound in ("lower", "upper"):
        _close(
            expected_overall["confidence_interval"]["per_class_iou"][herbaceous][bound],
            reported_overall["confidence_interval"]["per_class_iou"][herbaceous][bound],
            f"shuffle rangeland CI {bound}",
        )
        _close(
            expected_supported["confidence_interval"][bound], reported_supported["confidence_interval"][bound], f"shuffle supported CI {bound}"
        )
    criterion = summary.get("primary_support_criterion", {})
    seedwise = [
        b1_item[0]["exact_leaf"]["per_class_iou"][herbaceous] - shuffle_item[0]["exact_leaf"]["per_class_iou"][herbaceous]
        for b1_item, shuffle_item in zip(b1, shuffled)
    ]
    if criterion.get("ci_lower_gt_zero") is not True or criterion.get("all_three_seed_differences_positive") is not True:
        raise ValueError("shuffle primary support criterion is not satisfied")
    if not all(value > 0.0 for value in seedwise):
        raise ValueError("shuffle seedwise direction criterion is not satisfied")
    return {
        "status": "pass",
        "training": training,
        "rangeland_b1_minus_shuffle": expected_overall["point_difference"]["per_class_iou"][herbaceous],
        "rangeland_ci": expected_overall["confidence_interval"]["per_class_iou"][herbaceous],
        "supported_leaf_difference": expected_supported["point_difference"],
        "supported_leaf_ci": expected_supported["confidence_interval"],
        "seedwise_rangeland_differences": seedwise,
    }


def _load_semantic(directory: Path, arm: str, seed: int | None) -> tuple[dict, dict]:
    report = _json(directory / "metrics.json")
    values = _json(directory / "per_identifier.json")
    config = report.get("run_config")
    if not isinstance(config, dict) or config.get("experiment") != f"semantic_drone_confirmation_{arm}":
        raise ValueError(f"{directory}: Semantic Drone arm mismatch")
    if seed is not None:
        checkpoint = config.get("checkpoint_run_config")
        if not isinstance(checkpoint, dict) or checkpoint.get("seed") != seed or checkpoint.get("protocol_mode") != FIXED_MODE:
            raise ValueError(f"{directory}: checkpoint seed/protocol mismatch")
    aggregate = aggregate_grass_counts(values)
    if aggregate != report.get("grass"):
        raise ValueError(f"{directory}: stored raster counts do not reproduce reported grass counts")
    return report, values


def _audit_semantic_drone(root: Path) -> dict:
    e0_report, e0 = _load_semantic(root / "semantic_drone_confirmation_e0_full", "e0", None)
    b0 = [_load_semantic(root / f"semantic_drone_confirmation_b0{suffix}", "b0", seed) for suffix, seed in (("_full", 19), ("_seed37_full", 37), ("_seed73_full", 73))]
    b1 = [_load_semantic(root / f"semantic_drone_confirmation_b1{suffix}", "b1", seed) for suffix, seed in (("_full", 19), ("_seed37_full", 37), ("_seed73_full", 73))]
    reports = [e0_report, *(item[0] for item in b0), *(item[0] for item in b1)]
    manifest_hashes = {report["run_config"].get("manifest_sha256") for report in reports}
    if len(manifest_hashes) != 1 or None in manifest_hashes:
        raise ValueError("Semantic Drone outputs do not share one frozen manifest")
    summary = _json(root / "semantic_drone_confirmation_bootstrap_3seed.json")
    expected = paired_raster_bootstrap_mean_difference(
        [e0] * 3, [item[1] for item in b1], replicates=2000, seed=20260804
    )
    expected_b1_b0 = paired_raster_bootstrap_mean_difference(
        [item[1] for item in b0], [item[1] for item in b1], replicates=2000, seed=20260804
    )
    reported = summary.get("comparisons", {}).get("paired_raster_bootstrap_b1_minus_b0", {})
    for field in ("point_difference",):
        _close(expected_b1_b0[field], reported[field], f"Semantic Drone B1-B0 {field}")
    for bound in ("lower", "upper"):
        _close(expected_b1_b0["confidence_interval"][bound], reported["confidence_interval"][bound], f"Semantic Drone B1-B0 CI {bound}")
    if expected_b1_b0["confidence_interval"]["lower"] <= 0.0:
        raise ValueError("Semantic Drone three-seed B1-B0 lower CI is not positive")
    return {
        "status": "pass",
        "manifest_sha256": manifest_hashes.pop(),
        "b1_minus_b0": expected_b1_b0["point_difference"],
        "b1_minus_b0_ci": expected_b1_b0["confidence_interval"],
        "b1_minus_e0": expected["point_difference"],
        "b1_minus_e0_ci": expected["confidence_interval"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise SystemExit(f"output already exists: {args.output_json}")
    payload = {
        "version": "p1-shuffle-semantic-drone-independent-audit-v2",
        "audited_source_sha256": _audit_source_code(),
        "shuffle_gate_g2": _audit_shuffle(args.outputs_root),
        "semantic_drone_gate_g3": _audit_semantic_drone(args.outputs_root),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "p1_shuffle_semantic_drone_independent_audit_complete:",
        f"g2_ci=[{payload['shuffle_gate_g2']['rangeland_ci']['lower']:.6f},{payload['shuffle_gate_g2']['rangeland_ci']['upper']:.6f}]",
        f"g3_ci=[{payload['semantic_drone_gate_g3']['b1_minus_b0_ci']['lower']:.6f},{payload['semantic_drone_gate_g3']['b1_minus_b0_ci']['upper']:.6f}]",
        f"output={args.output_json}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
