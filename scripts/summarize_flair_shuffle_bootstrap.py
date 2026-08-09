"""Summarize the post-primary FLAIR image/mask-pairing robustness control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1eval.b2_bootstrap import paired_city_bootstrap_mean_difference
from p1eval.supported_leaf_bootstrap import (
    paired_city_bootstrap_mean_supported_leaf_difference,
    supported_leaf_mean_iou,
)


EXPOSURE_MODE = "post_primary_loveda_exposure_control"
SHUFFLE_MODE = "post_primary_reviewer_driven_shuffled_pairing_robustness_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-half-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--b1-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--b1-shuffle-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--exclude-leaf", default="herbaceous_vegetation")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    return parser.parse_args()


def _source_protocol_mode(config: dict) -> str | None:
    source = config.get("source_checkpoint_run_config")
    return source.get("protocol_mode") if isinstance(source, dict) else None


def _source_seed(config: dict) -> int:
    source = config.get("source_checkpoint_run_config")
    seed = source.get("seed") if isinstance(source, dict) else None
    if not isinstance(seed, int):
        raise ValueError("replay lacks an integer source checkpoint seed")
    return seed


def _load(directory: Path, arm: str, expected_mode: str) -> tuple[dict, tuple[str, ...], dict, int]:
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = report.get("run_config", {})
    if config.get("arm") != arm or _source_protocol_mode(config) != expected_mode:
        raise ValueError(f"{directory}: expected {arm} with protocol mode {expected_mode}")
    payload = json.loads((directory / "city_confusions.json").read_text(encoding="utf-8"))
    names = tuple(payload["class_names"])
    cities = payload["city_confusions"]
    if tuple(report["city_bootstrap"]["city_names"]) != tuple(sorted(cities)):
        raise ValueError(f"{directory}: city manifest and city confusions disagree")
    return report, names, cities, _source_seed(config)


def _load_group(directories: list[Path], arm: str, mode: str) -> list[tuple[dict, tuple[str, ...], dict, int]]:
    if len(directories) != 3:
        raise ValueError(f"{arm} requires exactly three replay directories")
    group = sorted((_load(directory, arm, mode) for directory in directories), key=lambda item: item[3])
    seeds = [item[3] for item in group]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{arm} replay directories must use distinct checkpoint seeds")
    return group


def _arm_summary(group: list[tuple[dict, tuple[str, ...], dict, int]], names: tuple[str, ...], supported: tuple[str, ...]) -> dict:
    exact = [item[0]["exact_leaf"] for item in group]
    supported_values = [
        supported_leaf_mean_iou(np.asarray(list(item[2].values()), dtype=np.int64).sum(axis=0), names, supported)
        for item in group
    ]
    return {
        "seeds": [item[3] for item in group],
        "mean_iou": statistics.mean(float(value["mean_iou"]) for value in exact),
        "mean_iou_sample_std": statistics.stdev(float(value["mean_iou"]) for value in exact),
        "per_class_iou": {
            name: statistics.mean(float(value["per_class_iou"][name]) for value in exact) for name in names
        },
        "per_class_iou_sample_std": {
            name: statistics.stdev(float(value["per_class_iou"][name]) for value in exact) for name in names
        },
        "supported_leaf_mean_iou": statistics.mean(supported_values),
        "supported_leaf_mean_iou_sample_std": statistics.stdev(supported_values),
        "per_seed": [
            {
                "seed": item[3],
                "mean_iou": exact_value["mean_iou"],
                "per_class_iou": exact_value["per_class_iou"],
                "supported_leaf_mean_iou": supported_value,
            }
            for item, exact_value, supported_value in zip(group, exact, supported_values)
        ],
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if args.output_json.exists():
        raise SystemExit(f"output already exists: {args.output_json}")
    b0_half = _load_group(args.b0_half_replay_dir, "b0_half", EXPOSURE_MODE)
    b1 = _load_group(args.b1_replay_dir, "b1", "fixed_budget_primary")
    shuffled = _load_group(args.b1_shuffle_replay_dir, "b1_shuffle", SHUFFLE_MODE)
    _, names, reference_cities, reference_seed = b0_half[0]
    reference_hash = b0_half[0][0]["run_config"]["evaluated_pair_identifier_set_sha256"]
    supported = tuple(name for name in names if name != args.exclude_leaf)
    if len(supported) != len(names) - 1:
        raise SystemExit(f"excluded leaf {args.exclude_leaf!r} is absent or duplicated")
    for report, candidate_names, cities, seed in (*b0_half, *b1, *shuffled):
        if candidate_names != names or tuple(sorted(cities)) != tuple(sorted(reference_cities)):
            raise ValueError("arms have different class names or city sets")
        if report["run_config"]["evaluated_pair_identifier_set_sha256"] != reference_hash:
            raise ValueError("arms use different OpenEarthMap raster sets")
    b1_seeds = [item[3] for item in b1]
    shuffle_seeds = [item[3] for item in shuffled]
    if b1_seeds != shuffle_seeds:
        raise ValueError("B1 and B1-shuffle must use the same ordered checkpoint seed set")
    b0_half_cities = [item[2] for item in b0_half]
    b1_cities = [item[2] for item in b1]
    shuffled_cities = [item[2] for item in shuffled]
    def overall(baseline: list[dict], model: list[dict]) -> dict:
        return paired_city_bootstrap_mean_difference(
            baseline, model, names, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
        )
    def supported_contrast(baseline: list[dict], model: list[dict]) -> dict:
        return paired_city_bootstrap_mean_supported_leaf_difference(
            baseline, model, names, supported, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
        )
    contrasts = {
        "b1_minus_b1_shuffle": {
            "overall": overall(shuffled_cities, b1_cities),
            "supported_leaf": supported_contrast(shuffled_cities, b1_cities),
        },
    }
    herbaceous = "herbaceous_vegetation"
    primary = contrasts["b1_minus_b1_shuffle"]["overall"]
    primary_ci = primary["confidence_interval"]["per_class_iou"][herbaceous]
    summary = {
        "version": "flair-shuffled-pairing-bootstrap-v1",
        "evaluated_pair_identifier_set_sha256": reference_hash,
        "city_count": len(reference_cities),
        "excluded_leaf": args.exclude_leaf,
        "supported_leaves": list(supported),
        "primary_support_criterion": {
            "contrast": "b1_minus_b1_shuffle",
            "metric": herbaceous,
            "requirement": "paired_city_bootstrap_ci_lower_gt_zero_and_all_three_seed_differences_positive",
            "ci_lower": primary_ci["lower"],
            "ci_lower_gt_zero": bool(primary_ci["lower"] > 0.0),
            "seedwise_differences": [
                b1_value["per_class_iou"][herbaceous] - shuffle_value["per_class_iou"][herbaceous]
                for b1_value, shuffle_value in zip(
                    _arm_summary(b1, names, supported)["per_seed"], _arm_summary(shuffled, names, supported)["per_seed"]
                )
            ],
        },
        "arms": {
            "b0_half": _arm_summary(b0_half, names, supported),
            "b1": _arm_summary(b1, names, supported),
            "b1_shuffle": _arm_summary(shuffled, names, supported),
        },
        "contrasts": contrasts,
        "source_replay_directories": {
            "b0_half": [str(value) for value in args.b0_half_replay_dir],
            "b1": [str(value) for value in args.b1_replay_dir],
            "b1_shuffle": [str(value) for value in args.b1_shuffle_replay_dir],
        },
    }
    summary["primary_support_criterion"]["all_three_seed_differences_positive"] = all(
        value > 0.0 for value in summary["primary_support_criterion"]["seedwise_differences"]
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "flair_shuffled_pairing_bootstrap_complete:",
        f"b1_minus_shuffle_rangeland={primary['point_difference']['per_class_iou'][herbaceous]:.6f}",
        f"ci=[{primary_ci['lower']:.6f},{primary_ci['upper']:.6f}]",
        f"all_seed_directions_positive={summary['primary_support_criterion']['all_three_seed_differences_positive']}",
        f"output={args.output_json}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
