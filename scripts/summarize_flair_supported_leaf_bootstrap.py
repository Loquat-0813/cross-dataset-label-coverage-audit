"""Summarize OpenEarthMap supported-leaf mIoU for the B0/B0-half/B1 controls."""

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

from p1eval.supported_leaf_bootstrap import (
    paired_city_bootstrap_mean_supported_leaf_difference,
    supported_leaf_mean_iou,
)


EXPOSURE_MODE = "post_primary_loveda_exposure_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--b0-half-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--b1-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--exclude-leaf", default="herbaceous_vegetation")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    return parser.parse_args()


def _source_protocol_mode(config: dict) -> str | None:
    source = config.get("source_checkpoint_run_config")
    return source.get("protocol_mode") if isinstance(source, dict) else None


def _load(directory: Path, arm: str, expected_mode: str) -> tuple[dict, tuple[str, ...], dict]:
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = report.get("run_config", {})
    if config.get("arm") != arm or _source_protocol_mode(config) != expected_mode:
        raise ValueError(f"{directory}: expected {arm} with protocol mode {expected_mode}")
    payload = json.loads((directory / "city_confusions.json").read_text(encoding="utf-8"))
    names = tuple(payload["class_names"])
    cities = payload["city_confusions"]
    if tuple(report["city_bootstrap"]["city_names"]) != tuple(sorted(cities)):
        raise ValueError(f"{directory}: city manifest and city confusions disagree")
    return report, names, cities


def _arm_mean(cities_by_seed: list[dict], names: tuple[str, ...], supported: tuple[str, ...]) -> dict:
    values = [
        supported_leaf_mean_iou(np.asarray(list(cities.values()), dtype=np.int64).sum(axis=0), names, supported)
        for cities in cities_by_seed
    ]
    return {
        "supported_leaf_mean_iou": statistics.mean(values),
        "supported_leaf_mean_iou_sample_std": statistics.stdev(values),
        "per_seed_supported_leaf_mean_iou": values,
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if args.output_json.exists():
        raise SystemExit(f"output already exists: {args.output_json}")
    if any(len(group) != 3 for group in (args.b0_replay_dir, args.b0_half_replay_dir, args.b1_replay_dir)):
        raise SystemExit("supported-leaf summary requires exactly three replay directories per arm")
    b0 = [_load(directory, "b0", "fixed_budget_primary") for directory in args.b0_replay_dir]
    b0_half = [_load(directory, "b0_half", EXPOSURE_MODE) for directory in args.b0_half_replay_dir]
    b1 = [_load(directory, "b1", "fixed_budget_primary") for directory in args.b1_replay_dir]
    _, names, reference_cities = b0[0]
    reference_hash = b0[0][0]["run_config"]["evaluated_pair_identifier_set_sha256"]
    supported = tuple(name for name in names if name != args.exclude_leaf)
    if len(supported) != len(names) - 1:
        raise SystemExit(f"excluded leaf {args.exclude_leaf!r} is absent or duplicated")
    for report, candidate_names, cities in (*b0, *b0_half, *b1):
        if candidate_names != names or tuple(sorted(cities)) != tuple(sorted(reference_cities)):
            raise ValueError("arms have different class names or city sets")
        if report["run_config"]["evaluated_pair_identifier_set_sha256"] != reference_hash:
            raise ValueError("arms use different OpenEarthMap raster sets")
    b0_cities = [cities for _, _, cities in b0]
    b0_half_cities = [cities for _, _, cities in b0_half]
    b1_cities = [cities for _, _, cities in b1]
    contrasts = {
        "b0_half_minus_b0": paired_city_bootstrap_mean_supported_leaf_difference(
            b0_cities, b0_half_cities, names, supported, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
        ),
        "b1_minus_b0_half": paired_city_bootstrap_mean_supported_leaf_difference(
            b0_half_cities, b1_cities, names, supported, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
        ),
        "b1_minus_b0": paired_city_bootstrap_mean_supported_leaf_difference(
            b0_cities, b1_cities, names, supported, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
        ),
    }
    summary = {
        "version": "flair-supported-leaf-bootstrap-v1",
        "evaluated_pair_identifier_set_sha256": reference_hash,
        "excluded_leaf": args.exclude_leaf,
        "supported_leaves": list(supported),
        "arms": {
            "b0": _arm_mean(b0_cities, names, supported),
            "b0_half": _arm_mean(b0_half_cities, names, supported),
            "b1": _arm_mean(b1_cities, names, supported),
        },
        "contrasts": contrasts,
        "source_replay_directories": {
            "b0": [str(item) for item in args.b0_replay_dir],
            "b0_half": [str(item) for item in args.b0_half_replay_dir],
            "b1": [str(item) for item in args.b1_replay_dir],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    b1_b0 = contrasts["b1_minus_b0"]
    ci = b1_b0["confidence_interval"]
    print(
        "flair_supported_leaf_bootstrap_complete:",
        f"b1_minus_b0={b1_b0['point_difference']:.6f}",
        f"ci=[{ci['lower']:.6f},{ci['upper']:.6f}]",
        f"output={args.output_json}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
