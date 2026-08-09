"""Summarize the LoveDA exposure-matched batch-size control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1eval.b2_bootstrap import paired_city_bootstrap_mean_difference


CONTROL_MODE = "post_primary_loveda_exposure_matched_batch_size_control"
EXPOSURE_MODE = "post_primary_loveda_exposure_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-half-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--b0-half-batch2-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--b1-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    return parser.parse_args()


def _source_protocol_mode(replay_config: dict) -> str | None:
    source_config = replay_config.get("source_checkpoint_run_config")
    if not isinstance(source_config, dict):
        return None
    mode = source_config.get("protocol_mode")
    return mode if isinstance(mode, str) else None


def _load(directory: Path, arm: str, expected_mode: str) -> tuple[dict, tuple[str, ...], dict]:
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = report.get("run_config", {})
    if config.get("arm") != arm or _source_protocol_mode(config) != expected_mode:
        raise ValueError(f"{directory}: expected {arm} with protocol mode {expected_mode}")
    city_payload = json.loads((directory / "city_confusions.json").read_text(encoding="utf-8"))
    names = tuple(city_payload["class_names"])
    cities = city_payload["city_confusions"]
    if tuple(report["city_bootstrap"]["city_names"]) != tuple(sorted(cities)):
        raise ValueError(f"{directory}: city manifest and city confusions disagree")
    return report, names, cities


def _mean_metrics(reports: list[dict], names: tuple[str, ...]) -> dict:
    exact = [report["exact_leaf"] for report in reports]
    return {
        "mean_iou": statistics.mean(result["mean_iou"] for result in exact),
        "mean_iou_sample_std": statistics.stdev(result["mean_iou"] for result in exact),
        "per_class_iou": {name: statistics.mean(result["per_class_iou"][name] for result in exact) for name in names},
        "per_class_iou_sample_std": {
            name: statistics.stdev(result["per_class_iou"][name] for result in exact) for name in names
        },
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if args.output_json.exists():
        raise SystemExit(f"output already exists: {args.output_json}")
    directories = (args.b0_half_replay_dir, args.b0_half_batch2_replay_dir, args.b1_replay_dir)
    if any(len(items) != 3 for items in directories):
        raise SystemExit("the batch-control summary requires exactly three replay directories per arm")
    b0_half = [_load(directory, "b0_half", EXPOSURE_MODE) for directory in args.b0_half_replay_dir]
    b0_half_batch2 = [
        _load(directory, "b0_half_batch2", CONTROL_MODE) for directory in args.b0_half_batch2_replay_dir
    ]
    b1 = [_load(directory, "b1", "fixed_budget_primary") for directory in args.b1_replay_dir]
    _, names, reference_cities = b0_half[0]
    reference_hash = b0_half[0][0]["run_config"]["evaluated_pair_identifier_set_sha256"]
    for report, candidate_names, cities in (*b0_half, *b0_half_batch2, *b1):
        if candidate_names != names or tuple(sorted(cities)) != tuple(sorted(reference_cities)):
            raise ValueError("batch-control arms have different class names or city sets")
        if report["run_config"]["evaluated_pair_identifier_set_sha256"] != reference_hash:
            raise ValueError("batch-control arms use different OpenEarthMap raster sets")
    batch2_minus_half = paired_city_bootstrap_mean_difference(
        [cities for _, _, cities in b0_half],
        [cities for _, _, cities in b0_half_batch2],
        names,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    b1_minus_batch2 = paired_city_bootstrap_mean_difference(
        [cities for _, _, cities in b0_half_batch2],
        [cities for _, _, cities in b1],
        names,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    summary = {
        "version": "flair-batch-control-bootstrap-v1",
        "class_names": list(names),
        "evaluated_pair_identifier_set_sha256": reference_hash,
        "b0_half_mean": _mean_metrics([report for report, _, _ in b0_half], names),
        "b0_half_batch2_mean": _mean_metrics([report for report, _, _ in b0_half_batch2], names),
        "b1_mean": _mean_metrics([report for report, _, _ in b1], names),
        "paired_city_bootstrap_mean_b0_half_batch2_minus_b0_half": batch2_minus_half,
        "paired_city_bootstrap_mean_b1_minus_b0_half_batch2": b1_minus_batch2,
        "source_replay_directories": {
            "b0_half": [str(directory) for directory in args.b0_half_replay_dir],
            "b0_half_batch2": [str(directory) for directory in args.b0_half_batch2_replay_dir],
            "b1": [str(directory) for directory in args.b1_replay_dir],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    herb = "herbaceous_vegetation"
    batch_interval = batch2_minus_half["confidence_interval"]["per_class_iou"][herb]
    b1_interval = b1_minus_batch2["confidence_interval"]["per_class_iou"][herb]
    print(
        "flair_batch_control_bootstrap_complete:",
        f"batch2_minus_half_rangeland={batch2_minus_half['point_difference']['per_class_iou'][herb]:.6f}",
        f"batch2_minus_half_ci=[{batch_interval['lower']:.6f},{batch_interval['upper']:.6f}]",
        f"b1_minus_batch2_rangeland={b1_minus_batch2['point_difference']['per_class_iou'][herb]:.6f}",
        f"b1_minus_batch2_ci=[{b1_interval['lower']:.6f},{b1_interval['upper']:.6f}]",
        f"output={args.output_json}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
