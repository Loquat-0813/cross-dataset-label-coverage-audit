"""Summarize fixed-budget B1 coverage completion with a paired city bootstrap."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--b1-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    return parser.parse_args()


def _load(directory: Path) -> tuple[dict, tuple[str, ...], dict]:
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    city_payload = json.loads((directory / "city_confusions.json").read_text(encoding="utf-8"))
    names = tuple(city_payload["class_names"])
    cities = city_payload["city_confusions"]
    if tuple(report["city_bootstrap"]["city_names"]) != tuple(sorted(cities)):
        raise ValueError(f"{directory}: city manifest and city confusions disagree")
    return report, names, cities


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if args.output_json.exists():
        raise SystemExit(f"output already exists: {args.output_json}")
    if len(args.b0_replay_dir) != 3 or len(args.b1_replay_dir) != 3:
        raise SystemExit("the preregistered B1 summary requires exactly three B0 and three B1 replay directories")
    b0 = [_load(directory) for directory in args.b0_replay_dir]
    b0_report, names, _ = b0[0]
    b1 = [_load(directory) for directory in args.b1_replay_dir]
    for report, candidate_names, _ in (*b0, *b1):
        if candidate_names != names:
            raise ValueError("B0/B1 class names do not match")
        if report["run_config"]["evaluated_pair_identifier_set_sha256"] != b0_report["run_config"]["evaluated_pair_identifier_set_sha256"]:
            raise ValueError("B0/B1 evaluated OpenEarthMap pair sets do not match")
    paired = paired_city_bootstrap_mean_difference(
        [cities for _, _, cities in b0],
        [cities for _, _, cities in b1],
        names,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    b0_exact = [report["exact_leaf"] for report, _, _ in b0]
    b1_exact = [report["exact_leaf"] for report, _, _ in b1]
    summary = {
        "class_names": list(names),
        "evaluated_pair_identifier_set_sha256": b0_report["run_config"]["evaluated_pair_identifier_set_sha256"],
        "b0_seeds": b0_exact,
        "b0_mean": {
            "mean_iou": statistics.mean(result["mean_iou"] for result in b0_exact),
            "mean_iou_sample_std": statistics.stdev(result["mean_iou"] for result in b0_exact),
            "per_class_iou": {name: statistics.mean(result["per_class_iou"][name] for result in b0_exact) for name in names},
            "per_class_iou_sample_std": {
                name: statistics.stdev(result["per_class_iou"][name] for result in b0_exact) for name in names
            },
        },
        "b1_seeds": b1_exact,
        "b1_mean": {
            "mean_iou": statistics.mean(result["mean_iou"] for result in b1_exact),
            "mean_iou_sample_std": statistics.stdev(result["mean_iou"] for result in b1_exact),
            "per_class_iou": {name: statistics.mean(result["per_class_iou"][name] for result in b1_exact) for name in names},
            "per_class_iou_sample_std": {
                name: statistics.stdev(result["per_class_iou"][name] for result in b1_exact) for name in names
            },
        },
        "paired_city_bootstrap_mean_b1_minus_b0": paired,
        "source_replay_directories": [
            *(str(directory) for directory in args.b0_replay_dir),
            *(str(directory) for directory in args.b1_replay_dir),
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    herb = "herbaceous_vegetation"
    interval = paired["confidence_interval"]["per_class_iou"][herb]
    print(
        "flair_b1_bootstrap_summary_complete:",
        f"b1_mean_miou={summary['b1_mean']['mean_iou']:.6f}",
        f"b1_mean_rangeland_iou={summary['b1_mean']['per_class_iou'][herb]:.6f}",
        f"paired_rangeland_difference={paired['point_difference']['per_class_iou'][herb]:.6f}",
        f"paired_rangeland_ci=[{interval['lower']:.6f},{interval['upper']:.6f}]",
        f"output={args.output_json}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
