"""Combine completed B2 checkpoint replays into paired city-bootstrap contrasts."""

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
    parser.add_argument("--e0-replay-dir", type=Path, required=True)
    parser.add_argument("--e1-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_replay(directory: Path) -> tuple[dict, tuple[str, ...], dict]:
    report_path = directory / "metrics.json"
    city_path = directory / "city_confusions.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    city_payload = json.loads(city_path.read_text(encoding="utf-8"))
    class_names = tuple(city_payload["class_names"])
    city_confusions = city_payload["city_confusions"]
    if tuple(report["city_bootstrap"]["city_names"]) != tuple(sorted(city_confusions)):
        raise ValueError(f"{directory}: city bootstrap manifest does not match city confusions")
    return report, class_names, city_confusions


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if args.output_json.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output_json}; use --overwrite only intentionally")
    e0_report, class_names, e0_city_confusions = _load_replay(args.e0_replay_dir)
    e1 = [_load_replay(directory) for directory in args.e1_replay_dir]
    for report, names, _ in e1:
        if names != class_names:
            raise ValueError("B2 replay class names do not match")
        if report["run_config"]["evaluated_test_pair_identifier_set_sha256"] != e0_report["run_config"]["evaluated_test_pair_identifier_set_sha256"]:
            raise ValueError("B2 replay test pair fingerprints do not match")
    paired = paired_city_bootstrap_mean_difference(
        e0_city_confusions,
        [confusions for _, _, confusions in e1],
        class_names,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    per_seed = [report["test"] for report, _, _ in e1]
    summary = {
        "class_names": list(class_names),
        "test_pair_identifier_set_sha256": e0_report["run_config"]["evaluated_test_pair_identifier_set_sha256"],
        "e0": e0_report["test"],
        "e1_seeds": per_seed,
        "e1_mean": {
            "mean_iou": statistics.mean(report["mean_iou"] for report in per_seed),
            "mean_iou_sample_std": statistics.stdev(report["mean_iou"] for report in per_seed),
            "per_class_iou": {
                name: statistics.mean(report["per_class_iou"][name] for report in per_seed) for name in class_names
            },
            "per_class_iou_sample_std": {
                name: statistics.stdev(report["per_class_iou"][name] for report in per_seed) for name in class_names
            },
        },
        "paired_city_bootstrap_mean_e1_minus_e0": paired,
        "source_replay_directories": [str(args.e0_replay_dir), *(str(directory) for directory in args.e1_replay_dir)],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rangeland = "herbaceous_vegetation"
    interval = paired["confidence_interval"]["per_class_iou"][rangeland]
    print(
        "openearthmap_b2_bootstrap_summary_complete:",
        f"e1_mean_miou={summary['e1_mean']['mean_iou']:.6f}",
        f"e1_mean_rangeland_iou={summary['e1_mean']['per_class_iou'][rangeland]:.6f}",
        f"paired_rangeland_difference={paired['point_difference']['per_class_iou'][rangeland]:.6f}",
        f"paired_rangeland_ci=[{interval['lower']:.6f},{interval['upper']:.6f}]",
        f"output={args.output_json}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
