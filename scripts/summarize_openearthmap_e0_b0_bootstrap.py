"""Compare frozen E0 and three fixed-budget B0 replays on B2's held-out cities."""

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
    parser.add_argument("--b0-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load(directory: Path) -> tuple[dict, tuple[str, ...], dict]:
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    city_payload = json.loads((directory / "city_confusions.json").read_text(encoding="utf-8"))
    names = tuple(city_payload["class_names"])
    cities = city_payload["city_confusions"]
    if tuple(report["city_bootstrap"]["city_names"]) != tuple(sorted(cities)):
        raise ValueError(f"{directory}: city manifest and city confusions disagree")
    return report, names, cities


def _validate_e0(report: dict) -> tuple[dict, str]:
    config = report.get("run_config", {})
    if config.get("experiment") != "openearthmap_b2_e0_checkpoint_replay":
        raise ValueError("--e0-replay-dir must be a frozen B2 E0 replay")
    exact = report.get("test")
    pair_hash = config.get("evaluated_test_pair_identifier_set_sha256")
    if not isinstance(exact, dict) or not isinstance(pair_hash, str):
        raise ValueError("E0 replay is missing exact metrics or its evaluated-pair fingerprint")
    return exact, pair_hash


def _validate_b0(report: dict, e0_pair_hash: str, e0_city_hash: str | None) -> dict:
    config = report.get("run_config", {})
    if config.get("arm") != "b0" or config.get("experiment") != "flair_b1_openearthmap_b0_replay":
        raise ValueError("--b0-replay-dir must be a fixed-budget FLAIR B0 replay")
    exact = report.get("exact_leaf")
    if not isinstance(exact, dict):
        raise ValueError("B0 replay is missing exact metrics")
    if config.get("evaluated_pair_identifier_set_sha256") != e0_pair_hash:
        raise ValueError("E0 and B0 evaluated OpenEarthMap pair sets do not match")
    selection = config.get("target_selection")
    if not isinstance(selection, dict) or selection.get("mode") != "b2_frozen_test_city_manifest":
        raise ValueError("B0 replay was not restricted to the frozen B2 test-city manifest")
    if selection.get("selected_pair_identifier_set_sha256_after_geographic_guard") != e0_pair_hash:
        raise ValueError("B0 post-guard target fingerprint does not match E0")
    if e0_city_hash is not None and selection.get("b2_test_city_set_sha256") != e0_city_hash:
        raise ValueError("B0 city manifest hash does not match E0's frozen B2 test split")
    return exact


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    if len(args.b0_replay_dir) != 3:
        raise SystemExit("the E0/B0 diagnostic requires exactly three fixed-budget B0 replay directories")
    if args.output_json.exists() and not args.dry_run:
        raise SystemExit(f"output already exists: {args.output_json}")
    e0_report, names, e0_cities = _load(args.e0_replay_dir)
    e0_exact, e0_pair_hash = _validate_e0(e0_report)
    e0_city_hash = e0_report["run_config"].get("full_city_split", {}).get("test_city_set_sha256")
    b0 = [_load(directory) for directory in args.b0_replay_dir]
    b0_exact: list[dict] = []
    for report, candidate_names, cities in b0:
        if candidate_names != names or tuple(sorted(cities)) != tuple(sorted(e0_cities)):
            raise ValueError("E0/B0 class names or city sets do not match")
        b0_exact.append(_validate_b0(report, e0_pair_hash, e0_city_hash))
    paired = paired_city_bootstrap_mean_difference(
        e0_cities,
        [cities for _, _, cities in b0],
        names,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    summary = {
        "comparison": "mean_b0_minus_frozen_e0",
        "class_names": list(names),
        "evaluated_pair_identifier_set_sha256": e0_pair_hash,
        "e0": e0_exact,
        "b0_seeds": b0_exact,
        "b0_mean": {
            "mean_iou": statistics.mean(result["mean_iou"] for result in b0_exact),
            "mean_iou_sample_std": statistics.stdev(result["mean_iou"] for result in b0_exact),
            "per_class_iou": {
                name: statistics.mean(result["per_class_iou"][name] for result in b0_exact) for name in names
            },
            "per_class_iou_sample_std": {
                name: statistics.stdev(result["per_class_iou"][name] for result in b0_exact) for name in names
            },
        },
        "paired_city_bootstrap_mean_b0_minus_e0": paired,
        "source_replay_directories": {
            "e0": str(args.e0_replay_dir),
            "b0": [str(directory) for directory in args.b0_replay_dir],
        },
    }
    if args.dry_run:
        print("openearthmap_e0_b0_bootstrap_dry_run_valid:", json.dumps(summary, sort_keys=True))
        return 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    herb = "herbaceous_vegetation"
    interval = paired["confidence_interval"]["per_class_iou"][herb]
    print(
        "openearthmap_e0_b0_bootstrap_summary_complete:",
        f"e0_rangeland_iou={e0_exact['per_class_iou'][herb]:.6f}",
        f"b0_mean_rangeland_iou={summary['b0_mean']['per_class_iou'][herb]:.6f}",
        f"paired_b0_minus_e0_rangeland={paired['point_difference']['per_class_iou'][herb]:.6f}",
        f"paired_rangeland_ci=[{interval['lower']:.6f},{interval['upper']:.6f}]",
        f"output={args.output_json}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
