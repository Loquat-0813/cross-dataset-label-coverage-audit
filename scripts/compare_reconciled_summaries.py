#!/usr/bin/env python3
"""Compare archived summaries with deterministic reruns by numeric content."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


PAIRS = (
    "flair_b1_oem_no_paris_bootstrap_summary.json",
    "flair_supported_leaf_bootstrap_summary.json",
    "flair_b0_half_oem_no_paris_bootstrap_summary.json",
    "flair_batch_control_summary.json",
    "semantic_drone_confirmation_bootstrap_summary.json",
    "semantic_drone_confirmation_bootstrap_396_exploratory.json",
    "b2_openearthmap_bootstrap_summary.json",
    "e0_b0_oem_b2test_bootstrap_summary.json",
)


def normalize(value: Any, path: str = "") -> Any:
    if isinstance(value, dict):
        ignored = {"source_replay_directories", "source_result_directories"}
        if path == "":
            ignored.add("version")
        if path == "target":
            ignored.update({"excluded_identifiers", "manifest_evaluated_pairs"})
        return {key: normalize(item, f"{path}.{key}" if path else key) for key, item in value.items() if key not in ignored}
    if isinstance(value, list):
        return [normalize(item, f"{path}.{index}") for index, item in enumerate(value)]
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-output-root", type=Path, required=True)
    parser.add_argument("--rerun-output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    comparisons = []
    for name in PAIRS:
        archived = json.loads((args.archive_output_root / name).read_text(encoding="utf-8"))
        rerun = json.loads((args.rerun_output_root / name).read_text(encoding="utf-8"))
        archived_core = normalize(archived)
        rerun_core = normalize(rerun)
        comparisons.append(
            {
                "summary": name,
                "numeric_and_structural_core": "match" if archived_core == rerun_core else "mismatch",
                "archived_core_sha256": f"sha256:{canonical_hash(archived_core)}",
                "rerun_core_sha256": f"sha256:{canonical_hash(rerun_core)}",
            }
        )

    mismatches = sum(item["numeric_and_structural_core"] != "match" for item in comparisons)
    payload = {
        "comparison_version": "reconciled-summary-core-v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "summary": {
            "files_checked": len(comparisons),
            "core_matches": len(comparisons) - mismatches,
            "core_mismatches": mismatches,
            "verdict": "PASS" if mismatches == 0 else "FAIL",
        },
        "comparisons": comparisons,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"reconciled_summary_comparison_complete: verdict={payload['summary']['verdict']} output={args.output_json}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
