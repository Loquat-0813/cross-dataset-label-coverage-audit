"""Freeze the Semantic Drone target manifest before any target inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1data.semantic_drone import discover_semantic_drone_pairs
from p1data.splits import identifier_set_sha256


def build_manifest(extracted_root: Path, raw_audit: dict) -> dict:
    pairs = discover_semantic_drone_pairs(extracted_root)
    expected_count = int(raw_audit.get("paired_raster_count", -1))
    total_count = int(raw_audit.get("total_available_pairs", -1))
    if expected_count != len(pairs) or total_count != len(pairs):
        raise ValueError(
            "raw audit does not cover the complete Semantic Drone pair set: "
            f"audit_pairs={expected_count}, audit_total={total_count}, discovered={len(pairs)}"
        )
    fingerprint = identifier_set_sha256(pair.identifier for pair in pairs)
    if raw_audit.get("identifier_set_sha256") != fingerprint:
        raise ValueError("raw audit identifier fingerprint does not match current discovery")
    records = [
        {
            "identifier": pair.identifier,
            "image": pair.image_path.relative_to(extracted_root).as_posix(),
            "mask": pair.mask_path.relative_to(extracted_root).as_posix(),
        }
        for pair in pairs
    ]
    return {
        "version": "semantic-drone-confirmation-manifest-v1",
        "dataset": "semantic_drone",
        "pair_count": len(records),
        "identifier_set_sha256": fingerprint,
        "source_archive": raw_audit.get("archive"),
        "source_archive_sha256": raw_audit.get("archive_sha256"),
        "raw_audit": raw_audit,
        "mapping": {
            "raw_class": "grass",
            "grass_rgb": raw_audit.get("grass_rgb"),
            "p1_leaf": "herbaceous_vegetation",
            "other_raw_classes": "ignored",
        },
        "pairs": records,
        "frozen_before_inference": True,
        "target_metrics_inspected": False,
        "target_metrics_used_for_method_selection": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_audit = json.loads(args.raw_audit.read_text(encoding="utf-8"))
    manifest = build_manifest(args.extracted_root, raw_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "semantic_drone_confirmation_manifest_complete:",
        f"pairs={manifest['pair_count']}",
        f"identifier_set_sha256={manifest['identifier_set_sha256']}",
        f"output={args.output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
