"""Audit official FLAIR image/mask pairs before enabling the B1 source mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1data.flair import (
    FLAIR_ALL_DOMAINS,
    FLAIR_TRAIN_DOMAINS,
    FLAIR_VALIDATION_DOMAINS,
    audit_flair_pairs,
    discover_flair_domain_pairs,
    discover_flair_pairs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument(
        "--extracted-root",
        type=Path,
        help="Domain-preserving root written by prepare_flair_b1_source.py.",
    )
    parser.add_argument("--image-glob", default="**/*")
    parser.add_argument("--mask-glob", default="**/*")
    parser.add_argument("--herbaceous-raw-id", type=int, default=10)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    has_explicit_roots = args.image_root is not None or args.mask_root is not None
    if args.extracted_root is not None and has_explicit_roots:
        raise SystemExit("choose either --extracted-root or --image-root/--mask-root")
    if args.extracted_root is None and (args.image_root is None or args.mask_root is None):
        raise SystemExit("provide --extracted-root or both --image-root and --mask-root")
    if args.extracted_root is not None:
        all_pairs = discover_flair_domain_pairs(args.extracted_root, FLAIR_ALL_DOMAINS)
        train_pairs = tuple(pair for pair in all_pairs if pair.identifier.split("/", 1)[0] in FLAIR_TRAIN_DOMAINS)
        validation_pairs = tuple(
            pair for pair in all_pairs if pair.identifier.split("/", 1)[0] in FLAIR_VALIDATION_DOMAINS
        )
        audit = audit_flair_pairs(all_pairs, herbaceous_raw_id=args.herbaceous_raw_id)
        audit.update(
            {
                "extracted_root": str(args.extracted_root),
                "train_domains": list(FLAIR_TRAIN_DOMAINS),
                "validation_domains": list(FLAIR_VALIDATION_DOMAINS),
                "train": audit_flair_pairs(train_pairs, herbaceous_raw_id=args.herbaceous_raw_id),
                "validation": audit_flair_pairs(validation_pairs, herbaceous_raw_id=args.herbaceous_raw_id),
            }
        )
    else:
        pairs = discover_flair_pairs(
            args.image_root,
            args.mask_root,
            image_glob=args.image_glob,
            mask_glob=args.mask_glob,
        )
        audit = audit_flair_pairs(pairs, herbaceous_raw_id=args.herbaceous_raw_id)
        audit.update(
            {
                "image_root": str(args.image_root),
                "mask_root": str(args.mask_root),
                "image_glob": args.image_glob,
                "mask_glob": args.mask_glob,
            }
        )
    if args.source_manifest is not None:
        source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        archive_names = [record["archive"] for record in source_manifest.get("archives", [])]
        expected_archives = [f"{domain}.zip" for domain in FLAIR_ALL_DOMAINS]
        if archive_names != expected_archives:
            raise SystemExit("source manifest archive order does not match the frozen FLAIR domain list")
        audit["source_manifest"] = str(args.source_manifest)
        audit["source_archive_sha256"] = {record["archive"]: record["sha256"] for record in source_manifest["archives"]}
    audit["source_mapping_status"] = "source_audit_pending_geographic_overlap_review"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        "flair_b1_audit_complete:",
        f"pairs={audit['paired_raster_count']}",
        f"raw_values={sorted(audit['mask_values_pixel_counts'], key=int)}",
        f"herbaceous_pixels={audit['herbaceous_pixel_count']}",
        f"output={args.output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
