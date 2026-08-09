"""Audit the official Semantic Drone RGB/mask release before target use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1data.semantic_drone import (
    SEMANTIC_DRONE_CLASS_DICT_RELATIVE,
    audit_semantic_drone_pairs,
    discover_semantic_drone_pairs,
    load_semantic_drone_classes,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_root = args.extracted_root / "semantic_drone_dataset"
    class_dict_path = dataset_root / SEMANTIC_DRONE_CLASS_DICT_RELATIVE
    classes = load_semantic_drone_classes(class_dict_path)
    pairs = discover_semantic_drone_pairs(args.extracted_root)
    audit = audit_semantic_drone_pairs(pairs, classes, max_samples=args.max_samples)
    audit.update(
        {
            "extracted_root": str(args.extracted_root),
            "class_dict_path": str(class_dict_path),
            "class_dict_sha256": sha256_file(class_dict_path),
            "archive": str(args.archive) if args.archive is not None else None,
            "archive_sha256": sha256_file(args.archive) if args.archive is not None else None,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        "semantic_drone_audit_complete:",
        f"pairs={audit['paired_raster_count']}",
        f"grass_pixels={audit['grass_pixel_count']}",
        f"grass_fraction={audit['grass_pixel_fraction']:.8f}",
        f"output={args.output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
