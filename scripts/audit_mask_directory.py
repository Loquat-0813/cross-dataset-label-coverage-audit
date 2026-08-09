"""Perform a complete raw-label audit after a dataset has been extracted."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--glob", default="**/*.tif")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    masks = sorted(path for path in args.mask_root.glob(args.glob) if path.is_file())
    if not masks:
        raise SystemExit(f"no masks matched {args.glob!r} beneath {args.mask_root}")
    aggregate = Counter()
    modes = set()
    sizes = set()
    for path in masks:
        with Image.open(path) as image:
            labels = np.asarray(image)
            if labels.ndim != 2:
                raise SystemExit(f"{path}: expected a one-channel label mask, got {labels.shape}")
            ids, counts = np.unique(labels, return_counts=True)
            aggregate.update({int(label_id): int(count) for label_id, count in zip(ids, counts)})
            modes.add(image.mode)
            sizes.add(image.size)
    result = {
        "version": "mask-directory-audit-v1",
        "mask_root": str(args.mask_root),
        "glob": args.glob,
        "mask_count": len(masks),
        "modes": sorted(modes),
        "sizes": [list(size) for size in sorted(sizes)],
        "mask_values_pixel_counts": {str(label_id): count for label_id, count in sorted(aggregate.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"audit_valid: masks={len(masks)}; label_ids={sorted(aggregate)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
