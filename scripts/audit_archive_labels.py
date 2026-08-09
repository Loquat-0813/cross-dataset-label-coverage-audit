"""Audit TIFF label IDs directly from a ZIP archive without extracting a dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def choose_evenly(items: list[str], count: int) -> list[str]:
    if count < 1:
        raise ValueError("sample count must be positive")
    if len(items) <= count:
        return items
    positions = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[index] for index in positions]


def read_label_values(payload: bytes) -> tuple[Counter[int], tuple[int, int], str]:
    with Image.open(BytesIO(payload)) as image:
        values = np.asarray(image)
        if values.ndim != 2:
            raise ValueError(f"expected one-channel label TIFF, got array shape {values.shape}")
        ids, counts = np.unique(values, return_counts=True)
        return Counter({int(label_id): int(count) for label_id, count in zip(ids, counts)}), image.size, image.mode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--label-prefix", required=True, help="ZIP-internal label directory, such as masks/")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefix = args.label_prefix.rstrip("/") + "/"
    with ZipFile(args.archive) as archive:
        label_members = sorted(member.filename for member in archive.infolist() if member.filename.startswith(prefix) and not member.is_dir())
        if not label_members:
            raise SystemExit(f"no label files found below {prefix!r}")
        sampled = choose_evenly(label_members, args.sample_count)
        observed = Counter()
        records = []
        for member in sampled:
            counts, size, mode = read_label_values(archive.read(member))
            observed.update(counts)
            records.append({"member": member, "values": sorted(counts), "size": list(size), "mode": mode})

    result = {
        "version": "archive-label-audit-v1",
        "archive": args.archive.name,
        "label_prefix": prefix,
        "label_files_total": len(label_members),
        "sampled_label_files": len(sampled),
        "observed_label_ids": sorted(observed),
        "observed_label_pixel_counts": {str(label_id): count for label_id, count in sorted(observed.items())},
        "samples": records,
        "scope": "Sample audit only. Run a complete raw-label audit after extraction before publishing metrics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"audit_valid: {args.archive.name}; labels={result['observed_label_ids']}; samples={len(sampled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
