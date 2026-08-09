"""Audit LoveDA extraction without modifying image or mask files."""

from collections import Counter
from pathlib import Path
import argparse
import json

from PIL import Image
import numpy as np


def audit_split(split_root: Path) -> dict:
    missing_masks = []
    extra_masks = []
    sample_count = 0
    image_sizes = set()
    mask_sizes = set()
    image_modes = set()
    mask_modes = set()
    aggregate = Counter()
    for region in sorted(p for p in split_root.iterdir() if p.is_dir()):
        image_dir = region / "images_png"
        mask_dir = region / "masks_png"
        image_paths = {p.stem: p for p in image_dir.glob("*.png")}
        mask_paths = {p.stem: p for p in mask_dir.glob("*.png")}
        missing_masks.extend(f"{region.name}/{stem}" for stem in image_paths.keys() - mask_paths.keys())
        extra_masks.extend(f"{region.name}/{stem}" for stem in mask_paths.keys() - image_paths.keys())
        for stem in sorted(image_paths.keys() & mask_paths.keys()):
            with Image.open(image_paths[stem]) as image, Image.open(mask_paths[stem]) as mask:
                labels = np.asarray(mask.convert("L"), dtype=np.uint8)
                counts = np.bincount(labels.reshape(-1), minlength=256)
                aggregate.update({index: int(count) for index, count in enumerate(counts) if count})
                sample_count += 1
                image_sizes.add(image.size)
                mask_sizes.add(mask.size)
                image_modes.add(image.mode)
                mask_modes.add(mask.mode)
    return {
        "samples": sample_count,
        "missing_masks": sorted(missing_masks),
        "extra_masks": sorted(extra_masks),
        "image_sizes": sorted(image_sizes),
        "mask_sizes": sorted(mask_sizes),
        "image_modes": sorted(image_modes),
        "mask_modes": sorted(mask_modes),
        "mask_values_pixel_counts": {str(k): v for k, v in sorted(aggregate.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/loveda"))
    parser.add_argument("--output", type=Path, default=Path("data/audits/loveda_audit_v1.json"))
    args = parser.parse_args()
    result = {"version": "loveda-audit-v1", "splits": {}}
    for split_name, split_dir in (("train", args.root / "train" / "Train"), ("val", args.root / "val" / "Val")):
        if not split_dir.exists():
            raise SystemExit(f"missing split directory: {split_dir}")
        result["splits"][split_name] = audit_split(split_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, split in result["splits"].items():
        print(name, "samples=", split["samples"], "missing=", len(split["missing_masks"]), "extra=", len(split["extra_masks"]))
        print(name, "sizes=", split["image_sizes"], "mask_values=", sorted(split["mask_values_pixel_counts"]))


if __name__ == "__main__":
    main()
