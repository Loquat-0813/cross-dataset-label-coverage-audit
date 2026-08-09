"""Verify the P1 LoveDA data adapter against the extracted local dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1data.loveda import discover_loveda_pairs, load_loveda_example, random_aligned_crop
from p1eval.metrics import stable_node_index
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy


def choose_evenly(items: tuple, count: int) -> tuple:
    if count < 1:
        raise ValueError("sample_count must be positive")
    if count >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return tuple(items[index] for index in indices)


def smoke_split(root: Path, split: str, sample_count: int, crop_size: int) -> None:
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    node_to_id = stable_node_index(taxonomy.nodes)
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
    pairs = discover_loveda_pairs(root, split)
    sampled = choose_evenly(pairs, sample_count)
    valid_counts = Counter()
    ignored = 0
    rng = np.random.default_rng(19)
    for pair in sampled:
        image, target = load_loveda_example(pair, mapping, node_to_id)
        image, target = random_aligned_crop(image, target, crop_size, rng)
        if image.shape != (crop_size, crop_size, 3) or target.shape != (crop_size, crop_size):
            raise RuntimeError(f"{pair.identifier}: crop shape mismatch")
        valid = target >= 0
        ignored += int((~valid).sum())
        valid_counts.update(target[valid].tolist())
    named_counts = {taxonomy.nodes[class_id]: count for class_id, count in sorted(valid_counts.items())}
    print(
        "loveda_data_smoke_valid:",
        f"split={split}",
        f"pairs={len(pairs)}",
        f"sampled={len(sampled)}",
        f"crop={crop_size}",
        f"ignored={ignored}",
        f"class_pixels={named_counts}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "loveda")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=512)
    args = parser.parse_args()
    smoke_split(args.root, args.split, args.sample_count, args.crop_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
