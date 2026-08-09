"""Describe whether B1/B0 leaf changes align with frozen CLIP prompt distance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1eval.prompts import load_leaf_prompts
from p1eval.taxonomy import load_taxonomy
from p1model.taxonomy import build_taxonomy_leaf_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-b1-summary", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-config", type=Path, default=ROOT / "configs" / "taxonomy_prompts_v0.yaml")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    if args.output_json.exists():
        raise SystemExit(f"output already exists: {args.output_json}")
    summary = json.loads(args.b0_b1_summary.read_text(encoding="utf-8"))
    class_names = tuple(summary.get("class_names", ()))
    herb = "herbaceous_vegetation"
    paired = summary.get("paired_city_bootstrap_mean_b1_minus_b0", {})
    changes = paired.get("point_difference", {}).get("per_class_iou", {})
    if herb not in class_names or set(changes) != set(class_names):
        raise ValueError("B0/B1 summary does not contain the required exact-leaf paired changes")
    covered = tuple(name for name in class_names if name != herb)
    if len(covered) != 6:
        raise ValueError("semantic-allocation diagnostic expects six LoveDA-covered leaves")
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    leaves = build_taxonomy_leaf_index(taxonomy)
    if tuple(leaves.leaf_names) != class_names:
        raise ValueError("summary class order does not match the frozen taxonomy leaf order")
    prompts = load_leaf_prompts(args.prompt_config, leaves.leaf_names)
    device = _device(args.device)
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    features = torch.nn.functional.normalize(backbone.encode_text(prompts), dim=1).cpu().numpy()
    herb_index = class_names.index(herb)
    records = []
    for index, name in enumerate(class_names):
        if name == herb:
            continue
        similarity = float(np.dot(features[index], features[herb_index]))
        records.append(
            {
                "leaf": name,
                "b1_minus_b0_iou": float(changes[name]),
                "frozen_clip_prompt_cosine_similarity_to_herbaceous": similarity,
                "frozen_clip_prompt_cosine_distance_to_herbaceous": 1.0 - similarity,
            }
        )
    distances = np.asarray([record["frozen_clip_prompt_cosine_distance_to_herbaceous"] for record in records])
    deltas = np.asarray([record["b1_minus_b0_iou"] for record in records])
    correlation = float(np.corrcoef(distances, deltas)[0, 1])
    report = {
        "analysis_type": "post_primary_descriptive_semantic_allocation_diagnostic",
        "interpretation_limit": (
            "Six covered leaves only. Frozen CLIP prompt distance and IoU change are descriptive; "
            "their correlation is not evidence of a semantic-gradient mechanism."
        ),
        "b0_b1_summary": str(args.b0_b1_summary),
        "b0_b1_summary_sha256": _sha256(args.b0_b1_summary),
        "model_id": args.model_id,
        "prompt_config": str(args.prompt_config),
        "prompt_config_sha256": _sha256(args.prompt_config),
        "device": str(device),
        "reference_leaf": herb,
        "covered_leaf_count": len(records),
        "pearson_correlation_distance_vs_b1_minus_b0_iou": correlation,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "flair_b1_semantic_allocation_diagnostic_complete:",
        f"covered_leaves={len(records)}",
        f"pearson_distance_vs_delta={correlation:.6f}",
        f"output={args.output_json}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
