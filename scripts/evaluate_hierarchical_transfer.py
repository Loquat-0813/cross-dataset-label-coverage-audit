"""Evaluate semantic-segmentation masks under the P1 frozen taxonomy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from PIL import Image

from p1eval.metrics import hierarchy_iou, iou_from_taxonomy_ids, stable_node_index, unknown_rejection_auroc
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f"{path}: expected one-channel mask, got {mask.shape}")
    return mask


def parse_prediction_mapping(path: Path, taxonomy_nodes: tuple[str, ...]) -> dict[int, str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_mapping = document.get("prediction_id_to_taxonomy_node", document)
    mapping = {int(key): value for key, value in raw_mapping.items()}
    for prediction_id, node in mapping.items():
        if prediction_id < 0 or node not in taxonomy_nodes:
            raise ValueError(f"prediction mapping has invalid entry {prediction_id}: {node}")
    return mapping


def adapt_prediction(raw_prediction: np.ndarray, mapping: dict[int, str], node_to_id: dict[str, int]) -> np.ndarray:
    result = np.full(raw_prediction.shape, -1, dtype=np.int64)
    for prediction_id, node in mapping.items():
        result[raw_prediction == prediction_id] = node_to_id[node]
    unrecognised = np.unique(raw_prediction[result == -1])
    if unrecognised.size:
        raise ValueError(f"prediction contains IDs absent from prediction mapping: {unrecognised.tolist()}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-map", type=Path, required=True)
    parser.add_argument("--unknown-score", type=Path, help="Optional .npy array where higher means more likely genuinely unknown.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    parser.add_argument("--mappings", type=Path, default=ROOT / "ontology" / "dataset_label_mappings_v0.yaml")
    args = parser.parse_args()

    taxonomy, levels = load_taxonomy(args.taxonomy)
    node_to_id = stable_node_index(taxonomy.nodes)
    dataset_mapping = load_dataset_mapping(args.mappings, args.dataset, taxonomy)
    raw_target = read_mask(args.target)
    target = dataset_mapping.adapt(raw_target, node_to_id)
    prediction = adapt_prediction(read_mask(args.prediction), parse_prediction_mapping(args.prediction_map, taxonomy.nodes), node_to_id)
    exact_target = target.copy()
    exact_target[~dataset_mapping.exact_valid_mask(raw_target)] = -1
    exact = iou_from_taxonomy_ids(exact_target, prediction, taxonomy.nodes)
    result = {
        "protocol": "hierarchical-transfer-v0",
        "taxonomy": taxonomy.version,
        "dataset": args.dataset,
        "mapping_coverage": {
            "raw_semantic_ids": sorted(dataset_mapping.raw_id_to_node),
            "exact_raw_ids": sorted(dataset_mapping.exact_raw_ids),
            "coarse_raw_ids": sorted(dataset_mapping.coarse_raw_ids),
            "ignored_raw_ids": sorted(dataset_mapping.ignored_raw_ids),
            "unknown_raw_ids": sorted(dataset_mapping.unknown_raw_ids),
            "valid_pixels": exact.valid_pixels,
        },
        "exact_node": exact.as_dict(),
        "ancestor_levels": {},
    }
    for level in sorted(set(levels.values())):
        result["ancestor_levels"][str(level)] = hierarchy_iou(target, prediction, taxonomy, levels, node_to_id, level).as_dict()
    if not dataset_mapping.unknown_raw_ids:
        result["unknown_rejection"] = {"status": "not_applicable_no_genuine_unknown_labels"}
    elif args.unknown_score is None:
        result["unknown_rejection"] = {"status": "pending_unknown_score"}
    else:
        unknown_target, unknown_valid = dataset_mapping.unknown_target_and_valid_mask(raw_target)
        unknown_score = np.load(args.unknown_score)
        result["unknown_rejection"] = {
            "status": "computed",
            "auroc": unknown_rejection_auroc(unknown_target, unknown_score, unknown_valid),
            "eligible_pixels": int(unknown_valid.sum()),
            "genuinely_unknown_pixels": int(unknown_target[unknown_valid].sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"evaluation_valid: dataset={args.dataset}; valid_pixels={exact.valid_pixels}; exact_miou={exact.macro_iou}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
