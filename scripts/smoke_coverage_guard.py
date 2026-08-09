"""CUDA smoke test for coverage-aware conformal routing under the P1 taxonomy."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1eval.metrics import stable_node_index
from p1eval.selective_hierarchy import build_coverage_node_lookup
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.coverage_guard import CoverageGuard, split_conformal_probability_threshold
from p1model.taxonomy import build_taxonomy_leaf_index
from p1train.loveda_e1 import source_supervised_leaf_mask


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("coverage guard smoke requires CUDA")
    device = torch.device("cuda")
    taxonomy, levels = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "loveda", taxonomy)
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    covered = source_supervised_leaf_mask(mapping, leaves.leaf_names)
    lookup = build_coverage_node_lookup(
        taxonomy, levels, node_to_id, leaves.leaf_names, covered.numpy().astype(np.bool_, copy=False)
    )
    leaf_nodes = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], dtype=torch.long)
    guard = CoverageGuard(covered, leaf_nodes, torch.from_numpy(lookup)).to(device)
    herb = leaves.leaf_names.index("herbaceous_vegetation")
    base = torch.full((1, len(leaves.leaf_names), 2, 2), -4.0, device=device)
    base[:, herb] = 6.0
    adapted = torch.full_like(base, -4.0)
    adapted[:, leaves.leaf_names.index("woody_vegetation")] = 6.0
    output = guard(base, adapted, probability_threshold=0.5)
    expected_parent = leaves.node_to_id["vegetated_surface"]
    if not output.routed_to_base.all() or not torch.equal(output.node_prediction, torch.full_like(output.node_prediction, expected_parent)):
        raise RuntimeError("coverage guard did not promote the uncovered herbaceous set to vegetated_surface")
    calibration_logits = torch.tensor([[[[4.0, 0.0]], [[0.0, 4.0]]]], device=device)
    calibration_target = torch.tensor([[[0, 1]]], device=device)
    threshold = split_conformal_probability_threshold(calibration_logits, calibration_target, alpha=0.1)
    print(
        "coverage_guard_smoke_valid:",
        f"device={torch.cuda.get_device_name(0)}",
        f"leaves={len(leaves.leaf_names)}",
        f"covered={int(covered.sum())}",
        f"threshold={threshold:.6f}",
        f"emitted_node={expected_parent}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
