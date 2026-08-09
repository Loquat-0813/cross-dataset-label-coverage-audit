"""GPU/CPU smoke test for the P1 semantic calibration adapter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from p1eval.taxonomy import load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.losses import taxonomy_partial_label_nll
from p1model.taxonomy import build_taxonomy_leaf_index


def main() -> int:
    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    leaf_index = build_taxonomy_leaf_index(taxonomy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(19)
    model = LowRankSemanticCalibration(embedding_dim=512, rank=32).to(device)
    pixels = torch.randn(2, 512, 6, 7, device=device, requires_grad=True)
    prompts = torch.randn(len(leaf_index.leaf_names), 512, device=device, requires_grad=True)
    target = torch.full((2, 6, 7), leaf_index.node_to_id["artificial_surface"], device=device, dtype=torch.long)
    logits = model(pixels, prompts)
    mask = torch.from_numpy(leaf_index.descendant_leaf_mask).to(device=device)
    loss = taxonomy_partial_label_nll(logits, target, mask)
    loss.backward()
    if pixels.grad is not None or prompts.grad is not None:
        raise RuntimeError("frozen backbone features unexpectedly received gradients")
    if not all(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("a calibration parameter did not receive a gradient")
    print(
        "semantic_calibration_smoke_valid:",
        f"device={device.type}",
        f"leaves={len(leaf_index.leaf_names)}",
        f"parameters={model.trainable_parameter_count}",
        f"logits={tuple(logits.shape)}",
        f"loss={loss.item():.6f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
