"""Replay frozen OEM checkpoints and emit non-visual arrays for an R figure.

This script intentionally writes numeric CSV arrays only. The publication
figure is assembled exclusively by figures/generate_fig5_oem_qualitative_20260806.R.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1data.external import discover_external_pairs
from p1data.openearthmap_city_split import exclude_openearthmap_cities
from p1data.torch_dataset import TaxonomyRasterDataset
from p1eval.metrics import stable_node_index
from p1eval.prompts import load_leaf_prompts
from p1eval.qualitative_selection import select_oem_qualitative_identifiers
from p1eval.taxonomy import load_dataset_mapping, load_taxonomy
from p1model.calibration import LowRankSemanticCalibration
from p1model.taxonomy import build_taxonomy_leaf_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--b1-checkpoint", type=Path, required=True)
    parser.add_argument("--b0-confusions", type=Path, required=True)
    parser.add_argument("--b1-confusions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, default=ROOT / "configs" / "taxonomy_prompts_v0.yaml")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--min-reference-pixels", type=int, default=10_000)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_confusions(path: Path) -> tuple[tuple[str, ...], dict[str, list[list[int]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(payload.get("class_names", ()))
    matrices = payload.get("confusions")
    if not names or not isinstance(matrices, dict):
        raise ValueError(f"{path}: expected class_names and per-raster confusions")
    return names, matrices


def _checkpoint(path: Path, expected_arm: str, embedding_dim: int, prompts: tuple[str, ...]) -> LowRankSemanticCalibration:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config")
    if not isinstance(config, dict) or config.get("experiment") != f"flair_coverage_{expected_arm}":
        raise ValueError(f"{path}: expected a completed {expected_arm.upper()} checkpoint")
    if tuple(config.get("prompts", ())) != prompts:
        raise ValueError(f"{path}: prompt configuration does not match the frozen prompt list")
    adapter = LowRankSemanticCalibration(embedding_dim, rank=int(config["rank"]))
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    return adapter


def _target_leaf_ids(target: torch.Tensor, leaf_node_ids: torch.Tensor) -> torch.Tensor:
    lookup = torch.full((int(leaf_node_ids.max()) + 1,), -1, device=target.device, dtype=torch.long)
    lookup[leaf_node_ids] = torch.arange(len(leaf_node_ids), device=target.device)
    result = torch.full_like(target, -1)
    valid = target >= 0
    result[valid] = lookup[target[valid]]
    return result


@torch.inference_mode()
def _predict(image: torch.Tensor, backbone, adapter, text_features: torch.Tensor, tile_size: int, device: torch.device) -> np.ndarray:
    height, width = image.shape[-2:]
    prediction = torch.empty((height, width), dtype=torch.long, device=device)
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = torch.zeros((image.shape[0], tile_size, tile_size), dtype=image.dtype)
            tile[:, : bottom - top, : right - left] = image[:, top:bottom, left:right]
            logits = adapter(backbone.encode_image(tile.unsqueeze(0).to(device)), text_features)
            logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
            prediction[top:bottom, left:right] = logits.argmax(dim=1)[0, : bottom - top, : right - left]
    return prediction.cpu().numpy().astype(np.int16, copy=False)


def _write_csv(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, array.reshape(-1, array.shape[-1]) if array.ndim == 3 else array.reshape(-1, 1), fmt="%d", delimiter=",")


def _error_codes(reference: np.ndarray, prediction: np.ndarray, herbaceous: int, cropland: int) -> np.ndarray:
    output = np.zeros(reference.shape, dtype=np.int8)
    wrong = (reference >= 0) & (reference != prediction)
    output[wrong] = 1
    output[wrong & (reference == herbaceous)] = 2
    output[wrong & (reference == cropland)] = 3
    output[reference < 0] = 4
    return output


def main() -> int:
    args = parse_args()
    if args.tile_size < 1 or args.tile_size % 16:
        raise SystemExit("--tile-size must be a positive multiple of 16")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("--output-dir must be empty")
    names_b0, b0_confusions = _load_confusions(args.b0_confusions)
    names_b1, b1_confusions = _load_confusions(args.b1_confusions)
    if names_b0 != names_b1:
        raise ValueError("B0 and B1 confusion files use different leaf orders")
    selection = select_oem_qualitative_identifiers(
        b0_confusions, b1_confusions, names_b0, min_reference_pixels=args.min_reference_pixels
    )
    identifiers = [str(item["identifier"]) for item in selection["selected"].values()]

    taxonomy, _ = load_taxonomy(ROOT / "ontology" / "land_cover_taxonomy_v0.yaml")
    mapping = load_dataset_mapping(ROOT / "ontology" / "dataset_label_mappings_v0.yaml", "openearthmap", taxonomy)
    node_to_id = stable_node_index(taxonomy.nodes)
    leaves = build_taxonomy_leaf_index(taxonomy)
    if tuple(leaves.leaf_names) != names_b0:
        raise ValueError("confusion leaf order does not match the frozen taxonomy")
    prompts = load_leaf_prompts(args.prompt_config, leaves.leaf_names)
    discovery = discover_external_pairs("openearthmap", args.data_root)
    pairs = exclude_openearthmap_cities(discovery.pairs, ("paris",))
    selected_pairs = tuple(pair for pair in pairs if pair.identifier in identifiers)
    if len(selected_pairs) != len(identifiers):
        missing = sorted(set(identifiers).difference(pair.identifier for pair in selected_pairs))
        raise ValueError(f"selected qualitative identifiers are absent from the supplied OEM root: {missing}")

    if not torch.cuda.is_available():
        raise SystemExit("qualitative replay requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    backbone = FrozenCLIPPatchEncoder(args.model_id).to(device)
    text_features = backbone.encode_text(prompts)
    b0_adapter = _checkpoint(args.b0_checkpoint, "b0", backbone.embedding_dim, prompts).to(device)
    b1_adapter = _checkpoint(args.b1_checkpoint, "b1", backbone.embedding_dim, prompts).to(device)
    herbaceous = leaves.leaf_names.index("herbaceous_vegetation")
    cropland = leaves.leaf_names.index("cropland")
    dataset = TaxonomyRasterDataset(selected_pairs, mapping, node_to_id, crop_size=None, seed=19)
    metadata = {
        "protocol": "protocols/P1_QUALITATIVE_SUPPLEMENT_PROTOCOL_2026-08-06.md",
        "selection": selection,
        "class_names": list(leaves.leaf_names),
        "model_id": args.model_id,
        "tile_size": args.tile_size,
        "b0_checkpoint": str(args.b0_checkpoint),
        "b0_checkpoint_sha256": _sha256(args.b0_checkpoint),
        "b1_checkpoint": str(args.b1_checkpoint),
        "b1_checkpoint_sha256": _sha256(args.b1_checkpoint),
        "b0_confusions_sha256": _sha256(args.b0_confusions),
        "b1_confusions_sha256": _sha256(args.b1_confusions),
        "arrays": {},
    }
    leaf_node_ids = torch.tensor([leaves.node_to_id[name] for name in leaves.leaf_names], device=device)
    for example in dataset:
        identifier = str(example["identifier"])
        image = example["image"]
        reference = _target_leaf_ids(example["target"].to(device), leaf_node_ids).cpu().numpy().astype(np.int16)
        b0_prediction = _predict(image, backbone, b0_adapter, text_features, args.tile_size, device)
        b1_prediction = _predict(image, backbone, b1_adapter, text_features, args.tile_size, device)
        rgb = image.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
        stem = identifier.replace("/", "__").replace("\\", "__").replace(".", "_")
        _write_csv(args.output_dir / f"{stem}_rgb.csv", rgb)
        _write_csv(args.output_dir / f"{stem}_reference.csv", reference)
        _write_csv(args.output_dir / f"{stem}_b0_prediction.csv", b0_prediction)
        _write_csv(args.output_dir / f"{stem}_b1_prediction.csv", b1_prediction)
        _write_csv(args.output_dir / f"{stem}_b0_error.csv", _error_codes(reference, b0_prediction, herbaceous, cropland))
        _write_csv(args.output_dir / f"{stem}_b1_error.csv", _error_codes(reference, b1_prediction, herbaceous, cropland))
        metadata["arrays"][identifier] = {"stem": stem, "height": int(reference.shape[0]), "width": int(reference.shape[1])}
    (args.output_dir / "qualitative_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"oem_qualitative_inputs_complete: examples={len(selected_pairs)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
