"""Source-only calibration of frozen base prediction sets for coverage routing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from p1model.coverage_guard import conformal_membership, split_conformal_probability_threshold_from_scores
from p1train.loveda_e1 import base_leaf_logits


@dataclass(frozen=True)
class SourceConformalCalibration:
    probability_threshold: float
    alpha: float
    score_count: int
    sampled_tiles: int

    def as_dict(self) -> dict:
        return {
            "probability_threshold": self.probability_threshold,
            "alpha": self.alpha,
            "score_count": self.score_count,
            "sampled_tiles": self.sampled_tiles,
        }


@dataclass(frozen=True)
class SourceConformalDiagnostics:
    coverage: float
    mean_set_size: float
    valid_pixels: int

    def as_dict(self) -> dict:
        return {
            "coverage": self.coverage,
            "mean_set_size": self.mean_set_size,
            "valid_pixels": self.valid_pixels,
        }


@dataclass(frozen=True)
class SourceConformalGridDiagnostics:
    """Source-only summary of a candidate conformal routing threshold."""

    valid_pixels: int
    covered_labels: int
    set_size_sum: int
    routed_pixels: int
    root_routed_pixels: int
    parent_routed_pixels: int

    def as_dict(self) -> dict:
        return {
            "coverage": self.covered_labels / self.valid_pixels,
            "mean_set_size": self.set_size_sum / self.valid_pixels,
            "route_rate": self.routed_pixels / self.valid_pixels,
            "root_route_rate": self.root_routed_pixels / self.valid_pixels,
            "parent_route_rate": self.parent_routed_pixels / self.valid_pixels,
            "valid_pixels": self.valid_pixels,
        }


def target_nodes_to_leaf_ids(target: Tensor, leaf_node_ids: Tensor) -> Tensor:
    """Convert exact taxonomy node targets to leaf IDs while preserving ignore."""
    lookup = torch.full((int(leaf_node_ids.max()) + 1,), -1, device=target.device, dtype=torch.long)
    lookup[leaf_node_ids] = torch.arange(len(leaf_node_ids), device=target.device)
    output = torch.full_like(target, -1)
    valid = target >= 0
    output[valid] = lookup[target[valid]]
    return output


def _seed(identifier: str, top: int, left: int, seed: int) -> int:
    payload = f"{seed}:{identifier}:{top}:{left}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


@torch.inference_mode()
def collect_frozen_base_nonconformity_scores(
    dataset,
    backbone,
    text_features: Tensor,
    leaf_node_ids: Tensor,
    device: torch.device,
    tile_size: int,
    pixels_per_tile: int,
    seed: int,
) -> tuple[Tensor, int]:
    """Collect deterministic held-out frozen E0 nonconformity scores."""
    if tile_size < 1 or pixels_per_tile < 1:
        raise ValueError("tile_size and pixels_per_tile must be positive")
    scores: list[np.ndarray] = []
    sampled_tiles = 0
    for example in dataset:
        image = example["image"]
        target = example["target"]
        identifier = str(example["identifier"])
        height, width = target.shape
        if height % tile_size or width % tile_size:
            raise ValueError(f"{identifier}: image size {(height, width)} is not divisible by tile_size {tile_size}")
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                tile = image[:, top : top + tile_size, left : left + tile_size].unsqueeze(0).to(device)
                features = backbone.encode_image(tile)
                logits = base_leaf_logits(features, text_features)
                logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                target_leaf = target_nodes_to_leaf_ids(
                    target[top : top + tile_size, left : left + tile_size].unsqueeze(0).to(device), leaf_node_ids
                )
                valid = target_leaf >= 0
                if not valid.any():
                    continue
                probability = logits.softmax(dim=1).permute(0, 2, 3, 1)[valid]
                labels = target_leaf[valid].to(dtype=torch.long)
                nonconformity = (1.0 - probability.gather(1, labels.unsqueeze(1)).squeeze(1)).cpu().numpy()
                count = min(pixels_per_tile, nonconformity.size)
                generator = np.random.default_rng(_seed(identifier, top, left, seed))
                scores.append(generator.choice(nonconformity, size=count, replace=False))
                sampled_tiles += 1
    if not scores:
        raise ValueError("source calibration partition had no exact valid pixels")
    return torch.from_numpy(np.concatenate(scores).astype(np.float32, copy=False)), sampled_tiles


@torch.inference_mode()
def calibrate_frozen_base_probability_threshold(
    dataset,
    backbone,
    text_features: Tensor,
    leaf_node_ids: Tensor,
    device: torch.device,
    tile_size: int,
    pixels_per_tile: int,
    alpha: float,
    seed: int,
) -> SourceConformalCalibration:
    """Sample held-out source pixels and fit a split-conformal set.

    Pixel samples are drawn only from an identifier-disjoint source calibration
    partition. The resulting threshold is never optimized on external labels.
    """
    all_scores, sampled_tiles = collect_frozen_base_nonconformity_scores(
        dataset,
        backbone,
        text_features,
        leaf_node_ids,
        device,
        tile_size,
        pixels_per_tile,
        seed,
    )
    return SourceConformalCalibration(
        probability_threshold=split_conformal_probability_threshold_from_scores(all_scores, alpha),
        alpha=alpha,
        score_count=int(all_scores.numel()),
        sampled_tiles=sampled_tiles,
    )


def summarize_coverage_routing_grid(
    membership: Tensor,
    target_leaf_ids: Tensor,
    covered_leaf_mask: Tensor,
    coverage_node_lookup: Tensor,
    node_depth_by_id: Tensor,
) -> SourceConformalGridDiagnostics:
    """Summarize one conformal set tensor under the fixed coverage routing rule."""
    if membership.ndim != 4 or target_leaf_ids.shape != membership.shape[:1] + membership.shape[2:]:
        raise ValueError("membership and target_leaf_ids must align as [B, L, H, W] and [B, H, W]")
    if covered_leaf_mask.shape != (membership.shape[1],) or covered_leaf_mask.dtype != torch.bool:
        raise ValueError("covered_leaf_mask must be a bool vector aligned to membership leaves")
    valid = target_leaf_ids >= 0
    if not valid.any():
        return SourceConformalGridDiagnostics(0, 0, 0, 0, 0, 0)
    member_by_pixel = membership.permute(0, 2, 3, 1)[valid]
    labels = target_leaf_ids[valid].to(dtype=torch.long)
    label_covered = member_by_pixel.gather(1, labels.unsqueeze(1)).sum()
    set_sizes = member_by_pixel.sum(dim=1)
    uncovered_member = membership & ~covered_leaf_mask.view(1, -1, 1, 1)
    route = uncovered_member.any(dim=1)[valid]
    weights = (1 << torch.arange(membership.shape[1], device=membership.device, dtype=torch.long)).view(1, -1, 1, 1)
    set_code = (membership.to(dtype=torch.long) * weights).sum(dim=1)[valid]
    node_ids = coverage_node_lookup[set_code]
    node_depth = node_depth_by_id[node_ids]
    return SourceConformalGridDiagnostics(
        valid_pixels=int(labels.numel()),
        covered_labels=int(label_covered),
        set_size_sum=int(set_sizes.sum()),
        routed_pixels=int(route.sum()),
        root_routed_pixels=int((route & (node_depth == 0)).sum()),
        parent_routed_pixels=int((route & (node_depth == 1)).sum()),
    )


@torch.inference_mode()
def evaluate_frozen_base_coverage_routing_grid(
    dataset,
    backbone,
    text_features: Tensor,
    leaf_node_ids: Tensor,
    device: torch.device,
    tile_size: int,
    thresholds: dict[str, float],
    covered_leaf_mask: Tensor,
    coverage_node_lookup: Tensor,
    node_depth_by_id: Tensor,
) -> dict[str, SourceConformalGridDiagnostics]:
    """Evaluate predeclared source-only conformal routing candidates in one pass."""
    if tile_size < 1 or not thresholds:
        raise ValueError("tile_size must be positive and thresholds must be non-empty")
    totals = {key: SourceConformalGridDiagnostics(0, 0, 0, 0, 0, 0) for key in thresholds}
    for example in dataset:
        image = example["image"]
        target = example["target"]
        height, width = target.shape
        if height % tile_size or width % tile_size:
            raise ValueError(f"{example['identifier']}: image size {(height, width)} is not divisible by tile_size {tile_size}")
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                tile = image[:, top : top + tile_size, left : left + tile_size].unsqueeze(0).to(device)
                features = backbone.encode_image(tile)
                logits = F.interpolate(
                    base_leaf_logits(features, text_features),
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )
                target_leaf = target_nodes_to_leaf_ids(
                    target[top : top + tile_size, left : left + tile_size].unsqueeze(0).to(device), leaf_node_ids
                )
                for key, threshold in thresholds.items():
                    summary = summarize_coverage_routing_grid(
                        conformal_membership(logits, threshold),
                        target_leaf,
                        covered_leaf_mask,
                        coverage_node_lookup,
                        node_depth_by_id,
                    )
                    total = totals[key]
                    totals[key] = SourceConformalGridDiagnostics(
                        valid_pixels=total.valid_pixels + summary.valid_pixels,
                        covered_labels=total.covered_labels + summary.covered_labels,
                        set_size_sum=total.set_size_sum + summary.set_size_sum,
                        routed_pixels=total.routed_pixels + summary.routed_pixels,
                        root_routed_pixels=total.root_routed_pixels + summary.root_routed_pixels,
                        parent_routed_pixels=total.parent_routed_pixels + summary.parent_routed_pixels,
                    )
    if any(not summary.valid_pixels for summary in totals.values()):
        raise ValueError("source conformal routing grid had no exact valid pixels")
    return totals


@torch.inference_mode()
def evaluate_frozen_base_prediction_sets(
    dataset,
    backbone,
    text_features: Tensor,
    leaf_node_ids: Tensor,
    device: torch.device,
    tile_size: int,
    probability_threshold: float,
) -> SourceConformalDiagnostics:
    """Measure frozen E0 prediction-set coverage on a source-only dataset."""
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    covered = 0
    set_size_sum = 0
    valid_pixels = 0
    for example in dataset:
        image = example["image"]
        target = example["target"]
        height, width = target.shape
        if height % tile_size or width % tile_size:
            raise ValueError(f"{example['identifier']}: image size {(height, width)} is not divisible by tile_size {tile_size}")
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                tile = image[:, top : top + tile_size, left : left + tile_size].unsqueeze(0).to(device)
                features = backbone.encode_image(tile)
                logits = F.interpolate(
                    base_leaf_logits(features, text_features),
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )
                target_leaf = target_nodes_to_leaf_ids(
                    target[top : top + tile_size, left : left + tile_size].unsqueeze(0).to(device), leaf_node_ids
                )
                valid = target_leaf >= 0
                if not valid.any():
                    continue
                membership = conformal_membership(logits, probability_threshold)
                membership_by_pixel = membership.permute(0, 2, 3, 1)[valid]
                labels = target_leaf[valid].to(dtype=torch.long)
                covered += int(membership_by_pixel.gather(1, labels.unsqueeze(1)).sum())
                set_size_sum += int(membership_by_pixel.sum())
                valid_pixels += int(labels.numel())
    if not valid_pixels:
        raise ValueError("source conformal diagnostics had no exact valid pixels")
    return SourceConformalDiagnostics(
        coverage=covered / valid_pixels,
        mean_set_size=set_size_sum / valid_pixels,
        valid_pixels=valid_pixels,
    )
