"""Repair the protocol metadata of an already-trained batch-control checkpoint.

This hotfix never changes adapter tensors. It writes a new checkpoint copy so
the original artifact remains available for audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib

import torch


EXPECTED_EXPERIMENT = "flair_coverage_b0_half_batch2"
OLD_MODE = "post_primary_loveda_exposure_control"
NEW_MODE = "post_primary_loveda_exposure_matched_batch_size_control"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repair_checkpoint(input_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {output_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no run_config")
    if config.get("experiment") != EXPECTED_EXPERIMENT or config.get("arm") != "b0_half_batch2":
        raise ValueError("checkpoint is not a b0_half_batch2 checkpoint")
    if config.get("protocol_mode") != OLD_MODE:
        raise ValueError(f"unexpected protocol mode: {config.get('protocol_mode')!r}")
    if "adapter_state_dict" not in checkpoint:
        raise ValueError("checkpoint has no adapter_state_dict")
    repaired = dict(checkpoint)
    repaired_config = dict(config)
    repaired_config["protocol_mode"] = NEW_MODE
    repaired_config["protocol_metadata_hotfix"] = {
        "reason": "training script wrote the exposure-control mode for the batch-size control arm",
        "weights_changed": False,
        "source_protocol_mode": OLD_MODE,
        "date": "2026-08-05",
    }
    repaired["run_config"] = repaired_config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(repaired, output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_sha256": _sha256(input_path),
        "output_sha256": _sha256(output_path),
        "protocol_mode": NEW_MODE,
        "weights_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = repair_checkpoint(args.input_checkpoint, args.output_checkpoint)
    print("b0_half_batch2_checkpoint_protocol_repaired:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
