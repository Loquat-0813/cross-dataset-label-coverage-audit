"""Fail fast with exact dependency and CUDA diagnostics before E0/E1."""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    required = ("torch", "transformers", "safetensors")
    missing = []
    versions = {}
    for name in required:
        try:
            module = importlib.import_module(name)
        except ImportError:
            missing.append(name)
        else:
            versions[name] = getattr(module, "__version__", "unknown")
    if missing:
        print(f"e0_e1_preflight_missing: {', '.join(missing)}")
        return 2
    import torch

    if not torch.cuda.is_available():
        print("e0_e1_preflight_missing: CUDA")
        return 2
    print(
        "e0_e1_preflight_valid:",
        f"versions={versions}",
        f"gpu={torch.cuda.get_device_name(0)}",
        f"cuda={torch.version.cuda}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
