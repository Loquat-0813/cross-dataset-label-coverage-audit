"""Validate the frozen P1 dataset and open-vocabulary split protocol."""

from pathlib import Path
import sys

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency diagnostic
    raise SystemExit("PyYAML is required to validate configs/ov_splits.yaml") from exc


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "ov_splits.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    classes = set(config["classes"])
    if len(classes) != 15:
        raise SystemExit(f"expected 15 foreground classes, got {len(classes)}")
    for name in ("split_a", "split_b"):
        split = config[name]
        unseen = set(split["unseen"])
        seen = set(split["seen"])
        if len(unseen) != 4 or len(seen) != 11:
            raise SystemExit(
                f"{name}: expected 4 unseen and 11 seen foreground classes "
                "(12 seen labels including background); "
                f"got {len(unseen)} and {len(seen)}"
            )
        if unseen & seen or unseen | seen != classes:
            raise SystemExit(f"{name}: seen/unseen partition is invalid")
    roles = yaml.safe_load((root / "configs" / "dataset_roles.yaml").read_text(encoding="utf-8"))
    active = roles["tracks"]["taxonomy_aware_land_cover_transfer"]
    protocols = active["protocols"]
    if protocols["single_source_generalization"]["train_source"] != "loveda":
        raise SystemExit("single-source protocol must retain LoveDA as its training source")
    if protocols["single_source_generalization"]["external_evaluation"] != ["openearthmap", "landcoverai"]:
        raise SystemExit("single-source external evaluation datasets are not frozen")
    if protocols["heterogeneous_source_partial_label"]["train_sources"] != ["loveda", "openearthmap"]:
        raise SystemExit("heterogeneous protocol source datasets are not frozen")
    if protocols["heterogeneous_source_partial_label"]["external_evaluation"] != ["landcoverai"]:
        raise SystemExit("heterogeneous protocol external evaluation dataset is not frozen")
    expected_paths = {
        "loveda": "data/raw/loveda",
        "openearthmap": "data/raw/openearthmap",
        "landcoverai": "data/raw/landcoverai",
    }
    for dataset, expected_path in expected_paths.items():
        actual_path = roles["datasets"][dataset].get("local_path")
        if actual_path != expected_path:
            raise SystemExit(f"{dataset}: expected local_path {expected_path}, got {actual_path}")
    print("protocol_valid: p1-ov-splits-v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
