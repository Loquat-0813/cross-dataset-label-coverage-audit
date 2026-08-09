"""Validate the versioned P1 land-cover taxonomy and dataset mappings."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    taxonomy = yaml.safe_load((ROOT / "ontology" / "land_cover_taxonomy_v0.yaml").read_text(encoding="utf-8"))
    mappings = yaml.safe_load((ROOT / "ontology" / "dataset_label_mappings_v0.yaml").read_text(encoding="utf-8"))
    nodes = taxonomy["nodes"]
    if taxonomy["root"] not in nodes or nodes[taxonomy["root"]]["parent"] is not None:
        raise SystemExit("taxonomy root is invalid")
    for name, node in nodes.items():
        parent = node["parent"]
        if parent is not None and parent not in nodes:
            raise SystemExit(f"{name}: unknown parent {parent}")
    allowed_relations = {"exact", "coarser_than", "ambiguous", "ignore", "unknown"}
    for dataset, spec in mappings["datasets"].items():
        scoring_status = spec.get("scoring_status")
        if scoring_status not in {"ready", "source_audit_pending"}:
            raise SystemExit(f"{dataset}: missing or invalid scoring_status")
        seen_names = set()
        seen_ids = set()
        for label in spec["labels"]:
            name = label["source_name"]
            if name in seen_names:
                raise SystemExit(f"{dataset}: duplicate source label {name}")
            seen_names.add(name)
            if label["relation"] not in allowed_relations:
                raise SystemExit(f"{dataset}/{name}: invalid relation")
            target = label["target"]
            source_id = label["source_id"]
            if source_id is None:
                raise SystemExit(f"{dataset}/{name}: source ID is not locked")
            if source_id in seen_ids:
                raise SystemExit(f"{dataset}: duplicate source ID {source_id}")
            seen_ids.add(source_id)
            if label["relation"] in {"ambiguous", "ignore", "unknown"} and target is not None:
                raise SystemExit(f"{dataset}/{name}: non-semantic label must not have a target")
            if label["relation"] in {"exact", "coarser_than"} and target not in nodes:
                raise SystemExit(f"{dataset}/{name}: target is absent from taxonomy")
    print(f"taxonomy_valid: {taxonomy['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
