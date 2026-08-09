"""Record the conservative source/target geographic guard before B1 GPU use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1data.external import discover_external_pairs
from p1data.openearthmap_city_split import city_from_openearthmap_identifier, exclude_openearthmap_cities
from p1data.splits import identifier_set_sha256


REQUIRED_EXCLUDED_CITIES = ("paris",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--openearthmap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    raw = json.loads(args.raw_audit.read_text(encoding="utf-8"))
    if raw.get("source_mapping_status") != "source_audit_pending_geographic_overlap_review":
        raise SystemExit("raw audit is not in the expected pending-overlap state")
    if raw.get("herbaceous_raw_id") != 10 or raw.get("train", {}).get("herbaceous_pixel_count", 0) <= 0:
        raise SystemExit("raw audit does not prove usable raw-ID-10 training supervision")
    discovery = discover_external_pairs("openearthmap", args.openearthmap_root)
    all_cities = tuple(sorted({city_from_openearthmap_identifier(pair.identifier) for pair in discovery.pairs}))
    if not set(REQUIRED_EXCLUDED_CITIES).issubset(all_cities):
        raise SystemExit("required conservative exclusion city is absent from this OpenEarthMap release")
    eligible_pairs = exclude_openearthmap_cities(discovery.pairs, REQUIRED_EXCLUDED_CITIES)
    excluded_pairs = tuple(
        pair for pair in discovery.pairs if city_from_openearthmap_identifier(pair.identifier) in REQUIRED_EXCLUDED_CITIES
    )
    ready = dict(raw)
    ready.update(
        {
            "version": "flair-b1-source-readiness-v1",
            "source_mapping_status": "source_audit_ready",
            "geographic_overlap_review": {
                "review_type": "conservative source-country target-city exclusion",
                "source_geographic_scope": "France (official FLAIR aerial release)",
                "limitation": "FLAIR GeoTIFFs do not expose usable geographic tags; exact tile-coordinate intersection is unavailable.",
                "decision": "Exclude the complete OpenEarthMap Paris city block before every B0/B1 target replay.",
                "excluded_cities": list(REQUIRED_EXCLUDED_CITIES),
                "excluded_pair_count": len(excluded_pairs),
                "excluded_pair_identifier_set_sha256": identifier_set_sha256(pair.identifier for pair in excluded_pairs),
                "eligible_city_count": len(all_cities) - len(REQUIRED_EXCLUDED_CITIES),
                "eligible_pair_count": len(eligible_pairs),
                "eligible_pair_identifier_set_sha256": identifier_set_sha256(pair.identifier for pair in eligible_pairs),
                "all_paired_city_count": len(all_cities),
                "all_paired_raster_count": len(discovery.pairs),
                "unpaired_label_mask_count": len(discovery.unpaired_mask_identifiers),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ready, indent=2), encoding="utf-8")
    print(
        "flair_b1_source_readiness_complete:",
        f"excluded_cities={','.join(REQUIRED_EXCLUDED_CITIES)}",
        f"excluded_pairs={len(excluded_pairs)}",
        f"eligible_pairs={len(eligible_pairs)}",
        f"output={args.output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
