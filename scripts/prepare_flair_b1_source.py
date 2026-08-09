"""Create a resumable, hash-recorded extraction of official FLAIR B1 archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1data.flair import FLAIR_ALL_DOMAINS, FLAIR_TRAIN_DOMAINS, FLAIR_VALIDATION_DOMAINS


TRAINING_ROOTS = frozenset({"aerial", "labels"})
EXTRACTION_PROFILE = "aerial_labels_only_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_safe_member(name: str) -> bool:
    member = PurePosixPath(name)
    return not member.is_absolute() and ".." not in member.parts


def inspect_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        unsafe = [info.filename for info in infos if not _is_safe_member(info.filename)]
        if unsafe:
            raise ValueError(f"{path.name}: archive has unsafe member paths: {unsafe[:3]}")
        names = [info.filename for info in infos]
        image_count = sum(name.rsplit("/", 1)[-1].startswith("IMG_") and name.lower().endswith(".tif") for name in names)
        mask_count = sum(name.rsplit("/", 1)[-1].startswith("MSK_") and name.lower().endswith(".tif") for name in names)
        if image_count == 0 or mask_count == 0:
            raise ValueError(f"{path.name}: expected official IMG_/MSK_ GeoTIFF members")
        if image_count != mask_count:
            raise ValueError(f"{path.name}: IMG/MSK count mismatch: {image_count} vs {mask_count}")
        roots = sorted({PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts})
        required_roots = {"aerial", "labels"}
        if not required_roots.issubset(roots):
            raise ValueError(f"{path.name}: missing official roots {sorted(required_roots - set(roots))}")
        return {
            "archive": path.name,
            "sha256": sha256_file(path),
            "compressed_bytes": path.stat().st_size,
            "uncompressed_bytes": sum(info.file_size for info in infos),
            "training_uncompressed_bytes": sum(
                info.file_size for info in infos if PurePosixPath(info.filename).parts[0] in TRAINING_ROOTS
            ),
            "member_count": len(infos),
            "training_member_count": sum(
                1 for info in infos if PurePosixPath(info.filename).parts[0] in TRAINING_ROOTS
            ),
            "image_tiff_count": image_count,
            "mask_tiff_count": mask_count,
            "top_level_directories": roots,
        }


def _extract_archive(path: Path, destination: Path, record: dict) -> str:
    marker = destination / ".p1_flair_extract.json"
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("sha256") != record["sha256"] or previous.get("extraction_profile") != EXTRACTION_PROFILE:
            raise ValueError(f"{destination}: existing extraction marker disagrees with the B1 archive/profile")
        return "reused"
    if destination.exists():
        raise FileExistsError(
            f"{destination}: extraction directory exists without a completion marker; "
            "inspect it manually rather than overwriting it"
        )
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not _is_safe_member(info.filename):
                    raise ValueError(f"{path.name}: unsafe member during extraction: {info.filename}")
                if PurePosixPath(info.filename).parts[0] in TRAINING_ROOTS:
                    archive.extract(info, destination)
        if not (destination / "aerial").is_dir() or not (destination / "labels").is_dir():
            raise ValueError(f"{path.name}: extraction did not create aerial/ and labels/")
        marker.write_text(json.dumps({**record, "extraction_profile": EXTRACTION_PROFILE}, indent=2), encoding="utf-8")
    except Exception:
        # Keep failed extractions visible and refuse to overwrite them on a retry.
        raise
    return "extracted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--extract", action="store_true", help="Extract archives after integrity and layout inspection.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archives = {path.stem: path for path in args.archive_root.glob("D*.zip") if path.is_file()}
    missing = sorted(set(FLAIR_ALL_DOMAINS) - set(archives))
    unexpected = sorted(set(archives) - set(FLAIR_ALL_DOMAINS))
    if missing or unexpected:
        raise SystemExit(f"official B1 archives mismatch: missing={missing}, unexpected={unexpected}")
    records = [inspect_archive(archives[domain]) for domain in FLAIR_ALL_DOMAINS]
    extraction_status: dict[str, str] = {}
    if args.extract:
        for domain, record in zip(FLAIR_ALL_DOMAINS, records):
            extraction_status[domain] = _extract_archive(archives[domain], args.extract_root / domain, record)
            print(f"flair_b1_archive_{extraction_status[domain]}: domain={domain}", flush=True)
    manifest = {
        "version": "flair-b1-source-manifest-v1",
        "official_release": "IGNF/FLAIR-1-2 data/train-val",
        "license": "Etalab Open Licence 2.0",
        "archive_root": str(args.archive_root),
        "extract_root": str(args.extract_root),
        "train_domains": list(FLAIR_TRAIN_DOMAINS),
        "validation_domains": list(FLAIR_VALIDATION_DOMAINS),
        "archives": records,
        "total_compressed_bytes": sum(record["compressed_bytes"] for record in records),
        "total_uncompressed_bytes": sum(record["uncompressed_bytes"] for record in records),
        "total_training_uncompressed_bytes": sum(record["training_uncompressed_bytes"] for record in records),
        "extraction_profile": EXTRACTION_PROFILE,
        "extraction_status": extraction_status,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "flair_b1_source_manifest_complete:",
        f"archives={len(records)}",
        f"images={sum(record['image_tiff_count'] for record in records)}",
        f"masks={sum(record['mask_tiff_count'] for record in records)}",
        f"extract={args.extract}",
        f"output={args.manifest_output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
