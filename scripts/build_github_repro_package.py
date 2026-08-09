"""Build a curated, GitHub-uploadable reviewer archive without raw datasets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    return parser.parse_args()


def copy_tree(source: Path, target: Path, *, patterns: tuple[str, ...] | None = None) -> None:
    if not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if patterns is not None and not any(path.match(pattern) for pattern in patterns):
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_current_paper_sources(target: Path) -> None:
    """Copy only the active manuscript source, never build output or backups."""
    root_files = (
        "main.tex",
        "math_commands.tex",
        "references.bib",
        "supplement.tex",
        "README_SUBMISSION.md",
        "SUBMISSION_GATE.md",
        "PAPER_CLAIM_AUDIT.json",
        "PAPER_CLAIM_AUDIT.md",
        "CITATION_AUDIT.json",
        "CITATION_AUDIT.md",
        "CITATION_AUDIT_EXTERNAL_RAW.json",
        "CITATION_RESOLUTION_CHECK.md",
    )
    for name in root_files:
        copy_file(ROOT / "paper" / name, target / name)
    copy_tree(ROOT / "paper" / "sections", target / "sections", patterns=("*.tex",))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output_dir: Path, zip_path: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    copy_file(ROOT / "reproducibility" / "README.md", output_dir / "README.md")
    copy_file(ROOT / "reproducibility" / "QUALITATIVE_REPLAY_STATUS.md", output_dir / "QUALITATIVE_REPLAY_STATUS.md")
    copy_file(ROOT / "requirements-e1.txt", output_dir / "requirements-e1.txt")
    for directory in ("configs", "ontology", "p1backbone", "p1data", "p1eval", "p1model", "p1train", "scripts", "tests", "protocols", "figures", "evidence"):
        copy_tree(ROOT / directory, output_dir / directory)
    copy_tree(ROOT / "reproducibility", output_dir / "reproducibility", patterns=("*.md", "*.cff"))
    copy_current_paper_sources(output_dir / "paper")
    # Include the designated clean-build PDFs when available. Do not discover
    # arbitrary PDFs under paper/: a desktop viewer can hold an older file open.
    final_main = ROOT / "submission" / "P1_JAG_MANUSCRIPT_FINAL_20260806.pdf"
    final_supplement = ROOT / "submission" / "P1_JAG_SUPPLEMENT_FINAL_20260806.pdf"
    copy_file(final_main if final_main.is_file() else ROOT / "paper" / "main.pdf", output_dir / "paper" / "main.pdf")
    copy_file(final_supplement if final_supplement.is_file() else ROOT / "paper" / "supplement.pdf", output_dir / "paper" / "supplement.pdf")
    # Include small provider-evidence screenshots; raw dataset imagery remains
    # excluded because only the audit evidence is needed for review.
    copy_tree(ROOT / "audits", output_dir / "audits", patterns=("*.json", "*.md", "*.png"))
    copy_tree(ROOT / "data" / "audits", output_dir / "data" / "audits", patterns=("*.json", "*.jsonl", "*.md"))
    copy_tree(ROOT / "archive" / "P1_PRIMARY_EXECUTION_ARCHIVE_20260806_v2" / "P1_PRIMARY_EXECUTION_ARCHIVE_20260806" / "outputs", output_dir / "results", patterns=("metrics.json", "run_config.json", "per_raster_confusions.json", "city_confusions.json", "per_identifier.json"))
    copy_file(ROOT / "P1_PRIMARY_EXECUTION_ARCHIVE_20260806.tar.gz.sha256", output_dir / "execution_archive.sha256")
    copy_file(ROOT / "P1_PRIMARY_EXECUTION_ARCHIVE_20260806.tar.gz", output_dir / "execution_archive.tar.gz")
    sums = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent).as_posix())
    print(f"github_repro_package_complete: files={len(sums)} zip={zip_path} sha256={sha256(zip_path)}")


if __name__ == "__main__":
    args = parse_args()
    build(args.output_dir, args.zip)
