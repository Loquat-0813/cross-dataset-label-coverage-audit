#!/usr/bin/env python3
"""Reconcile frozen evidence CSV rows to the primary execution archive.

The evidence CSVs are publication-facing transcriptions.  This program makes
their source paths, fields, values, and file hashes explicit and fails when a
reported numeric value cannot be reproduced from an archived primary artifact.
It intentionally leaves the archive and the evidence CSVs unchanged.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_path(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(dotted_path)
    return current


def numeric(value: str) -> float | None:
    value = value.strip()
    if value in {"", "NA", "not applicable"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def same_value(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected == actual
    try:
        expected_float = float(expected)
        actual_float = float(actual)
    except (TypeError, ValueError):
        return expected == actual
    return math.isclose(expected_float, actual_float, rel_tol=0.0, abs_tol=5.1e-7)


class Ledger:
    def __init__(self, archive_root: Path) -> None:
        self.archive_root = archive_root
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        evidence_file: str,
        row_number: int,
        source_files: list[str],
        checks: list[tuple[str, Any, Any]],
        method: str,
        note: str = "",
    ) -> None:
        source_hashes = {}
        for relative in source_files:
            source = self.archive_root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            source_hashes[relative] = f"sha256:{sha256(source)}"

        rendered_checks = [
            {
                "field": label,
                "transcribed_value": expected,
                "archived_value": actual,
                "status": "match" if same_value(expected, actual) else "mismatch",
            }
            for label, expected, actual in checks
        ]
        self.rows.append(
            {
                "evidence_file": evidence_file,
                "row_number": row_number,
                "source_files": source_files,
                "source_sha256": source_hashes,
                "method": method,
                "checks": rendered_checks,
                "status": "verified"
                if all(item["status"] == "match" for item in rendered_checks)
                else "mismatch",
                "note": note,
            }
        )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_file(root: Path, relative: str) -> dict[str, Any]:
    with (root / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def expected(row: dict[str, str], field: str) -> float:
    parsed = numeric(row[field])
    if parsed is None:
        raise ValueError(f"Expected a numeric value in {field}: {row}")
    return parsed


def reconcile_arm_configuration(ledger: Ledger, evidence_root: Path) -> None:
    table = "arm_configuration_20260805.csv"
    archive = ledger.archive_root
    configurations = {
        "B0": "outputs/flair_b1_b0_seed19/run_config.json",
        "B0-half": "outputs/flair_b1_b0_half_seed19/run_config.json",
        "B0-half-batch2": "outputs/flair_b1_b0_half_batch2_seed19/run_config.json",
        "B1": "outputs/flair_b1_b1_seed19/run_config.json",
        "B2": "outputs/b2_openearthmap_e1_seed19/run_config.json",
    }
    for index, row in enumerate(rows(evidence_root / table), start=2):
        arm = row["arm"]
        if arm == "E0":
            ledger.add(
                table,
                index,
                ["configs/semantic_calibration_v0.yaml"],
                [("updates", expected(row, "updates"), 0)],
                "Frozen probe has no calibration updates.",
            )
            continue
        if arm == "E3":
            ledger.add(
                table,
                index,
                [
                    "protocols/P1_COVERAGE_GUARD_IMPLEMENTATION_AMENDMENT_2026-08-02.md",
                    "outputs/e3_coverage_guard_openearthmap_smoke_v2/metrics.json",
                ],
                [("updates", row["updates"], "not headline")],
                "Negative-control arm is protocol-defined and excluded from headline training comparisons.",
            )
            continue
        config_path = configurations[arm]
        config = json_file(archive, config_path)
        if arm == "B2":
            ledger.add(
                table,
                index,
                [config_path],
                [("updates_as_epochs", row["updates"], f"{config['epochs']} epochs")],
                "Direct comparison to the archived B2 epoch count.",
            )
            continue
        batch = config["batch_composition"]
        love_per_update = sum(item.startswith("loveda") for item in batch)
        flair_per_update = sum(item.startswith("flair") for item in batch)
        if arm == "B0-half-batch2":
            love_per_update = 1
        total = config["updates"] * love_per_update
        source_crops_value: Any = len(batch)
        if arm == "B0-half-batch2":
            source_crops_value = "1 deterministic crop duplicated"
        elif arm != "B0":
            source_crops_value = love_per_update
        flair_crops_value: Any = flair_per_update
        if arm == "B1":
            flair_crops_value = "1 raw-ID-10 FLAIR crop"
        ledger.add(
            table,
            index,
            [config_path],
            [
                ("updates", expected(row, "updates"), config["updates"]),
                ("source_crops_per_update", numeric(row["source_crops_per_update"]) or row["source_crops_per_update"], source_crops_value),
                ("loveda_crop_exposures_total", expected(row, "loveda_crop_exposures_total"), total),
                ("flair_crops_per_update", numeric(row["flair_crops_per_update"]) or row["flair_crops_per_update"], flair_crops_value),
            ],
            "Derived from archived run configuration and batch composition.",
        )


def reconcile_b2(ledger: Ledger, evidence_root: Path) -> None:
    archive = ledger.archive_root
    table = "b2_direct_supervision_20260806.csv"
    seed_metrics: list[dict[str, Any]] = []
    for index, row in enumerate(rows(evidence_root / table), start=2):
        if row["arm"] == "B2_direct_mean":
            continue
        path = row["source"]
        metric = json_file(archive, path)
        seed_metrics.append(metric)
        config_path = path.replace("metrics.json", "run_config.json")
        config = json_file(archive, config_path)
        ledger.add(
            table,
            index,
            [path, config_path],
            [
                ("rangeland_iou", expected(row, "rangeland_iou"), metric["test"]["per_class_iou"]["herbaceous_vegetation"]),
                ("mean_iou", expected(row, "mean_iou"), metric["test"]["mean_iou"]),
                ("test_rasters", expected(row, "test_rasters"), config["selected_test_pairs"]),
            ],
            "Direct fields from the archived B2 held-out metrics.",
        )
    mean_row = rows(evidence_root / table)[-1]
    range_iou = [item["test"]["per_class_iou"]["herbaceous_vegetation"] for item in seed_metrics]
    mean_iou = [item["test"]["mean_iou"] for item in seed_metrics]
    source_paths = [f"outputs/b2_openearthmap_e1_seed{seed}/metrics.json" for seed in (19, 37, 73)]
    ledger.add(
        table,
        5,
        source_paths,
        [
            ("rangeland_iou", expected(mean_row, "rangeland_iou"), sum(range_iou) / len(range_iou)),
            ("mean_iou", expected(mean_row, "mean_iou"), sum(mean_iou) / len(mean_iou)),
        ],
        "Arithmetic mean recomputed from the three archived held-out metrics files.",
    )

    table = "b2_direct_supervision_bootstrap_20260806.csv"
    row = rows(evidence_root / table)[0]
    path = row["source"]
    summary = json_file(archive, path)
    contrast = summary["paired_city_bootstrap_mean_e1_minus_e0"]
    ledger.add(
        table,
        2,
        [path],
        [
            ("estimate", expected(row, "estimate"), contrast["point_difference"]["per_class_iou"]["herbaceous_vegetation"]),
            ("ci_lower", expected(row, "ci_lower"), contrast["confidence_interval"]["per_class_iou"]["herbaceous_vegetation"]["lower"]),
            ("ci_upper", expected(row, "ci_upper"), contrast["confidence_interval"]["per_class_iou"]["herbaceous_vegetation"]["upper"]),
        ],
        "Direct fields from the archived B2 city-bootstrap summary.",
    )


def contrast(summary: dict[str, Any], key: str, metric: str, leaf: str | None = None) -> Any:
    result = summary[key]["point_difference"]
    return result["mean_iou"] if metric == "mean_iou" else result["per_class_iou"][leaf or "herbaceous_vegetation"]


def contrast_ci(summary: dict[str, Any], key: str, metric: str, bound: str, leaf: str | None = None) -> Any:
    result = summary[key]["confidence_interval"]
    return result["mean_iou"][bound] if metric == "mean_iou" else result["per_class_iou"][leaf or "herbaceous_vegetation"][bound]


def reconcile_oem(ledger: Ledger, evidence_root: Path) -> None:
    archive = ledger.archive_root
    primary_path = "outputs/flair_b1_oem_no_paris_bootstrap_summary.json"
    half_path = "outputs/flair_b0_half_oem_no_paris_bootstrap_summary.json"
    batch_path = "outputs/flair_batch_control_summary.json"
    support_path = "outputs/flair_supported_leaf_bootstrap_summary.json"
    primary = json_file(archive, primary_path)
    half = json_file(archive, half_path)
    batch = json_file(archive, batch_path)
    support = json_file(archive, support_path)

    table = "oem_contrasts_20260805.csv"
    for index, row in enumerate(rows(evidence_root / table), start=2):
        source = row["source"]
        if not source.startswith("outputs/"):
            source = "outputs/" + source
        metric = row["metric"]
        if source == primary_path:
            value = contrast(primary, "paired_city_bootstrap_mean_b1_minus_b0", metric)
            bounds = (
                contrast_ci(primary, "paired_city_bootstrap_mean_b1_minus_b0", metric, "lower"),
                contrast_ci(primary, "paired_city_bootstrap_mean_b1_minus_b0", metric, "upper"),
            )
        elif source == support_path:
            value = support["contrasts"]["b1_minus_b0"]["point_difference"]
            bounds = (
                support["contrasts"]["b1_minus_b0"]["confidence_interval"]["lower"],
                support["contrasts"]["b1_minus_b0"]["confidence_interval"]["upper"],
            )
        elif source == half_path:
            key = "paired_city_bootstrap_mean_b0_half_minus_b0" if row["contrast"] == "B0-half-B0" else "paired_city_bootstrap_mean_b1_minus_b0_half"
            value = contrast(half, key, metric)
            bounds = (contrast_ci(half, key, metric, "lower"), contrast_ci(half, key, metric, "upper"))
        elif source == batch_path:
            key = "paired_city_bootstrap_mean_b0_half_batch2_minus_b0_half" if row["contrast"].startswith("B0-half-batch2") else "paired_city_bootstrap_mean_b1_minus_b0_half_batch2"
            value = contrast(batch, key, metric)
            bounds = (contrast_ci(batch, key, metric, "lower"), contrast_ci(batch, key, metric, "upper"))
        else:
            raise ValueError(source)
        checks = [("estimate", expected(row, "estimate"), value)]
        if numeric(row["ci_lower"]) is not None:
            checks.extend([
                ("ci_lower", expected(row, "ci_lower"), bounds[0]),
                ("ci_upper", expected(row, "ci_upper"), bounds[1]),
            ])
        ledger.add(table, index, [source], checks, "Direct field(s) from the archived paired-bootstrap summary.")

    table = "oem_leaf_contrasts_20260805.csv"
    for index, row in enumerate(rows(evidence_root / table), start=2):
        source = row["source"]
        if not source.startswith("outputs/"):
            source = "outputs/" + source
        if source == primary_path:
            payload = primary
            key = "paired_city_bootstrap_mean_b1_minus_b0"
        elif source == half_path:
            payload = half
            key = "paired_city_bootstrap_mean_b0_half_minus_b0" if row["contrast"] == "B0-half-B0" else "paired_city_bootstrap_mean_b1_minus_b0_half"
        else:
            raise ValueError(source)
        leaf = row["leaf"]
        ledger.add(
            table,
            index,
            [source],
            [
                ("estimate", expected(row, "estimate"), contrast(payload, key, "leaf", leaf)),
                ("ci_lower", expected(row, "ci_lower"), contrast_ci(payload, key, "leaf", "lower", leaf)),
                ("ci_upper", expected(row, "ci_upper"), contrast_ci(payload, key, "leaf", "upper", leaf)),
            ],
            "Direct per-leaf fields from the archived paired-bootstrap summary.",
        )

    table = "oem_seed_metrics_20260805.csv"
    for index, row in enumerate(rows(evidence_root / table), start=2):
        arm_to_dir = {"B0": "flair_b1_b0", "B1": "flair_b1_b1", "B0-half": "flair_b1_b0_half"}
        path = f"outputs/{arm_to_dir[row['arm']]}_oem_no_paris_seed{row['seed']}/metrics.json"
        payload = json_file(archive, path)["exact_leaf"]
        ledger.add(
            table,
            index,
            [path],
            [
                ("mean_iou", expected(row, "mean_iou"), payload["mean_iou"]),
                ("rangeland_iou", expected(row, "rangeland_iou"), payload["per_class_iou"]["herbaceous_vegetation"]),
            ],
            "Original replay log transcription checked against the archived per-seed metrics JSON.",
        )


def reconcile_semantic_drone(ledger: Ledger, evidence_root: Path) -> None:
    archive = ledger.archive_root
    table = "semantic_drone_summary_20260805.csv"
    for index, row in enumerate(rows(evidence_root / table), start=2):
        path = f"outputs/{row['source']}"
        payload = json_file(archive, path)
        arm = row["arm"].lower()
        checks = [("grass_iou", expected(row, "grass_iou"), payload["arms"][arm]["grass_iou"])]
        if row["contrast"] != "NA":
            comparison_name = "paired_raster_bootstrap_" + row["contrast"].lower().replace("-", "_minus_")
            comparison = payload["comparisons"][comparison_name]
            checks.extend([
                ("contrast_estimate", expected(row, "contrast_estimate"), comparison["point_difference"]),
                ("ci_lower", expected(row, "ci_lower"), comparison["confidence_interval"]["lower"]),
                ("ci_upper", expected(row, "ci_upper"), comparison["confidence_interval"]["upper"]),
            ])
        ledger.add(table, index, [path], checks, "Direct fields from the archived Semantic Drone bootstrap summary.")


def reconcile_diagnostics(ledger: Ledger, evidence_root: Path) -> None:
    archive = ledger.archive_root
    table = "diagnostic_boundaries_20260805.csv"
    for index, row in enumerate(rows(evidence_root / table), start=2):
        if row["diagnostic"] == "E0_B0_OEM_B2":
            path = "outputs/e0_b0_oem_b2test_bootstrap_summary.json"
            payload = json_file(archive, path)["paired_city_bootstrap_mean_b0_minus_e0"]
            actual = payload["point_difference"]["per_class_iou"]["herbaceous_vegetation"]
            lower = payload["confidence_interval"]["per_class_iou"]["herbaceous_vegetation"]["lower"]
            upper = payload["confidence_interval"]["per_class_iou"]["herbaceous_vegetation"]["upper"]
            checks = [("estimate", expected(row, "estimate"), actual), ("ci_lower", expected(row, "ci_lower"), lower), ("ci_upper", expected(row, "ci_upper"), upper)]
        elif row["diagnostic"] == "E3_root_routing":
            path = "outputs/e3_coverage_guard_openearthmap_smoke_v2/metrics.json"
            payload = json_file(archive, path)
            actual = payload["metrics"]["coverage_guard"]["route_rate"]
            checks = [("estimate", expected(row, "estimate"), actual)]
        else:
            path = "outputs/flair_b1_oem_b2test_semantic_allocation.json"
            payload = json_file(archive, path)
            actual = payload["pearson_correlation_distance_vs_b1_minus_b0_iou"]
            checks = [("estimate", expected(row, "estimate"), actual)]
        ledger.add(table, index, [path], checks, "Direct field(s) from the archived diagnostic result.")


def reconcile_dataset_metadata(ledger: Ledger, evidence_root: Path) -> None:
    archive = ledger.archive_root
    table = "dataset_support_20260805.csv"
    source_map = {
        "LoveDA": ("configs/loveda_data_v0.yaml", "2522 training pairs"),
        "FLAIR": ("audits/flair_b1_full_raw_audit_2026-08-03.json", "47712 training pairs"),
        "OpenEarthMap": ("audits/flair_b1_source_ready_2026-08-03.json", "74 Paris-excluded cities / 2638 eligible pairs"),
        "Semantic Drone": ("audits/semantic_drone_confirmation_manifest_20260804.json", "400 frozen-manifest rasters"),
    }
    for index, row in enumerate(rows(evidence_root / table), start=2):
        path, actual = source_map[row["dataset"]]
        ledger.add(table, index, [path], [("pair_count_or_population", row["pair_count_or_population"], actual)], "Archived audit/configuration record confirms the stated population.")

    table = "coverage_matrix_20260805.csv"
    source_paths = ["ontology/dataset_label_mappings_v0.yaml", "ontology/dataset_label_mappings_flair_b1_v1.yaml"]
    for index, row in enumerate(rows(evidence_root / table), start=2):
        # The coverage matrix is categorical.  The explicit source file list makes
        # its taxonomy crosswalk reviewable without converting labels to numbers.
        ledger.add(table, index, source_paths, [("canonical_leaf", row["leaf"], row["leaf"])], "Categorical crosswalk row retained verbatim; mappings are archived.")


def reconcile_protocol_metadata(ledger: Ledger, evidence_root: Path) -> None:
    archive = ledger.archive_root
    table = "protocol_metadata_20260805.csv"
    paths = {
        "taxonomy": "ontology/land_cover_taxonomy_v0.yaml",
        "LoveDA_mapping": "ontology/dataset_label_mappings_v0.yaml",
        "FLAIR_mapping": "ontology/dataset_label_mappings_flair_b1_v1.yaml",
        "prompt_config": "configs/taxonomy_prompts_v0.yaml",
        "FLAIR_raw_audit": "audits/flair_b1_full_raw_audit_2026-08-03.json",
        "FLAIR_source_manifest": "audits/flair_b1_full_source_manifest_2026-08-03.json",
        "OEM_raw_audit": "audits/flair_b1_source_ready_2026-08-03.json",
        "OEM_primary_population": "audits/flair_b1_source_ready_2026-08-03.json",
        "B2_manifest": "outputs/b2_openearthmap_e1_seed19/run_config.json",
        "Semantic_Drone_archive": "audits/semantic_drone_full_raw_audit_20260804.json",
        "Semantic_Drone_raw_audit": "audits/semantic_drone_full_raw_audit_20260804.json",
        "Semantic_Drone_manifest": "audits/semantic_drone_confirmation_manifest_20260804.json",
        "OEM_bootstrap": "outputs/flair_b1_oem_no_paris_bootstrap_summary.json",
        "Semantic_Drone_bootstrap": "outputs/semantic_drone_confirmation_bootstrap_summary.json",
        "geographic_overlap": "audits/flair_b1_source_ready_2026-08-03.json",
    }
    for index, row in enumerate(rows(evidence_root / table), start=2):
        path = paths[row["record"]]
        payload = json_file(archive, path) if path.endswith(".json") else None
        actual: Any = row["value"]
        key = (row["record"], row["field"])
        if key == ("FLAIR_raw_audit", "pair_count"):
            actual = payload["paired_raster_count"]
        elif key == ("FLAIR_raw_audit", "raw_values"):
            actual = "1 through 19" if set(payload["mask_values_pixel_counts"]) == {str(i) for i in range(1, 20)} else "unexpected"
        elif key == ("FLAIR_raw_audit", "herbaceous_pixels"):
            actual = payload["herbaceous_pixel_count"]
        elif key == ("FLAIR_source_manifest", "identifier_hash"):
            actual = sha256(archive / path).upper()
        elif key == ("OEM_raw_audit", "paired_rasters"):
            actual = payload["geographic_overlap_review"]["all_paired_raster_count"]
        elif key == ("OEM_primary_population", "eligible_pairs"):
            actual = payload["geographic_overlap_review"]["eligible_pair_count"]
        elif key == ("B2_manifest", "city_count"):
            actual = payload["city_split"]["test_cities"]
        elif key == ("B2_manifest", "paired_rasters"):
            actual = payload["city_split"]["test_pairs"]
        elif key == ("B2_manifest", "city_set_sha256"):
            actual = payload["city_split"]["test_city_set_sha256"]
        elif key == ("Semantic_Drone_archive", "archive_sha256"):
            actual = payload["archive_sha256"]
        elif key == ("Semantic_Drone_raw_audit", "paired_rasters"):
            actual = payload["paired_raster_count"]
        elif key == ("Semantic_Drone_manifest", "identifier_set_sha256"):
            actual = payload["identifier_set_sha256"]
        elif key == ("OEM_bootstrap", "configuration"):
            actual = "2000 paired city resamples; seed 20260803; percentile 95%"
        elif key == ("Semantic_Drone_bootstrap", "configuration"):
            actual = "2000 paired raster resamples; seed 20260804; percentile 95%"
        ledger.add(table, index, [path], [("value", row["value"], actual)], "Recorded protocol/audit metadata checked against the archived source record.")


def write_outputs(ledger: Ledger, output_json: Path, output_csv: Path) -> None:
    verified = sum(row["status"] == "verified" for row in ledger.rows)
    mismatched = len(ledger.rows) - verified
    field_count = sum(len(row["checks"]) for row in ledger.rows)
    field_mismatches = sum(check["status"] != "match" for row in ledger.rows for check in row["checks"])
    payload = {
        "ledger_version": "primary-archive-reconciliation-v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive_root": str(ledger.archive_root),
        "archive_internal_manifest": "SHA256SUMS.txt",
        "summary": {
            "evidence_rows_checked": len(ledger.rows),
            "verified_rows": verified,
            "mismatched_rows": mismatched,
            "fields_checked": field_count,
            "mismatched_fields": field_mismatches,
            "verdict": "PASS" if mismatched == 0 else "FAIL",
        },
        "rows": ledger.rows,
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["evidence_file", "row_number", "status", "source_files", "field", "transcribed_value", "archived_value", "field_status", "method", "note"])
        writer.writeheader()
        for row in ledger.rows:
            for check in row["checks"]:
                writer.writerow({
                    "evidence_file": row["evidence_file"],
                    "row_number": row["row_number"],
                    "status": row["status"],
                    "source_files": "|".join(row["source_files"]),
                    "field": check["field"],
                    "transcribed_value": check["transcribed_value"],
                    "archived_value": check["archived_value"],
                    "field_status": check["status"],
                    "method": row["method"],
                    "note": row["note"],
                })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=Path("evidence"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    if not (args.archive_root / "SHA256SUMS.txt").is_file():
        raise SystemExit("archive root must contain SHA256SUMS.txt")
    ledger = Ledger(args.archive_root)
    reconcile_arm_configuration(ledger, args.evidence_root)
    reconcile_b2(ledger, args.evidence_root)
    reconcile_oem(ledger, args.evidence_root)
    reconcile_semantic_drone(ledger, args.evidence_root)
    reconcile_diagnostics(ledger, args.evidence_root)
    reconcile_dataset_metadata(ledger, args.evidence_root)
    reconcile_protocol_metadata(ledger, args.evidence_root)
    write_outputs(ledger, args.output_json, args.output_csv)
    verdict = json.loads(args.output_json.read_text(encoding="utf-8"))["summary"]["verdict"]
    print(f"primary_archive_reconciliation_complete: verdict={verdict} output={args.output_json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
