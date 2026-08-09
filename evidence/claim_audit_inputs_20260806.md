# Expanded Claim-Audit Input Boundary

The final paper-to-evidence audit must read every active manuscript source under
`paper/sections/`, the root TeX files under `paper/`, the five generated table
TeX files under `figures/tables/`, and the archived raw-audit JSON files and
summary JSON files listed below. This boundary is intentionally broader than
the earlier output-only audit so that dataset support, raw-label counts, hash
prefixes, and table cells are independently traceable.

Raw-audit inputs:

- `archive/P1_PRIMARY_EXECUTION_ARCHIVE_20260806_v2/P1_PRIMARY_EXECUTION_ARCHIVE_20260806/audits/flair_b1_full_raw_audit_2026-08-03.json`
- `archive/P1_PRIMARY_EXECUTION_ARCHIVE_20260806_v2/P1_PRIMARY_EXECUTION_ARCHIVE_20260806/audits/flair_b1_source_ready_2026-08-03.json`
- `archive/P1_PRIMARY_EXECUTION_ARCHIVE_20260806_v2/P1_PRIMARY_EXECUTION_ARCHIVE_20260806/audits/openearthmap_full_audit_v1.json`
- `archive/P1_PRIMARY_EXECUTION_ARCHIVE_20260806_v2/P1_PRIMARY_EXECUTION_ARCHIVE_20260806/audits/semantic_drone_full_raw_audit_20260804.json`
- `archive/P1_PRIMARY_EXECUTION_ARCHIVE_20260806_v2/P1_PRIMARY_EXECUTION_ARCHIVE_20260806/audits/semantic_drone_confirmation_manifest_20260804.json`

Summary inputs:

- `flair_b1_oem_no_paris_bootstrap_summary.json`
- `flair_supported_leaf_bootstrap_summary.json`
- `flair_b0_half_oem_no_paris_bootstrap_summary.json`
- `flair_batch_control_summary.json`
- `semantic_drone_confirmation_bootstrap_summary.json`
- `semantic_drone_confirmation_bootstrap_396_exploratory.json`
- `b2_openearthmap_bootstrap_summary.json`
- `e0_b0_oem_b2test_bootstrap_summary.json`

The archive inventory and evidence reconciliation record the SHA-256 values
for these inputs. A fresh zero-context reviewer must be rerun against this
expanded boundary after every manuscript edit affecting quantitative claims.
