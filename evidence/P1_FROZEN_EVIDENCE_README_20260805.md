# P1 Frozen Evidence Transcription

This directory is a deterministic transcription of the frozen result numbers
shown in the execution records and recorded in the dated protocol documents.
The primary execution archive is now present under
`archive/P1_PRIMARY_EXECUTION_ARCHIVE_20260806_v2/`; the reconciliation ledger
`primary_archive_reconciliation_20260806.json` checks every evidence row and
field against the archived artifacts. The R scripts in `figures/` read these
files; no plotted value is embedded in plotting code.

The updated archive has passed its internal SHA-256 inventory and all
deterministic summaries reproduce the stored numeric cores. The downloaded
outer `.sha256` sidecar is currently stale and must be matched before release.
The paper-to-evidence claim audit remains a separate final gate.

The 396-raster Semantic Drone analysis is intentionally flagged as exploratory.
The original record documents pipeline-smoke exposure for identifiers
`000`--`003`, but does not establish an independently dated pre-result
exclusion rule.
