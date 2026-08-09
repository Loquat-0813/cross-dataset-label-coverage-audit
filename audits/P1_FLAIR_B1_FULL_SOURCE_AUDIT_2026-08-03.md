# P1 FLAIR B1 Full-Source Audit

Date: 2026-08-03  
Status: source data integrity passed; geographic-overlap review pending

## Acquired Release

- Source: official `IGNF/FLAIR-1-2`, `data/train-val`
- License: Etalab Open Licence 2.0
- Archives: 40 frozen protocol domains, retained as ZIP files
- Source manifest: `audits/flair_b1_full_source_manifest_2026-08-03.json`
- Raw audit: `audits/flair_b1_full_raw_audit_2026-08-03.json`
- Cloud extraction profile: `aerial_labels_only_v1`; `sentinel/` is excluded
  because B1 never reads that modality.

The source manifest records SHA-256 for every archive. Its archive layout
contains the expected `aerial/` and `labels/` directories, with 61,712
`IMG_*.tif` / `MSK_*.tif` pairs in total.

## Raster and Label Results

| Item | Full source | Train domains | Validation domains |
| --- | ---: | ---: | ---: |
| Paired rasters | 61,712 | 47,712 | 14,000 |
| Pair identifier SHA-256 | `3147befbc58ed12c50d2ac22b833ffb026c2b9ed772c5b41222e05f03ad754ae` | `877fba6eadb9fe2db411dfd6e72d641f7cec767fee2e1443311ed5d31c8b286d` | `c9944ff66e2aaa13c853ed9037f2ae9581ac949169f5658ae6dca02c2a17226a` |
| Raw ID 10 pixels | 2,886,249,939 | 2,213,531,387 | 672,718,552 |
| Raw ID 10 fraction | 17.8412% | 17.6978% | 18.3301% |

Every imagery raster is `512 x 512 x 5`, `uint8`; every mask is `512 x 512`,
`uint8`. The RGB CLIP input uses exactly FLAIR channels 0, 1, and 2. Raw label
ID 10 is the sole B1 supervision channel and maps to
`herbaceous_vegetation`; every other FLAIR raw label remains ignored.

## Current Gate Decision

The raw-data gate passes. B1 remains **not authorized for GPU optimization**
until both of the following are recorded:

1. The immutable source archives and this audit are reproduced on the cloud
   storage that will execute the experiment.
2. A provenance/city-level geographic-overlap review is completed against the
   exact paired OpenEarthMap release. The FLAIR GeoTIFFs do not expose usable
   geographic tags, so this review cannot be honestly reduced to a local
   coordinate-intersection assertion.

Once those checks are complete, set the derived audit status to
`source_audit_ready`; the fixed-budget training entrypoint rejects any other
status.
