# Auditing Label-Coverage Gaps in Cross-Dataset Land-Cover Segmentation

Lightweight, GitHub-ready evidence and code release accompanying an anonymous
submission to the *International Journal of Applied Earth Observation and
Geoinformation* (JAG).

## What is included

- `paper/`: current LaTeX source, bibliography, compiled manuscript PDF, and
  claim-audit record.
- `figures/`: manuscript figure PDFs, table sources, and the deterministic
  Figure 5 rendering script and selection metadata.
- `scripts/`, `p1data/`, `p1eval/`, `p1model/`, `p1train/`: audit, training,
  evaluation, and paired-bootstrap code.
- `configs/`, `ontology/`, `protocols/`: frozen configurations, ontology
  mappings, amendments, and execution ledgers.
- `audits/`, `data_audits/`, `evidence/`, `outputs/`: derived metrics,
  city-level confusion summaries, raw-label audit summaries, and claim evidence.
- `tests/`: deterministic unit tests for the shuffle, budget, and bootstrap logic.

## What is intentionally excluded

No source imagery, raster masks, dataset archives, CLIP base weights, adapter
checkpoints, or other restricted/licensed artifacts are included. Obtain LoveDA,
FLAIR, OpenEarthMap, Semantic Drone, and the CLIP model from their original
providers under their respective terms. The included run configurations identify
the expected inputs and model identifier.

LandCoverAI is a documentary feasibility screen, not a second efficacy
replication. The B1-shuffle analysis is a dated post-primary robustness
amendment; the governing protocol is retained in `protocols/`.

## Reproduce the transparent checks

Install the minimal audit dependencies:

```text
pip install -r requirements-review.txt
```

From the repository root, run:

```text
python -m unittest -v tests.test_flair_shuffle tests.test_fixed_budget tests.test_semantic_drone_bootstrap tests.test_supported_leaf_bootstrap
```

The shipped derived artifacts permit inspection of the reported calculations
without redistributing restricted pixels. Re-running pixel-level evaluation or
training also requires the original datasets, CLIP model, and CUDA environment.

## Manuscript

Compile from `paper/` with an Elsevier-compatible LaTeX distribution:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The figure path and bibliography path are self-contained in this release.

## Integrity

After downloading, verify all release files using:

```text
sha256sum -c SHA256SUMS.txt
```

`SHA256SUMS.txt` is generated at release time. Schedule digests and checkpoint
hashes for the shuffle audit are retained in the independent audit JSON.

## Before a public release

Choose and add an explicit code/data license, replace anonymous authorship and
declaration placeholders in the manuscript, and update the data-and-code URL in
the final submission metadata. Do not upload restricted source pixels or model
weights to a public repository.
