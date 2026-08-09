# P1 E3 Coverage-Guard Smoke Record

Date: 2026-08-02  
Status: implementation smoke; not a headline or full-seed result

## Purpose

This record preserves the first end-to-end execution of the preregistered
`e3_coverage_guard` implementation. Its purpose was to test artifact format,
source conformal diagnostics, actual hierarchy-node evaluation, and the
predeclared stop gate before any 20-epoch or full-target E3 run.

## Fixed Inputs

- Frozen backbone: local `clip-vit-base-patch16`.
- Prompt list: `taxonomy_prompts_v0.yaml`, including `grassland or rangeland`.
- Source coverage: LoveDA has six exact leaves and lacks
  `herbaceous_vegetation`.
- Source split: deterministic 85/15 identifier split specified in protocol
  amendment v0.2.
- Smoke optimization: seed 19, one epoch, first 256 discovered LoveDA training
  pairs, four LoveDA validation pairs.
- External smoke: first four RGB-paired OpenEarthMap rasters only.

## Observed Source Diagnostics

```text
train loss                       1.640199
LoveDA validation mIoU           0.054864
E0 probability threshold         0.113525
sampled calibration scores       77,824
source validation set coverage   0.938326
source validation mean set size  5.915677 / 7
```

The nominal 90 percent source coverage condition was met, but the prediction
set was already nearly the full seven-leaf taxonomy on the small source
validation smoke.

## Observed OpenEarthMap Smoke

Exact eligible pixels: 3,796,997. Covered targets: 3,096,089. Source-uncovered
`herbaceous_vegetation` targets: 700,908.

| Metric | Matched flat adapter | Coverage guard |
| --- | ---: | ---: |
| Covered-leaf mIoU | 0.145546 | 0.002073 |
| Uncovered ancestor correctness | 0.000000 | 0.930201 |
| Violation leaf rate | 1.000000 | 0.069799 |
| Ancestor-specific utility | 0.000000 | 0.001999 |
| Route rate | 0.000000 | 0.977932 |

The guard emitted the root node for 3,706,035 pixels (97.60 percent of exact
eligible target pixels), with only 7,168 parent-level and 83,794 leaf-level
outputs. Its high ancestor correctness is consequently mechanical: the root
is an ancestor of every valid target. The near-zero ancestor-specific utility
and collapsed covered-leaf mIoU expose the failure.

## Decision

The implementation is functioning, but the v0.2 rule is not eligible for a
20-epoch or full-target expansion. The routing decision is driven by frozen E0
conformal sets, so additional adapter optimization cannot correct the observed
97.8 percent route rate.

No OpenEarthMap alpha sweep is authorized. The next diagnostic is source-only:
evaluate the predeclared conformal alpha grid in
`configs/source_conformal_grid_v1.yaml` on LoveDA's deterministic calibration
and validation partitions. It determines whether any candidate can produce a
nondegenerate source-domain routing distribution before a new hierarchical
method or protocol amendment is considered.
