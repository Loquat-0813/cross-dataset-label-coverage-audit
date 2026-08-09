# P1 FLAIR #1 Candidate Data Audit

Date: 2026-08-02  
Status: candidate reviewed; acquisition and local mask audit still required

## Decision Scope

FLAIR #1 is the candidate source for the B1 coverage-completion experiment in
the frozen P1 protocol. Its role is not broad data aggregation. The initial
causal intervention is deliberately narrow: add independently labelled
herbaceous supervision while retaining LoveDA as the source for the six
already-covered taxonomy leaves.

## Officially Verified Facts

Source: [IGN FLAIR #1 official page](https://ignf.github.io/FLAIR/FLAIR1/flair_1.html),
accessed 2026-08-02.

- Publisher: French National Institute of Geographical and Forest Information
  (IGN).
- License: Etalab Open Licence 2.0, as declared on the official FLAIR portal.
- FLAIR-one: 77,412 aerial-image patches, each `512 x 512` pixels at `0.2 m`
  spatial resolution; official page describes 19 semantic classes, with a
  13-class challenge remapping for baseline use.
- The official original class table declares `herbaceous vegetation` as value
  `10`, with `17.84%` of train and `22.17%` of test pixels.
- Coverage: approximately 812 square kilometres sampled across metropolitan
  France, with spatial and temporal domains.
- Official downloads: train aerial imagery `50.7 GiB`, train labels `485 MiB`,
  aerial metadata `16.1 MiB`, and a `215 MiB` toy subset. The official page
  also links [Hugging Face dataset `IGNF/FLAIR`](https://huggingface.co/datasets/IGNF/FLAIR).

## Candidate Mapping and Constraints

The only proposed initial B1 addition is:

```text
FLAIR original raw value 10, herbaceous vegetation
    -> herbaceous_vegetation (exact, provisional until local audit)
```

Every other FLAIR class is ignored in the initial completion arm, including
`agricultural land`, `plowed land`, tree subclasses, and surface subclasses.
This prevents an unreviewed many-to-one mapping from changing the six existing
LoveDA supervision channels. It isolates the causal question: does independent
herbaceous supervision repair the documented coverage gap?

No claim is made that French herbaceous vegetation is identical to every
OpenEarthMap rangeland condition. Geographic and acquisition differences are
part of the B1 transfer test, not something removed by relabelling.

## Required Gate Before Any B1 Training

1. Acquire the official toy archive first and inspect its directory layout,
   image/mask pairing, dimensions, raw value range, and whether stored PNG
   values retain the official one-based value `10`.
2. Save a complete audit JSON for the full training labels before changing a
   mapping from provisional to scoreable.
3. Record the full-download checksums, exact extraction layout, disk use, and
   train/validation/domain split rule.
4. Create a separate versioned FLAIR mapping file. Do not edit the frozen
   `dataset_label_mappings_v0.yaml`.
5. Train B1 with LoveDA and the audited one-class FLAIR addition under a
   predeclared sampling ratio; compare to a matched LoveDA-only arm using the
   same adapter budget and seeds.

## Acquisition Cost

The full B1 training input requires at least `50.7 GiB + 485 MiB` compressed
downloads before extraction. The cloud disk must have enough room for both the
archives and the extracted data. This document does not authorize a download.
