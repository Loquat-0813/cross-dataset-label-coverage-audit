# P1 Herbaceous Prompt Audit Evidence Record

Date: 2026-08-02  
Status: completed preregistered diagnostic; not a prompt-selection result for a
headline zero-shot claim.

## Protocol

- Target: all 2,687 RGB-paired OpenEarthMap rasters.
- Backbone: frozen `openai/clip-vit-base-patch16`.
- Variants: six preregistered herbaceous prompt strings plus their normalized
  text-feature mean ensemble.
- All non-herbaceous prompts, imagery, tiling, taxonomy, and evaluator were
  held fixed.
- E0 has no trained parameters. E1 uses
  `outputs/e1_loveda_seed19/adapter_final.pt`, trained only on LoveDA.

## Full OpenEarthMap Results

| Prompt variant | E0 exact mIoU | E0 rangeland IoU | E0 predicted herbaceous pixels | E1 exact mIoU | E1 rangeland IoU | E1 predicted herbaceous pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grassland_or_rangeland | 0.078628 | 0.012933 | 20,600,118 | 0.376398 | 0.000000 | 30 |
| grassland | 0.079032 | 0.015968 | 29,122,607 | 0.376398 | 0.000000 | 0 |
| rangeland | 0.076769 | 0.000701 | 853,868 | 0.376398 | 0.000000 | 0 |
| pasture | 0.076649 | 0.000041 | 40,491 | 0.376398 | 0.000000 | 0 |
| meadow | 0.076644 | 0.000018 | 18,029 | 0.376398 | 0.000000 | 0 |
| herbaceous_vegetation | 0.104257 | 0.210247 | 954,526,771 | 0.376404 | 0.000026 | 28,504 |
| normalized_text_feature_mean | 0.077227 | 0.002936 | 3,812,383 | 0.376398 | 0.000000 | 0 |

## Interpretation

The initial frozen baseline prompt, `a satellite image of grassland or
rangeland`, is not evidence that CLIP lacks herbaceous semantics. The
predeclared `a satellite image of herbaceous vegetation` variant raises E0
rangeland IoU to 0.210247.

Under the same seven variants, the LoveDA-trained E1 adapter almost entirely
suppresses the herbaceous output channel. The strongest E1 prompt variant emits
only 28,504 herbaceous pixels across the complete evaluation, versus
954,526,771 in E0. Since only the herbaceous text feature changes between
variants, this is evidence of adaptation-induced suppression of a source-
uncovered leaf, not a frozen-CLIP lexical blind spot.

## Decision

1. Do not run further E1-gated or E2-fixedscale-gated training/evaluation;
   their prompt-independent smoke failures are explained by the stronger full
   E1 result above.
2. Do not use `herbaceous vegetation` as a target-selected headline prompt on
   OpenEarthMap. This audit is a diagnostic. Any subsequent main evaluation
   needs a prompt fixed without selecting on OpenEarthMap labels or a fresh
   geographically/dataset-disjoint confirmation target.
3. The next method must preserve a frozen base-model route when a calibrated
   prediction set contains a source-uncovered leaf, rather than merely freezing
   that leaf's individual logit while allowing trained leaves to overwhelm it.
