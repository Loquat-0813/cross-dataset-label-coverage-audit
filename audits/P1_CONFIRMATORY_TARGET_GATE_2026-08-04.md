# P1 Confirmatory Target Admission Gate

Date: 2026-08-04  
Status: required before any P1 fusion design, tuning, or confirmatory target run

## Required Evidence

A confirmation target must be admitted only after an audit records:

1. An official, usable research license and the exact acquisition source.
2. Complete raw-label values, pixel support, imagery/mask pairing, and ignored
   labels for the acquired release.
3. An exact mapping to a P1 taxonomy leaf absent from LoveDA, rather than a
   loose vegetation synonym or a coarse parent.
4. Geographic, temporal, and source-overlap review relative to LoveDA and
   FLAIR; exclusions must be applied before any target inference.
5. A target identifier fingerprint and a declaration that no target metric has
   participated in fusion-rule, prompt, checkpoint, or threshold selection.

## Current Candidate Status

| Candidate | Status | Reason |
| --- | --- | --- |
| DeepGlobe land cover | Blocked | Semantically attractive `rangeland`, but the existing official-license audit does not establish non-challenge research use. |
| LandCoverAI | Ineligible for herbaceous confirmation | Its audited mapping has no exact herbaceous/rangeland leaf. |
| OpenEarthMap | Exploratory only | It has already been inspected and used for P1 analysis. |

No candidate is admitted by this document. It is a gate, not an acquisition
authorization. The next dated audit must identify one candidate and satisfy
all five requirements before data download or GPU work begins.
