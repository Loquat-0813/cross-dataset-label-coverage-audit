# P1 DeepGlobe Land-Cover Candidate Audit

Date: 2026-08-02  
Status: semantic candidate blocked from P1 acquisition by official licensing evidence

## Positive Evidence

The official [DeepGlobe CVPR 2018 challenge page](http://deepglobe.org/challenge.html)
defines its land-cover track as multi-class segmentation over `urban`,
`agriculture`, `rangeland`, `forest`, `water`, `barren`, and `unknown`.
This makes its `rangeland` label a strong semantic candidate for the P1
`herbaceous_vegetation` coverage gap. A narrow experimental mapping would use
only that source label and ignore the remaining DeepGlobe labels, avoiding
unreviewed urban/building-road and agriculture/cropland conflations.

## License Block

The official [DeepGlobe resource page](http://deepglobe.org/resources.html),
accessed 2026-08-02, states:

> By accepting the terms and conditions, the participant agrees that the
> dataset is currently only released for this challenge. We are planning to
> release it with an open license, however this is not the case for now.

The same page directs users to challenge-specific Codalab registration for
downloads. This conflicts with using an unaffiliated Kaggle mirror as a
reproducible, redistributable P1 training source. No subsequent official open
license was verified during this audit.

## Decision

Do not download or train on a Kaggle DeepGlobe mirror for the P1 paper unless
an official current license explicitly authorizes this non-challenge research
use and its terms are saved with the experiment artifacts.

DeepGlobe remains a semantically attractive candidate but is not an eligible
B1 source under the current evidence. The zero-download OpenEarthMap B2
cross-city control is promoted as the next experiment. FLAIR remains the
audited open-license candidate for a later B1 coverage-completion arm.
