# P1 Coverage-Gap Evidence Record

Date: 2026-08-04  
Status: completed-result record; not a new preregistration or a model-selection protocol

## Claim Boundary

P1 studies an auditable source-label coverage gap and its consequences under a
fixed source-compute budget. It does not claim universal fine-tuning
suppression, general FLAIR harm, or a finished corrective method.

The supported paper-level wording is:

> Independent herbaceous supervision completes a documented source coverage
> gap, while fixed-capacity source mixing reallocates performance across
> already-covered leaves.

## Completed Evidence

| Analysis | Target population | Result |
| --- | --- | --- |
| Frozen E0 versus LoveDA B0 | B2 test manifest: 16 cities, 595 paired rasters | E0 rangeland `0.010389`; three-seed B0 mean `0.000000`; B0-E0 CI `[-0.014148, -0.007013]`. |
| FLAIR B1 versus B0 | Paris-excluded OpenEarthMap: 74 cities | B1-B0 rangeland `+0.311761`, CI `[+0.282103, +0.339855]`; mIoU change `-0.042163`. |
| FLAIR B1 versus B0 | B2 test manifest: 16 cities, 595 paired rasters | B1 rangeland `0.305904`; B1-B0 CI `[+0.270132, +0.341509]`. |
| B0-half versus B0 | Paris-excluded OpenEarthMap: 74 cities | mIoU `-0.006941`, CI `[-0.010008, -0.003804]`; rangeland `0.000000`. |
| B1 versus B0-half | Paris-excluded OpenEarthMap: 74 cities | mIoU `-0.035222`, CI `[-0.053812, -0.018212]`; rangeland `+0.311761`, CI `[+0.282103, +0.339855]`. |

The B0-half control matches B1's 25,220 LoveDA crop exposures but has a
one-example effective microbatch rather than B1's two-example mixture. It
separates the exposure component from most of the observed additional mixture
effect, but it is not a complete causal identification of every optimization
difference.

## Class-Specific Allocation Pattern

The B1-minus-B0-half intervals are fully negative for
`transport_surface` (`-0.146782`), `woody_vegetation` (`-0.149864`),
`cropland` (`-0.194076`), and `bare_surface` (`-0.058337`). In contrast,
the `surface_water` decline is captured by B0-half-minus-B0 (`-0.032898`),
with a B1-minus-B0-half interval spanning zero. This is evidence of a
class-specific allocation pattern under the observed source mixture, not a
claim that FLAIR intrinsically harms those categories.

## Secondary Mechanistic Observation

The E0/B0 diagnostic passes the predeclared post-training gate for restricted
wording: frozen E0 has a small nonzero rangeland signal and B0 is zero on the
same target cities. Therefore the paper may report a dataset-specific
observation that full LoveDA calibration suppresses a pre-trained uncovered
concept signal. It must not generalize this finding to all models, prompts,
datasets, or fine-tuning procedures.

The six-point frozen CLIP prompt-distance diagnostic has Pearson correlation
`0.317735` between distance to herbaceous and B1-minus-B0 IoU change. It is a
descriptive, underpowered visual diagnostic only and cannot establish a
semantic-gradient mechanism.

## Discontinued and Next Gates

E3 remains a negative control: its source-conformal root routing is not
useful under external transfer and receives no further GPU tuning.

No fusion rule may be selected or promoted using OpenEarthMap results. The
next P1 compute gate is a separate target dataset that satisfies all of the
following before method design: official usable license, frozen raw-label
audit, exact audited leaf absent from LoveDA, no prior use in P1 target result
inspection, and source-only selection of any fusion parameters. The current
DeepGlobe candidate is not eligible because its official license status is
recorded as insufficient; LandCoverAI lacks an exact herbaceous leaf.
