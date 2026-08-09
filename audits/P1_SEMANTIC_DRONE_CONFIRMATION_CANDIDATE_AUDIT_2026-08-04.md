# P1 Semantic Drone Confirmation Candidate Audit

Date: 2026-08-04  
Status: documentation-stage candidate; eligible to request official acquisition, not
yet admitted for target inference, model selection, or GPU work

## Scope

This audit evaluates the Semantic Drone Dataset only as a fresh confirmation
target for P1. No P1 model has been run on it, and no target accuracy, IoU,
prompt, checkpoint, threshold, or fusion result has been inspected.

## Official Source and License

The official TU Graz IVC project page is:

<https://ivc.tugraz.at/research-project/semantic-drone-dataset/>

The page describes 400 publicly available, pixel-annotated training images at
6000 x 4000 pixels, and gives the official TU Graz Cloud acquisition entry
point:

<https://cloud.tugraz.at/index.php/s/csxYfaKmie6LyqA>

Its published terms allow academic research, teaching, scientific
publications, and personal experimentation on a non-commercial basis. They
require attribution, prohibit redistribution of raw or modified data, and
allow trained models as abstract derivative works. This is sufficient for the
planned non-commercial P1 experiment, subject to retaining the terms with the
acquired release and not redistributing data in artifacts.

## Exact Leaf Candidate

The official class table names `grass` as a distinct pixel-level class,
separate from `tree` and `other vegetation`. The proposed narrow mapping is:

| Semantic Drone raw class | P1 leaf | Use |
| --- | --- | --- |
| `grass` | `herbaceous_vegetation` | Scored leaf |
| all other raw classes | no P1 mapping | Ignored for the narrow confirmation score |

This is a direct lexical and taxonomic match to the P1 uncovered leaf. It is
not the loose `vegetation` mapping that excluded AeroScapes and VDD. LoveDA's
audited label mapping contains no exact `herbaceous_vegetation` leaf.

## Why It Is a Better Confirmation Candidate

Semantic Drone is a distinct high-resolution UAV domain, whereas the current
P1 exploratory target is OpenEarthMap and the completion source is FLAIR. Its
officially documented taxonomy avoids the two rejected alternatives:

| Dataset | Decision | Reason |
| --- | --- | --- |
| Semantic Drone | Proceed to acquisition audit | Official source and non-commercial research terms; separate `grass` leaf. |
| UAVid | Reserve only | Official terms are clear, but its target leaf is `low vegetation`, not an exact grass/herbaceous label. |
| AeroScapes | Exclude | Its `vegetation` class is broad. |
| VDD | Exclude | Its `vegetation` class merges tree and low vegetation. |
| DeepGlobe | Blocked | Existing audit lacks evidence of an official non-challenge research license. |

## Admission Conditions After Acquisition

The candidate does **not** yet pass the confirmation-target admission gate.
Before any model inference, a new raw-source audit must record all of the
following for the official release:

1. Archive checksum, file inventory, imagery/mask pairing rule, and identifier
   fingerprint.
2. Every raw mask value, the `grass` pixel count, ignored labels, and a visual
   sample of the RGB/mask alignment.
3. The exact raw-value-to-`grass` mapping, with all non-grass values explicitly
   ignored for the narrow score.
4. Geographic, temporal, and imagery-source overlap review against the LoveDA
   and FLAIR sources. The public project page identifies the collection type
   but does not provide enough acquisition-location metadata to close this
   check at documentation stage; inspect release metadata and record the
   outcome rather than inferring it from the hosting institution.
5. A frozen target manifest before its first P1 inference. No Semantic Drone
   target metric may select a fusion weight, prompt, threshold, or checkpoint.

If the raw audit finds an incompatible label encoding, insufficient grass
support, or unresolved overlap, reject the dataset without a target run.

## Proposed Confirmation Design, If Admitted

Use fixed B0 and B1 checkpoints already trained under the audited P1 source
protocol. Evaluate only the predeclared `grass` to
`herbaceous_vegetation` score on a tiled, extent-preserving inference path.
This initial run tests replication of the coverage-completion observation; it
does not evaluate a fusion method. Any later complementary-expert method must
fix its parameters using source-only evidence before this target is opened.
