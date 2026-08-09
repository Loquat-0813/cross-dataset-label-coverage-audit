# RSKT-Seg Competitor Audit

Status: completed source audit, 2026-08-01

## Sources inspected

- Paper: *Exploring Efficient Open-Vocabulary Segmentation in the Remote
  Sensing*, AAAI 2026 Oral, arXiv:2509.12040.
- Author repository: `LiBingyu01/RSKT-Seg`, commit
  `7b84091598e1edc3236dfbf45cc27e7e3436ffcb` (2026-06-30).
- The repository does not declare a license. It is retained under `vendor/`
  for reading and reproduction only; no source file may be copied into a
  publishable P1 implementation without explicit permission.

## What RSKT-Seg already covers

| Component | Evidence in author code | Consequence for P1 |
|---|---|---|
| Multi-directional cost map | `RSKT_Decoder.correlation_rotate` computes 0/90/180/270 degree CLIP costs | Not a P1 novelty candidate. |
| Remote-sensing CLIP transfer | `RSKT_Seg_Predictor.py` loads frozen RemoteCLIP ViT-B/32 | Not a P1 novelty candidate. |
| Remote-sensing DINO transfer | `RSKT_Seg.py` loads frozen RSIB/DINO features and passes cost and decoder guidance | Not a P1 novelty candidate. |
| Cost-map fusion | `RSKT_Decoder.simple_separate_corr` fuses CLIP and DINO cost maps | Not a P1 novelty candidate. |
| Lightweight guided upsampling | `RSKT_Upsample.py` applies two guided transpose-convolution stages | Not a P1 novelty candidate. |
| Cross-dataset benchmark | Dataset JSON files cover DLRSD, iSAID, LoveDA, Potsdam, Vaihingen, UAVid, UDD5 and VDD | P1 must report a protocol distinction, not duplicate this benchmark. |

## Reproduction facts

- The released iSAID ViT-B configuration uses 15 foreground labels,
  384 x 384 crops, batch size 8, 30,000 iterations, and the full set of
  rotation, RemoteCLIP, and RSIB/DINO modules.
- `KEY_reproduce.sh` reports four RTX 4090 GPUs. A one-RTX-5090 reproduction
  must reduce batch size and preserve the effective batch size with gradient
  accumulation; it cannot be labelled an exact compute reproduction.
- The author README pins Python 3.8, PyTorch 2.3.0 and CUDA 11.8. That runtime
  is inappropriate for an RTX 5090. Any local reproduction must port the
  dependencies to the current PyTorch/CUDA stack and record deviations.
- The Windows checkout reports a case collision between `Potsdam.json` and
  `potsdam.json`; the authoritative evaluation should occur on the Linux cloud
  host.

## P1 research boundary after audit

P1 will not claim novelty in rotation aggregation, multi-encoder transfer,
cost-map fusion, or guided upsampling. Its candidate contribution is instead
taxonomy-aware open-vocabulary transfer: isolating semantic competence from
label granularity mismatch across land-cover datasets.

The next protocol must distinguish:

1. equivalent/synonymous label transfer;
2. parent-child or coarse-fine label transfer; and
3. genuinely unseen semantic concepts.

