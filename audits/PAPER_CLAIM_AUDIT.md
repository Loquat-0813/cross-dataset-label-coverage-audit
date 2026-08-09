# Paper Claim Audit

Date: 2026-08-07

Overall verdict: WARN

The primary numerical claims are grounded and round correctly against the supplied evidence. The remaining warnings concern protocol provenance, not the headline result.

Verified core claims:

- OpenEarthMap B1 minus B0 rangeland IoU: `+0.3118`, 95% CI `[+0.2821, +0.3399]`, 74 cities.
- Supported-leaf mIoU contrast: `-0.1011`, 95% CI `[-0.1213, -0.0821]`.
- B1 minus B1-shuffle rangeland IoU: `+0.0385`, 95% CI `[+0.0333, +0.0439]`; all three seedwise differences are positive.
- Semantic Drone B1 minus B0: 95% CI `[+0.1999, +0.2535]` over 400 rasters and three checkpoint pairs.
- B2 direct-supervision rangeland IoUs: `0.3934`, `0.3890`, `0.3776`; mean `0.3867`; E1 minus E0 CI `[+0.3430, +0.4030]`.
- E3 root-routing rate: `0.9779315`, reported as approximately `97.8%`.

Warnings remaining before submission:

1. The manuscript still describes fixed schedules, derangement invariants, and chronology that are not all independently represented in the compact evidence input. Include the full per-seed schedule records or a deterministic reconstruction manifest, together with the chronology ledger, in the reviewer-access package.
2. The OpenEarthMap discovery chain is reported as 2,687 paired and 813 label-only masks, while the Paris-excluded replay population is described by its 74-city scope. The exact Paris exclusion count is intentionally not used in the final table.
3. The duplicate-crop control is now reported only as zero at displayed precision; no strict floating-point bound is claimed.

No numeric contradiction was found in the primary OEM result, shuffle control, Semantic Drone confirmation, B2 learnability control, or E3 negative control.
