# V8 directional single-exchange canary seal

V8 is sealed as a negative development result. Its 16 COCO identities are
consumed and must not be used to tune, calibrate, or validate a successor.
No V8 calibration or paper campaign is authorized.

## Immutable evidence

- Source base SHA: `8566a826578d5e4e3b4b6bec700b3fb7e5389d68`
- Plan SHA-256: `3ebd57a5ad9775b38dc4bae6245a6865910353746fe0c34d87980c9117ed855c`
- Ledger SHA-256: `42504161f56a06e5c9f22dbc2cb9cf16ff2e4ed079c26901e843796f765bc006`
- Analysis SHA-256: `1330c8d1714b16b3fe53fa434d4d8c55e7d525c742c6fc65d0481fcf3cd56b19`
- GPU array: `21520246`
- Analysis job: `21520248`

## Decision

The exact 25.5 MiB candidate reduced initial storage by 36.25% and final/peak
storage by 30.287% relative to Prompt Top-K at 40 MiB. Exact caps, zero
padding, retention-one parity, trace completeness, provenance, and label-blind
execution passed.

The candidate failed both preregistered scientific gates:

- all 16 samples fell back to the 25.5 MiB Prompt Top-K anchor, exceeding the
  maximum 20% fail-closed rate;
- CIDEr was 102.643 versus 102.244 for Prompt Top-K at 40 MiB, a paired
  difference of +0.399 with 95% CI [-13.450, +13.843], whose lower bound is
  below the required zero margin;
- only 3 of 32 two-token candidate probes passed the maximum-logit gate, and
  none remained valid through the eight-token extension.

V8 shows that directional layer exchanges are too coarse for the frozen
logit-stability thresholds. V9 therefore uses fixed-shape, byte-neutral
one-block substitutions inside one layer/head route and fresh identities.
