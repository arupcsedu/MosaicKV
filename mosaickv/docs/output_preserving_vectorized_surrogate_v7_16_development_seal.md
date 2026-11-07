# V7 vectorized-surrogate canary seal

V7 is sealed as a negative development result. Its 16 COCO identities are
consumed and must not be used to tune, calibrate, or validate a successor.
No V7 calibration or paper campaign is authorized.

## Immutable evidence

- Source base SHA: `8566a826578d5e4e3b4b6bec700b3fb7e5389d68`
- Plan SHA-256: `d7c295b604f11e842ecedee8e7619ea9e56b93580a6b9401ff1cd0647156d40c`
- Ledger SHA-256: `c03e5ce37e523e98671df566fd1a1dc18cd34d0f7624ed628ad3747b34eaac7f`
- Analysis SHA-256: `869262bdf63943a5652404e38a36018722b9825f7c8c15578c65a9e3168e0c34`
- GPU workers: `21519225`, `21519325`, `21519326`, `21519327`
- Analysis job: `21519227`

## Decision

The exact 25.5 MiB candidate reduced initial storage by 36.25% and final/peak
storage by 30.287% relative to the 40 MiB Prompt Top-K arm. Exact caps, zero
padding, retention-one parity, trace completeness, provenance, and label-blind
execution passed.

The candidate failed the registered scientific gate:

- fail-closed Prompt Top-K anchor use was 16/16 (100%), above the 20% limit;
- its answers therefore exactly matched Prompt Top-K at 25.5 MiB;
- paired CIDEr versus Prompt Top-K at 40 MiB was +19.647, but the registered
  95% confidence interval was [-5.435, 55.908], so the zero-margin lower-bound
  gate failed;
- four plans entered the eight-token extension and none remained within the
  frozen KL/logit limits.

V7 demonstrates that the shared one-scan planner fixed the V6 construction
cost, but it provides no evidence that aggressive multi-transfer plans improve
quality. V8 is therefore separately versioned and must use fresh identities.
