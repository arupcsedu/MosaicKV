# V9 within-route substitution canary seal

V9 is sealed as a negative development result. Its 16 COCO identities are
consumed and must not be used to tune, calibrate, or validate a successor.
The preregistered 128-sample calibration and every paper-scale successor are
blocked.

## Immutable evidence

- Source base SHA: `8566a826578d5e4e3b4b6bec700b3fb7e5389d68`
- Plan SHA-256: `e6cc334bd81be243f52082fa05d5d907bf1c9b633cad567b548bd520b7fc5967`
- Ledger SHA-256: `e5e3422869d60506d59ff12ba8c9635d9c0f63259dab0097ff5bad4e828e6571`
- GPU array: `21533794`
- Analysis job: `21533796`

Two earlier immutable attempts are retained as execution-negative artifacts:
the first rejected the registered 16-plan count at a legacy 15-plan boundary;
the second completed inference but its post-run validator expected the legacy
8-plan/2-probe schedule. Both defects were corrected with CPU regressions
before the successful execution above. No benchmark answers or sample failure
identities were inspected during either correction.

## Passed gates

- all four arms completed with complete rows and provenance;
- retention-one parity and the label-blind probe trace passed;
- the 25.5 MiB initial cap and zero-padding exact-cache construction passed;
- initial storage was 36.25% below Prompt Top-K at 40 MiB;
- final and peak storage were 30.287% below Prompt Top-K at 40 MiB.

## Failed gates

- 15 of 16 requests (93.75%) used the fail-closed 25.5 MiB Prompt Top-K
  anchor, above the preregistered 20% maximum;
- V9 CIDEr was 131.489 versus 137.416 for Prompt Top-K at 40 MiB;
- the paired difference was -5.926 CIDEr with 95% CI [-36.052, +16.372],
  below the registered zero-margin lower-bound requirement.

Across the four final candidate records per request, only 21/64 satisfied the
maximum-logit threshold, while 37/64 satisfied KL and 49/64 preserved greedy
tokens. Eleven short-horizon plans passed, thirteen plans reached the extended
record, and only two extended records passed; those passing records occurred
in one request. Thus fixed-shape one-block substitutions avoid the coarse
layer-exchange failure but do not provide enough logit-stable choices.

Mean label-blind planning/probe phases were 0.381 s for analytic candidate
generation, 8.497 s for the batched four-candidate two-token probe, and 4.379 s
for the two-candidate eight-token extension when invoked. These timings are
development diagnostics, not latency claims.
