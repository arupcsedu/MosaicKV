# V102 negative debug-gate seal

V102 is sealed as a negative engineering result. The one-sample-per-task A100
debug array failed on both tasks before producing a complete MosaicKV candidate
row. The 32-sample calibration and conditional freeze were not submitted, no
`frozen_candidate.yaml` was created, and held-out evaluation remains forbidden.

## What ran

The immutable source snapshot was
`4449e6d673dba5b0f65235f9967f7b6da0b00444`. Slurm array `18816597` ran one
TextVQA-train sample and one InfographicVQA sample, each on one A100-SXM4-80GB.
Both workers exited nonzero after the FullKV and Prompt Top-K controls completed.

| Task | FullKV active bytes | Prompt Top-K active bytes | Candidate status | Candidate trace time |
|---|---:|---:|---|---:|
| TextVQA train | 318,242,816 | 40,894,464 | failed | 284.5914 s |
| InfographicVQA | 322,961,408 | 41,418,752 | failed | 258.6678 s |

These are debug measurements, not paper-result rows. The candidate exception in
both traces was `selected cache exceeds its dense physical prompt capacity`.
Source-control-flow review shows that the bounded hierarchical search had chosen
its fail-safe FullKV fallback. The then-current runtime expanded the logical
capacity for that fallback and subsequently attempted to pack it into a much
smaller physical cache. Historical traces did not retain the fallback decision,
attempted bytes, individual probe decisions, or phase timings, so that diagnosis
is explicitly source-derived rather than falsely presented as recorded telemetry.

## Corrective behavior

The runtime now treats an over-budget FullKV fallback as a structured, fail-closed
outcome before final cache packing. Its trace records the fallback flag, attempted
and hard-budget bytes and positions, the complete bounded-probe decision, and
timing accrued through rejection. It explicitly records that the over-budget
selected fallback cache was neither finally packed nor decoded. Bounded probe
candidates are still materialized in isolated, disposable states and are recorded
as such. This improves auditability; it does
not convert V102 into a successful result or retroactively repair its artifacts.

TextVQA provenance is also corrected. The lmms-eval target template can resolve
to the field name `answer`, whereas TextVQA scoring and provenance require the
concrete `answers` annotation list. New rows serialize that list. Existing V102
TextVQA rows retain the historical defect and are ineligible for scientific use.

## Scientific boundary

The V102 debug answers, references, scores, and failure identities were inspected.
Consequently, all samples in both V102 calibration ledgers are conservatively
ineligible for any future calibration—not merely the two executed samples. A
future effort must use a new protocol version, genuinely fresh samples, and a new
exclusion/disjointness review. No V102 job may be resumed or submitted.

The machine-readable seal is
[`configs/ledgers/choice_c_v102_negative_debug_gate_seal.yaml`](../configs/ledgers/choice_c_v102_negative_debug_gate_seal.yaml).
