# CPU-only recovery and parallel PrefixKV resource sharing

Array 22123858 completed 16/16 workers and 832/832 requests without sample
errors. Analysis 22123859 stopped because direct Python equality treats a
serialized JSON array as different from a tuple. Canonical config SHA, resolved
file SHA and producer SHA matched. This is a metadata representation defect,
not a numerical failure and not evidence about quality.

The correction compares strict canonical JSON identities. It still rejects
changed caps, modality values, nonfinite data, source identities and either
artifact/config hash drift. No model, precision, budget, policy, scorer or
tolerance is changed. Original failed analysis and stopped follow-up 22124325
are preserved. A new derived root imports all original hash-bound workers;
raw predictions and manifests keep their actual producer SHA `a2adfc3`.
The corrected analyzer has its own immutable SHA. No GPU rerun is needed.

A replacement finite r16 controller can bind only a verified CPU-recovery
root of the original cohort. Its numerical canary still uses the unchanged
producer source; r16 development uses the corrected CPU analyzer. A genuine
execution or numerical failure cannot trigger an algorithm retry.

For the newly authorized parallel PrefixKV work, reserve **three concurrent
A100 slots per workflow**, at most six combined; one GPU/12 CPUs per worker.
This is a scheduling amendment, not a model/budget/control change. Original
source snapshots and submissions remain unchanged. Both workflows use finite
`afterany` analysis callbacks. No continuous log polling, no untouched-data
claim for development inputs and no automatic paper-superiority claim.
