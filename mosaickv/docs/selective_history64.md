# Consumed-input selective history component, not an exact-cache claim

Prerequisite: `early-exact-headers-20261108-v1` passed bitwise 128-logit,
ownership, initial/final and request-wide KV/workspace peak checks on both
consumed inputs. Its peak comparison is single-trial, not statistically validated.
Both arms of this next component test receive the identical complete predicate,
early exact headers, and late lossless scale replay. Raw/affine-compatible
baseline transports must receive the same common optimizations in later tests.

The previous 16-recent-token INT8 policies failed their complete-horizon
numerical checks. Do not retry them or loosen their thresholds. This separately
registered policy keeps the **64 latest sealed tokens**, plus the mutable tail,
on the original exact path; only older source chunks use INT8 group32. It keeps
original positions, phases, model weights, prompt cache, and mandatory tokens.
All extra FP32 min/scale headers, joining scratch, old/new storage overlap, and
reconstruction buffers remain charged. CPU residual storage is zero. No task
answers or references are available to the policy.

CPU regressions prove no modification through 64 sealed tokens, four exact
recent chunks survive each transaction, all logical positions remain present,
and backing storage matches counters. The real CUDA group32 oracle and sentinel
gate precede model execution. Model tests use only the same two consumed inputs
and the original 128 predictions. Preserve the original 32-token gate **and**
require all 128 greedy tokens, maximum KL <=0.02 and maximum logit difference
<=1.0 across all 128 positions. Label KL as same-prefix only if token agreement
holds. A passing numerical diagnostic is not independent benchmark quality.

This history-only change cannot reduce initial storage relative to the
common-optimized storage-only comparator. Report that equality explicitly;
never call a final-memory reduction an initial-memory win. Require smaller final
storage and a non-increased audited peak for the component gate, then report
latency honestly. The complete initial/final/peak superiority flag remains false
unless an additional independently validated initial-storage intervention
exists. No new cohort, calibration or paper campaign is automatically launched.
