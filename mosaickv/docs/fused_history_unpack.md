# Same-precision mixed-history unpack launch fusion

This default-off consumed-input diagnostic follows measured common header and
complete-predicate tests. It changes neither the six adaptive mechanisms nor
the registered older-INT8/recent-exact precision split. It is a storage-variant
implementation optimization, not independently validated task-quality evidence.

The excluded warmup readout profile identified repeated source unpacking as
the largest sampled host phase. Fuse the optional older INT8 body and up to
four exact 16-token BF16 packets into one CUDA launch writing the **existing**
128-row source workspace. Keep separate FP32 multiply/add rounding for INT8;
reconstruct every BF16 bit, including special values, for exact packets. Native
projection, original positions, RoPE, all constructor verification passes and
counted numerical patches remain unchanged. No pointer-table GPU allocation is
introduced. Request-private compiled dispatch holds weak tensor references only.

Execution is CPU regressions -> actual operator kernels -> the same two consumed
pretrained inputs -> conditional direct three-arm randomized paired timing.
Both model arms receive identical complete predicates and early/late exact
headers. Require all 128 logit vectors bitwise against the unfused same-precision
reader, unchanged Full-parent token/KL/logit limits, identical persistent initial
and final bytes, complete traces, and independently recomputed ownership ledgers.
Preserve any peak failure rather than converting numerical success to a memory
claim. Timing uses five warmups and twenty repetitions, twelve preregistered
endpoints, adjusted intervals and an equal-or-faster upper bound <=1.00.

Fresh cohorts, benchmark calibration and paper-scale execution remain disabled.
This optimization is common infrastructure: compatible baseline/storage-only
rows receive the same exact header/predicate optimizations. None of the present
results establish a six-component, universal quality, or whole-process HBM claim.
