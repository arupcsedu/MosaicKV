# Event-only actual-model attribution

The native reader's positive operator test did not translate into a request win;
its registered 20-repetition native request gate is negative. CUPTI-instrumented
model windows showed smaller kernels, yet ordinary decode wall time did not
improve. This consumed-input diagnostic separates instrumentation effects from
the actual stream timeline before another runtime redesign.

No runtime policy, constructor, values, thresholds, model or sample changes.
Compare original EviCastKV, native EviCastKV and the original PrefixKV graph
control. After one excluded warmup per policy, measure all 127 step intervals
using CUDA events, with the 64 reader subintervals at steps 1/64/126. No CUPTI or
torch profiler is enabled and no synchronization is inserted inside these
windows. Events include host launch gaps and are **not pure kernel time**.
Full 128-logit-vector bitwise parity against each own parent is mandatory.

This is attribution, not latency acceptance, benchmark-quality evidence or a
whole-request peak claim. Two one-A100/12-CPU workers, at most six A100s aggregate;
full CPU suite first; afterany terminal analysis. No automatic inference retry,
fresh cohort, calibration or paper-scale campaign is authorized by this plan.
