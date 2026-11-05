# Prepared sparse-patch dispatch, not another projection-kernel sweep

The optimized actual-model CPU profile (43456db, two consumed inputs) attributes
7,104 JIT `run` calls to `dual_projection_kernel.apply_patch`, totaling about
0.49 s cumulatively. Projection already reuses its compiled kernel; the repeated
sparse-patch launch did not. This is host instrumentation, not a measured
0.49-second removable wall cost or a new latency win.

The separately flagged change reuses exactly the same Triton-3.4.0 compiled
sparse-patch binary with ALL original arguments, current pointer addresses,
integer dimensions and strides. Pointer/ABI/grid changes cause a new plan.
Plans hold only pointer/metadata keys and compiled launchers, never tensor
owners. They are bounded and cleared at request end. Default dispatch remains
original. No approximation, omission of numerical patches, position change,
weights change, extra GPU tensor, or weaker check is allowed.

Before checkpoint loading, real CUDA A-B-A controls check signed zeros, original
patched words and untouched sentinels across 0/1/127 padding. CPU tests cover
actual arguments, stride/pointer changes, owner lifetime, scratch rejection and
request isolation. Model controls require original packet hashes and ALL 128
own-parent vocabulary vectors bitwise, no fallback, unchanged initial/final/
decode-owned-peak backing storage and actual repeated dispatch hits.

Finite full-CPU -> two one-A100 screens -> terminal CPU analysis. The preceding
lookahead screen must finish first. Five warmups and three randomized requests
compare EviCastKV original/native-direct and both PrefixKV readers, WITH and
WITHOUT this common optimization. Every arm receives event-sum accounting;
lookahead is disabled for this ablation. One GPU, 12 CPUs, 48 GiB, twenty-minute
workers, no more than six aggregate A100s.

These are two consumed precision/storage inputs, independent-quality N=0,
not byte-matched algorithm rows, paper evidence, or full six-component method
validation. No formal latency acceptance from three repetitions; adjusted
twenty-repeat ratio <=0.96 versus every control remains required. No new cohort,
calibration or larger sweep follows automatically. Whole-request GPU peak
remains uncertified. Preserve negative outcomes and the original default.
