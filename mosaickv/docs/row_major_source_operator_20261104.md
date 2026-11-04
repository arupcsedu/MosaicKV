# Lossless row-major code transport: operator feasibility only

The prompt-graph controls passed complete parent-logit identity, but their
fixed buffer erased the MM-Vet decode-owned peak advantage against the original
PrefixKV control. Preserve that negative. Do not submit its paired timing.
The separate consumed-input attribution completed with all six parent-logit
checks. Graph setup took approximately 0.088 s for EviCastKV and 0.056 s for
PrefixKV; allocator snapshots totaled under 0.006 s. These instrumented single
requests, including cold compilation, cannot establish a latency win.

The next bounded feasibility test changes only **integer code layout**. Source
codes remain the original three-bit quantizer's values 0–7, with identical
headers, weights and floating-point operations. Reorder column-major packed
codes into a charged row-major nibble scratch, half a byte per existing code.
This is not four-bit requantization or a change to persistent packet contents.
The same original tensor-core projection reads those codes with coalesced row
access and no cross-byte three-bit extraction. Defaults remain unchanged.

Before model integration, run the full CPU suite, then synthetic source
geometries 600/604 tokens, 32 heads, head dimension 128, row tile 16 and four
heads per projection group. Require exact integer transport, **bitwise** K/V
and untouched history sentinels, plus non-increasing reported kernel spills.
Measure conversion once plus both K/V projections, five warmups and twenty
randomized paired repeats. The registered adjusted upper operator ratio is
<=0.80 (10,000 bootstrap draws; tail 0.05/8). Count the full scratch backing.
No model checkpoint is loaded; random frozen operator weights are synthetic
fixtures and never paper/task-quality observations. Independent quality N=0.

A passing operator gate permits consideration of model integration only. It
does not authorize it automatically, and does not establish the user's >=4%
full-request target. Such integration also needs safe scratch reuse during
history consolidation and capture of lossless code replay without capturing
its Python ordering state. All numerical, ownership and memory checks remain.
No fresh cohort, calibration or paper campaign is authorized by this study.
