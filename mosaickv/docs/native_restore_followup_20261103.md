# Native bitwise restoration follow-up

The negative native K/V and V-only request timings remain unchanged. The next
operator study completed on both consumed inputs with bitwise controls. Wide
GEMM passed its comparison to the original fused kernel in isolated layers,
but was slower than the existing native BMM there and required an additional
4.7 MiB output buffer. Do not promote it. Fixed 128-token tiling was slower and
required up to approximately 10 MiB of repairs per layer, exceeding the frozen
3 MiB request repair cap. Do not promote or expand it.

The representative instrumented BMM pipeline spent about 0.14 ms in numerical
repair dispatch. Intervals include host/stream gaps and are not pure kernel
times. Microbenchmarks do not reproduce the full-request ordering: all latency
promotion must still use the paired full-request gate, never the micro means.

The only new candidate is `kv_fused_restore`. Keep identical source unpacking,
BMM, RoPE, K/V bits, logical positions, selected prompt, history, and workspace.
Combine original and native numerical patches only after verifying unique,
in-range scalar indices disjoint from protected rows and each other. Use the
already audited bitwise scatter kernel to restore protected values and merged
patches in one launch, not separate protected/old/new dispatches. Keep the base
patch backing for provenance; duplicated nonempty backing is counted, not hidden.
Retain all existing per-request validations. The default reader is unchanged.

Same two consumed inputs only, independent quality N=0. Six algorithm mechanisms
remain disabled in this precision/storage variant. Baselines remain local
storage adaptations, not official code or byte-matched algorithm evidence.
No new benchmark labels enter construction. No fresh or larger quality campaign.
Decode-owned tensor peak is not whole-request GPU peak.

Finite afterany chain: complete clean-source CPU suite -> two operator workers
(layers 0/16/31, padding 0/64/127, five warmups and twenty repetitions) -> CPU
analysis. Require bitwise/history controls and >=10% lower uninstrumented
operator interval than BOTH original and unfused BMM on EVERY geometry and
both inputs, with zero additional output workspace. Only then run two audited
full-request feasibility workers, requiring unchanged 128-token parent logits,
actual fused restore activity, and initial/final/decode-owned peak smaller than
the minimum recorded baseline boundaries. The fixed 3 MiB repair cap is unchanged.

Only after those checks pass run randomized paired requests (5 warmups, 20
repetitions). Preserve the conservative sixteen-contrast adjustment and original
<=0.99 own-parent / <=1.05 Prefix adaptation upper-ratio gates. One policy only;
stop on failure. If passed, the existing all-five timing gate runs automatically
and still requires <=1.0 adjusted upper ratio against all adaptations. This is
engineering evidence, never independent answer-quality superiority.

One GPU, twelve CPUs and 48 GiB per worker; at most two workers concurrently,
six A100 aggregate maximum, reservation bi_fox_dgx. Operator jobs have ten-minute
limits; conditional complete-request workers have sixty-minute limits. All
CPU callbacks have ten-minute limits. No continuous queue/log polling or source
changes while workers enforce their immutable SHA. Every failure gets analysis.
