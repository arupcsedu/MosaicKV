# Conventional tensor-core tiling without full-source workspace

Preserve the negative graph/arena timing and both staging designs. The fastest
verified full request remains 1.7–1.9% slower than optimized PrefixKV on the
two consumed inputs. No larger or fresh campaign is justified.

Historical full-width native GEMM did improve the projection pipeline, but its
full decoded-source and output buffers violated the intended storage margin.
The fixed 128-row wide-GEMM alternative was slower and needed large repairs.
Simply staging BF16 words or byte codes in front of the old projection also
failed: source staging raised register pressure and slowed actual execution.

This new, default-off operator dequantizes directly inside a native-style
64-row/128-column/32-reduction tensor-core tile (four warps, three stages).
It allocates NO full decoded source and NO separate projected-output buffer.
The same source codes, headers, coherent input columns, original weights and
phases are retained. K is rounded to BF16 before the unchanged RoPE routine.
This is not another old bit-identical row/warp sweep: changed GEMM accumulation
is explicitly allowed ONLY with a frozen sparse repair that restores EVERY
original decoder output word. It is not residual repair or a new quality
algorithm; the six main components remain disabled in this precision variant.

Verify source-word equivalence separately before any repair, so repairs cannot
hide an input-dequantization defect. Fit only to the original fused decoder,
never benchmark answers, logits, or FullKV labels. Freeze both component patch
indices/values once. Reuse them across destination history paddings 0/1/64/127
without refitting; all K/V words and history sentinels must match exactly.
Cap total numerical repairs at 32,768 bytes per layer (at most 1 MiB/32-layer
model); additional persistent patches are charged, never called free. Unknown
scratch or any compiler/GPU spills blocks advancement. No tolerance is relaxed.

Both original and candidate sequences run under private CUDA graphs. Candidate
timing includes its projection, RoPE and numerical patch application. Five
warmups, twenty randomized paired trials, CUDA events/synchronization and
10,000 bootstrap repetitions; adjusted upper operator ratio must be <=0.80
on BOTH frozen geometries. Source fitting/construction is measured separately
and explicitly excluded from this operator replay—not from a future full
request. No model-level latency or quality claim follows from an operator win.

CPU suite and CPU compiler resource checks first, then two one-A100/12-CPU
operator workers with afterany terminal analysis. Reservation bi_fox_dgx,
<=15min operators, <=six aggregate A100s, all caches outside home. No automatic
model, fresh cohort, calibration or paper submission. If positive, integrate
and offer the SAME replacement to all compatible baseline sources, retain
every original and optimized reader, and require full-request adjusted upper
ratio <=0.96. Include fitting, capture, buffers, repair storage and retirement
in that later request measurement. Whole-request GPU peak is still uncertified;
independent benchmark quality N=0.
