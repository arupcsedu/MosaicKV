# One bounded intermediate-width native projection feasibility test

The measured Source prompt path still reconstructs K/V through five native
128-row GEMMs per component and layer. The new hypothesis combines full tiles
into 192 rows, requiring four GEMMs at both consumed source lengths (600/604).
This reduces decoder, GEMM, output-copy and replay dispatches, without
factorization or matrix-product reassociation. It is one registered geometry,
not a new row/warp/algorithm sweep. 256 rows would add 2 MiB workspace and exceed
the prior MM-Vet observed-peak advantage; 192 rows adds exactly 1 MiB.

Copy the original 128-row native algorithm descriptor for full tiles, the
original native tail algorithm for a cold tail and the already frozen joint
tail descriptor for the incoming row. Require the installed AlgoCheck to
accept each copied descriptor with zero algorithm workspace and original FP32
accumulation flags. Select no new heuristic or timing-best algorithm.

Before touching the runtime, test decoded source/incoming words, full K/V
projected words including original RoPE, unused strided workspace sentinels,
and CUDA-graph words. No extra numerical repairs are permitted to rescue a
different output. Five warmups and twenty randomized paired operator trials
measure actual synchronized CUDA intervals. Report the full 3 MiB new
source/output workspace versus 2 MiB old workspace. Synthetic fixtures are
operator diagnostics, never model-quality measurements or paper rows.

Only a positive exact operator result could justify all-layer real-checkpoint
verification and a new complete-request screen on the same consumed inputs.
It cannot itself establish latency superiority, certify whole-request peak,
authorize fresh quality data or advance a larger campaign.

## Separately registered lossless-correction successor

The zero-new-correction operator failed at N=600: K/V differed by 164/401
words; incoming rows were exact. Preserve that output as negative, never
reinterpret it as a pass. The explicit `--lossless-corrections` successor
constructs and counts actual original-word patches after RoPE using the
existing SparseBitPatch format and unchanged 32-KiB per-layer cap. It must
restore every original word and untouched sentinel in eager and graph replay.
Incoming rows still must be bitwise without any correction: they vary during
decoding and cannot be repaired by a prefill table. This changes the stored
representation and its measured bytes, not the numerical tolerance. The
full-request runtime must fit the new prompt corrections, charge them in
initial/final/peak storage and preserve all 128 logit vectors before any
latency screen. Neither the failed zero-correction policy nor a synthetic
fixture establishes real-model superiority.
