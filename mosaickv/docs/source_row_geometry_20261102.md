# Optional source projection geometry: bounded engineering diagnostic

The completed consumed complete-request studies show a separate 0.5--0.7 s
source decode gap against the PrefixKV precision adaptation. Construction
optimizations do not by themselves erase this gap. Test hardware tiling, not
another selection/quality change. Register only (rows, heads) = (16,4), (32,4),
(64,4), (32,2). Default remains (16,4). Wider row tiles can reuse projection
weights across more tokens; this is a hypothesis, not a measured speedup.

No codes, headers, protected values, corrections, positions, source slots,
weights, history, precision or persistent bytes change. Keep original K-loop
reduction order, separate BF16 RoPE operations, and disabled FP fusion. Any
single K/V bit or 128-token vocabulary-logit difference rejects that geometry.
Record kernel registers, spills and shared memory; require no spill increase.

Run only after the preceding asynchronous-verification constructor passes and
its full timing study completes negatively. Use the same two consumed inputs,
one original prefill, and original immutable packets. Compare paired and
sequential K/V reconstruction on all 32 layers. Then verify original greedy
128-token vocabulary logits and encoded bytes including lossless replay
scratch. Unsupported geometry stays an explicit error record. Controls are
diagnostic: original packets remain resident here, so it is not peak-memory or
complete-request performance evidence. Source geometry affects only eligible
shared-source packets; unrelated baseline formats remain untouched.

Randomize geometry order for five warmups/twenty timed component-fill trials
per layer. Select ONE common geometry only if its adjusted upper fill-time
ratio is <=0.95 on both inputs, with zero bit differences and no spill increase.
Correction covers three geometries times two inputs. This is not a task-quality
test. If none passes, stop; do not relax conditions or launch another cohort.

Only a passing geometry triggers two complete-request workers with all FIVE
methods, five warmups and twenty trials, the unchanged paired 1.05x latency
gate, original packet hashes and every frozen 128-token logit check. Decode
buffer/storage geometry is unchanged; new shared/register footprint is reported
separately rather than presented as HBM cache storage. Reuse same-source CPU
proof, cap aggregate use at six A100s, one GPU/12 CPUs per worker, afterany
analysis and no continuous log polling. This finite chain cannot establish
independent quality preservation or paper superiority from two consumed inputs.
