# Decode-latency optimization audit

## Implemented safe changes

The explicit Hugging Face loop now applies the following backend-neutral
optimizations without changing selected positions or K/V values:

- ordinary decode steps advance the live cache and no longer clone the full,
  growing cache; only a provisional step that may be repaired is isolated;
- `logits_to_keep=1` prevents materialization of unused prompt/decode logits;
- attention-mask and logical-position buffers grow geometrically on device;
- extending an all-valid decode mask reuses its initialized storage instead of
  launching a redundant one-element CUDA write per token;
- Qwen2.5-VL precomputes and reuses its decoded M-RoPE position-ID range, and
  preserves the immutable prefill `rope_deltas` instead of allocating both on
  every token;
- finite-logit flags are reduced once per sequence instead of synchronizing a
  second time at every token;
- append-only dynamic-cache storage is preallocated for the configured decode
  horizon and exposed through occupied-prefix views, avoiding per-token K/V
  concatenation while preserving the adapter-owned logical position;
- dense all-valid FullKV and exact-selection caches omit the model-side
  all-ones attention mask; and
- exact blocks are packed with one vectorized gather per layer instead of one
  gather per selected token;
- prototype construction gathers all unselected blocks once per layer/KV head,
  reuses those views for pinned residual slabs, and materializes the validated
  tier once instead of rebuilding it with and without residuals;
- low-memory forecast centroids stage one bounded layer of CUDA query samples
  at a time rather than synchronizing a host transfer for every KV head and
  prompt/draft source;
- the no-hook decode path skips per-token `ExitStack` construction and advances
  the documented mutable `DecodeState` in place; and
- an explicitly configured eager-prefill/SDPA-decode path retains eager prompt
  evidence while using fused SDPA for all-valid exact decoding.

Vectorized cache construction removed the dominant LLaVA bottleneck. The
subsequent controlled dirty-tree A100 job `17346904_2` completed five warmups
and 20 repetitions per bounded point. At output length 64, FullKV p50
inter-token latency was 42.897 ms and p50 decode was 2.701 s. Exact selection
at retention 0.1 used 22,544,384 active bytes but had p50 inter-token latency
43.075 ms; its 4.500 s compression cost dominated the 13.345 s end-to-end
latency. The prior projected-time failure is therefore fixed, but this short
LLaVA regime is weight/launch-bound and does not show a decode saving.

FullKV can use SDPA or FlashAttention-2 after its backend-specific parity gate.
Compression methods that require eager prompt attention remain labeled eager;
they are not silently compared with a fused FullKV row as though the backend
configuration were identical.

The real-checkpoint SDPA gate passed on the pinned Qwen2.5-VL-3B checkpoint in
exploratory Slurm job `17346718`: all 16 generated tokens agreed and the
maximum logit difference was `0.0` for ordinary generation, cache
reinjection, and retention-1 MosaicKV state reconstruction. The worktree was
dirty and is recorded as such, so this is implementation evidence rather than
a paper result.

The installed FlashAttention-2 2.8.3 SM80 inference wheel passed dense and
variable-length CUDA kernels at head dimensions 64, 80, and 128 in environment
job `17347802`. Real Qwen2.5-VL-3B job `17348171` then passed 16/16 ordinary,
reinjection, and retention-1 token parity with maximum logit difference `0.0`.
On the matched output-64 FullKV microbenchmark, job `17349594_0` measured p50
inter-token latency 53.144 ms and decode 3.351 s. FlashAttention-2 therefore
improved prefill/end-to-end time but did not beat the SDPA decode result in
this batch-one, one-token-query regime.

Preallocated cache growth was checked with matched five-warmup/20-repetition
Qwen2.5-VL-3B runs. It reduced p50 FullKV decode from 3.450 s to 3.377 s
(2.1%) and p50 MosaicKV-exact decode from 3.385 s to 3.338 s (1.4%). In the
new byte-matched pair, Prompt Top-K and MosaicKV exact both used exactly
1,769,472 active bytes; MosaicKV exact decoded in 3.338 s versus 3.396 s
(1.7% lower). MosaicKV's 0.778 s forecast/graph/selection/packing cost still
made its end-to-end latency higher, so the next target is compression planning
rather than token decoding.

Layer-staged query transfer job `17349904_0` preserved the exact 64-token
MosaicKV sequence. Relative to the matched preceding run, p50 centroid time
fell from 331.3 ms to 323.2 ms (2.4%) and p50 total compression from 778.4 ms
to 769.9 ms (1.1%). This is a safe incremental improvement, not a dominant
latency result; fixed clocks were not permitted and the independently measured
decode phase was slower on that allocation, so end-to-end values are not
attributed to the centroid change.

The paired 5-warmup/20-trial Qwen2.5-VL-3B jobs `17352166_0` and `17352272_0`
measured the new hot path at the same FullKV point. Eager p50 inter-token
latency/decode were 55.252 ms/3.482 s; eager-prefill/SDPA-decode measured
47.919 ms/3.019 s, reductions of 13.3% in both metrics. All 20 trials in each
job produced one identical deterministic token sequence. Fixed clocks were not
permitted, but both isolated A100s reported P0 at the same 210 MHz SM and 1593
MHz memory snapshot; these remain dirty-tree implementation evidence.

At the exact-selection retention-0.1 point, jobs `17353263_0` and `17353264_0`
used the same 589,824 active KV bytes. Switching decode to SDPA reduced p50
inter-token latency from 53.081 to 47.482 ms (10.5%), p50 decode from 3.346 to
2.995 s (10.5%), and p50 end-to-end latency from 4.762 to 4.400 s (7.6%). Each
job was internally deterministic; as expected for a changed numerical backend,
its greedy sequence is not assumed to equal eager at near-tie logits. The
retention-1 MosaicKV identity job `17353479_0` used the same 11,206,656 bytes
as hybrid FullKV and produced the exact same 64-token sequence in all 20 trials.
Comparisons must therefore use the same eager-prefill/SDPA-decode configuration
for every method.

## Next optimizations, in priority order

1. **Extend fused decode beyond the current safe gate.** The SDPA decode path
   is implemented for all-valid exact caches without repair. Padded per-head
   caches, prototype attention, and repair remain rejected until their own
   parity tests pass.
2. **Vectorized forecast and graph planning.** The layer-staged centroid path
   removes small D2H transfers without changing forecast math. Next profile
   graph construction and facility-location marginal updates, retaining sparse
   memory and deterministic selection tests before accepting a fast path.
3. **Serving-engine batching.** vLLM/SGLang FullKV parity now passes; use their
   continuous batching for throughput rows. This reduces per-request host
   overhead but is a different concurrency regime and cannot replace isolated
   batch-one latency measurements.
4. **CUDA graphs for stable paths.** Capture only fixed-shape FullKV or
   exact-only decode after correctness. Dynamic prototype promotion and repair
   invalidate the static-address assumptions and must remain outside this
   path.
5. **Repair work reduction.** For adapters that eventually gain phase-safe
   prototypes, compute entropy first and request prototype attention only on a
   possible repair step. This avoids eager attention materialization on the
   common no-repair path.
6. **Deferred event collection for production-only timing.** CUDA events can
   be queued per token and resolved at a sequence boundary to avoid a host
   synchronization per step. The existing scientific microbenchmark retains
   synchronized timing boundaries; any deferred mode must be separately
   labeled and validated against it.

## Constraints

Lower active KV bytes do not guarantee lower latency when the remaining
attention work is smaller than MLP, projection, Python, or synchronization
overheads. Results must therefore report both active bytes and measured p50/p95
decode latency. Native vLLM/SGLang cache mutation, post-RoPE prototype merging,
and residual repair remain fail-closed until their independent correctness
gates pass; simulated rows are prohibited. The one exception is the audited
HF eager Qwen2.5-VL/Transformers-4.55.2 path. Its v17 Video-MME diagnostic
reduced tier construction from 317.7 to 22.0 seconds through grouped gather and
single materialization; this is compression-overhead evidence, not a decode-
latency or paper claim.
