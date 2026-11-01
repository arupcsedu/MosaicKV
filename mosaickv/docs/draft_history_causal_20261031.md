# Online-draft generated-history drift control (consumed inputs)

Keep all prior negatives. A CPU screen found zero prospective passing MM-Vet
head-plus-visual-token combinations; no joint retry is authorized. Source-token
restoration improved one Description output, but failed MM-Vet. Independent N=0.

This separately versioned diagnostic isolates a DIFFERENT error source: even at
identical teacher-forced token prefixes, compressed hidden states can produce
different generated K/V, and that history drift can propagate. Do not change
the prompt selection, K/V quantization, donor policy or model weights.

Frozen procedure:

1. One pinned LLaVA-1.5 BF16/eager prefill. Forecast 32 greedy tokens online using
   the same complete prompt. Share its immutable prompt owners read-only, use
   private append-only history and one layer lease; do not deep-copy the whole
   prompt, mutate the original prefill, or do a second prefill. Require ordinary
   FullKV first-33 logit parity <=0.001 and identical tokens.
2. Capture only those 32 generated-token K/V states in pinned CPU BF16 memory,
   with original logical positions and draft INPUT token IDs. This costs about
   16 MiB plus token IDs on the pinned model. It is NOT free storage. Record its
   exact backing bytes and the complete forecast/copy cost. No benchmark labels,
   answers or saved evaluation oracle are constructor inputs.
3. During FINAL compressed greedy decoding, copy into the fresh model-projected
   CURRENT K/V tensors (no added staging owner), and replace GENERATED K/V only while the actual input prefix
   exactly matches the forecast. Never emit/force draft tokens. The first input
   mismatch permanently disables repair; unmodified model computation continues.
4. Use original post-RoPE keys, never rotate them again, never change positions
   or prompt K/V. Replacement happens only in this request's private cache update.
   All original immutable source hashes and decode positions remain unchanged.
5. Keep raw lossless BF16 append-only history for EVERY method. This diagnostic
   does not enable source-history reconstruction, prototypes or adapter repair
   capability claims. Default common runtime behavior remains unchanged.
6. Apply IDENTICAL forecast/history controls to EviCastKV, Prompt Top-K and the
   three local reimplementations LOOK-M, VL-Cache and PrefixKV. Each has inactive
   shadow and active repair. Shadow must reproduce its old 128 logits bitwise.
   Charge CPU reference even in the inactive causal control. Reject any current
   transfer destination aliasing immutable prompt, history, another component,
   or model-weight storage. Native projected K/V owners are already present in
   both executions; no separate GPU staging allocation is hidden or omitted.
7. FullKV with reference injection is a numerical CONTROL, not a FullKV method
   row: require <=0.001 max logit difference and identical tokens. Ordinary FullKV
   quality reference itself performs no repair or offload.
8. Fixed-prefix guard remains 32 tokens, max FP32 KL<=0.02, logit difference<=1.0
   and greedy agreement 1. Final measured run remains 128 greedy predictions,
   scoring through first EOS. Report repaired steps/blocks/bytes, transfers, stop
   step, initial/final/decode-owned peak GPU storage, HOST residual bytes, decode
   time, forecast time and construction time separately. No hidden latency win.

This is a generated-history repair ablation, not the original source-residual /
prototype algorithm. Query forecasting for selection, graph/submodular, value-aware
block utility and prototype mechanisms are replaced/disabled in this variant.
It is not an evaluation oracle: all reference states are computed ONLINE and
their work/storage is charged, and the final answer is independently decoded.

Two consumed inputs only, two one-A100/12-CPU/30-minute workers, full CPU suite
before admission and finite afterany analysis. No fresh or large campaign.
Both EviCastKV repaired inputs must have actual repairs, unchanged-guard passes
and non-regressed diagnostic ROUGE-L before considering a new quality study.

Different methods still have different encoded sizes inside their original own
source caps. Do NOT call these byte-matched paper evidence, PPL reproduction,
official MM-Vet scoring, or superiority. Strongest historical LOOK-M is not
silently replaced by this common-codec diagnostic. CPU memory may INCREASE total
storage despite smaller active GPU storage; report both. Whole-request peak
remains uncertified. No paper claim can follow from two consumed examples.

Transport-only retry: the first execution preserved valid partial rows but
stopped when its separate 0.5 MiB staging owner exceeded a baseline's own cap.
Do not raise that cap. New direct-copy outputs must reproduce every completed
partial 32/128-token trajectory bitwise; remeasure memory and timing because
the transfer implementation changed. Preserve and hash the failed attempt.

A greedy draft does not generally match annotation-based teacher-forcing targets.
Do not assume PPL improves, and NEVER substitute label-conditioned FullKV history
as online repair evidence. That would be an evaluation oracle. PPL and broad
PrefixKV reproduction claims remain separately gated and are not resolved here.
