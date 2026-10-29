# V9: shared global prior with individual-window evidence

EviCastKV is the paper name of MosaicKV. This is a separately labeled exact-only
diagnostic variant, not official PrefixKV or a change to frozen campaigns.

## Mechanistic distinction to isolate

The pinned local PrefixKV source
`mosaickv/baselines/prefixkv.py:prefixkv_attention_scores` follows upstream
commit `597f1ab032704951550f93bcc8a23f1454b80aa4`: average query heads in the
incoming attention dtype, promote to FP32, then sum all prompt query rows.
Its initial selection uses a common layer importance order across heads.
V7 instead keeps a distinct full-prompt prior for each head. These are
different evidence reductions, even when both are called full-prompt attention.
Do not claim that this global scoring principle is novel to EviCastKV.

Test a shared global prior using exactly that dtype/reduction order, dividing
by the query count to obtain a mean. For weight alpha:

`E[h,q,i] = (1-alpha)*A[h,q,i] + alpha*mean_q'(mean_h A[h,q',i])`.

At alpha 0.5 the recent individual-head/query component remains explicit. Alpha
one is a full-prompt, shared-head scoring ablation. Model forward probabilities,
logits and K/V are not replaced by E; only the private selection evidence changes.
There is no label fitting, training, extra model probe or prototype/repair.

Why this is worth isolating: the history-capacity study left generation quality
unchanged, so another history-size sweep cannot repair that prompt-selection
failure. The native scorer also aggregates heads differently. This test isolates
that distinction rather than loosening fidelity or memory thresholds. It is a
hypothesis, not a promised quality improvement.

## Registered bounded scope

Start only after V7 is complete and negative, and V8's collector optimization
passes exact model-level equivalence and allocator reduction. Use only the same
two consumed inputs and immutable V5 configurations. Keep checkpoint,
processor, prompts, media, generation, precision, backend, physical capacities,
layer-profile shape, mandatory tokens, FIFO history and accounting unchanged.

At 128/256/512 history positions (223/289/421 MiB complete transaction caps), run:

- EviCastKV shared-prior alpha 0.5 and 1.0;
- full-prior Top-K alpha 1.0, explicitly labeled a scoring ablation;
- ordinary window-only Prompt Top-K alpha zero, re-executed contemporaneously.

This is twenty-four one-arm shards, one A100 and 12 CPUs each, maximum six A100s
aggregate, 30 minutes/16 GiB per shard. Ordinary Top-K controls must reproduce
their frozen tokens, answer, NLL/counts, scores and config hashes exactly. This
also verifies that the collector optimization benefits both compared methods
without changing numerical behavior. Compare actual initial/final/working-peak
bytes with those current controls, not just the configured cap.

Keep native PrefixKV as a separate operating-point control; do not call it
byte-matched. Require non-worse PPL and ROUGE-L separately against both native
and contemporary Top-K, plus smaller three-boundary storage than native.
The comparison does not promise a 30% reduction, and must report actual savings.
All rows and negative outcomes remain visible. Timing is descriptive, not a
statistical no-slower result. No generic whole-request KV certificate follows
from a lower CUDA allocator peak. N=1 per task is not statistical evidence;
fresh evaluation, calibration and paper-scale promotion remain blocked.
