# Consumed nonuniform-value representation ablation

Freeze this common format before inspecting its task outputs. Preserve the
negative mixed-codec and mass/value-prototype diagnostics. No prototype,
repair, draft, new allocation, selection or history policy is used here.
This is not the original exact-selection paper method or official KVQuant.

## Why and what changes

Uniform two-bit values waste levels spanning a row's extrema. Test four
request-local BF16 centroids instead of a uniformly spaced grid. Four BF16
centroids occupy exactly the same bytes as the existing two FP32 affine
parameters. Codes remain two bits, with the same physical code/header budget.
Initialize from the actual old decoder's BF16 levels; make exactly eight
deterministic Lloyd updates, retaining an update only when its actual BF16
per-row MSE improves. Empty groups keep their previous centroid. No labels,
future reference answers, gradients, fine-tuning or model-weight changes.
Better scalar reconstruction is **not** a guarantee of generated quality.

Attribution: Hooper et al., *KVQuant: Towards 10 Million Context Length LLM
Inference with KV Cache Quantization*, NeurIPS 2024,
<https://arxiv.org/abs/2401.18079>, establishes nonuniform KV quantization.
This per-request, four-centroid implementation is our engineering choice,
not a KVQuant reproduction or claim of inventing nonuniform quantization.

Retain the common mixed format's three-bit keys, first/last-two-layer
eight-bit K/V, exact head-local text/first-four/last-eight protection and
funded scalar-exception count. Refit exceptions using the actual new decoder
error. Original logical positions and mandatory policy do not change. Keys
are bitwise the same as the mixed-codec control. No RoPE phase is averaged.
Direct CUDA decoding uses a centroid lookup into the same strided layer
lease; CPU/CUDA oracle checks decode the **same** encoded packet, not two
independent quantizers. The union3 control path must remain bitwise unchanged.

## Bounded evaluation

Offer this exact same format to EviCastKV, Prompt Top-K, LOOK-M, VL-Cache and
PrefixKV local adaptations. LOOK-M retains its actual merged K/V. EviCastKV
and Top-K retain their registered balanced capacities; other methods retain
native local allocation. Same raw source-slot cap, model/revision, BF16,
eager, prompts/media, 128-token greedy trajectory, first-EOS scoring and
lossless append-only consolidated source history. Do not represent differing
encoded byte counts as a shared-byte-matched comparison or native baseline
paper reproduction. Generalized COCO PrefixKV remains fidelity-unresolved.

Use only the same 16 consumed COCO images. Reuse all 768 prior rows, including
negative EviCastKV variants and the strongest smaller LOOK-M points. Add
80 requests: five methods, each four four-input shards. Complete CPU tests,
then one canary, then the remaining 19 one-A100 workers, maximum six A100s
aggregate, 12 CPUs/worker, 30-minute limits. Finite dependent CPU callbacks
promote/analyze after completion; failures block promotion, not disappear.

Recompute official corpus CIDEr inside 10,000 paired bootstrap draws. One
new EviCastKV candidate has 52 contrasts across 53 compressed points, with
Bonferroni simultaneous 95% intervals. FullKV is quality-only. Report all
initial/final/decode-owned peak bytes and actual construction/E2E latency.
Whole-request KV-only peak remains uncertified. Historical five-warmup rows
are not directly latency-comparable to one-warmup representation studies.
N=16 development only; independent quality N=0; no paper/fresh promotion.

Expand only after meaningful quality/storage improvement against the
strongest eligible baseline, not merely execution success or a weak-baseline
win. Otherwise preserve the negative result and stop this format.
