# MosaicKV paper candidate v1

## Claim under test

This candidate has one model, one nearest baseline, three tasks, and two
acceptance gates:

- model: `llava-hf/llava-1.5-7b-hf` at revision
  `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` in FP16 with HF eager
  attention;
- nearest baseline: `prompt_attention_topk`;
- tasks: MMStar, TextVQA, and DocVQA, scored by their official lmms-eval
  accuracy, exact-match, and ANLS metrics respectively;
- memory: MosaicKV must use strictly more than 30% fewer measured active K/V
  tensor-storage bytes than Prompt Attention Top-K on every paired sample;
- quality: MosaicKV's official score must be no lower than Prompt Attention
  Top-K separately on every task. Incompatible task metrics are never averaged.

The candidate is source-ready but GPU-unverified. No result or performance
claim follows from this document.

The authorized preflight uses two newly materialized, previously unobserved
samples per task. Three isolated one-A100 workers run only FullKV, Prompt
Attention Top-K, and this candidate. The dependent analyzer evaluates only the
two gates above and cannot submit a paper-scale campaign.

## Candidate design

The runnable template is
`configs/paper_candidate_llava_v1.yaml`. For each sample, Prompt Attention
Top-K runs first. `PaperCandidateContract` converts its measured active bytes
to a block-aligned MosaicKV cap no larger than 69.9%. Quantization preserves a
strict reduction above 30% while keeping the largest feasible cache.

The cache is constructed as follows:

1. Hybrid future-query forecasting combines the last 16 prompt queries with
   four deterministic draft queries.
2. Prompt anchors remain exact. Spatially stratified graph evidence reserves
   image regions needed by individual question and OCR tokens.
3. The sparse graph and value-aware utility rerank only uncertain heads.
4. Unselected image evidence is represented by at most two RoPE-safe
   prototypes per routed head. Text is never prototyped.
5. Lossless source residuals are stored in pinned CPU memory.
6. Decision-aware repair can exchange restored source evidence for a
   lower-risk active item; the hard active-byte cap does not increase.

This is `mosaickv_full`, not an exact-only or simulated method. Benchmark
labels and references are unavailable to forecasting, graph construction,
selection, prototypes, and repair. They are read only afterward by the
official scorer.

## Required execution order

Run FullKV and Prompt Attention Top-K on an immutable common sample ledger,
resolve each MosaicKV byte cap from the paired baseline row, and then run the
candidate with otherwise identical inputs and generation settings. Evaluate
the final JSONL rows with `evaluate_paper_candidate_gate`. The candidate is
accepted only when all three per-task reports pass both gates.
