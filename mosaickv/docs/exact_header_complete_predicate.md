# Exact scale headers and complete predicate fusion

This separately versioned, common-transport study implements the November 8
follow-up on the SAME TWO consumed COCO inputs. No new identities, label-based
tuning, default promotion, independent quality validation or paper campaign.
All six research mechanisms and validated storage-only defaults remain intact;
these two engineering changes do not establish six-mechanism superiority.

## Correct the header inventory before promising savings

The frozen request already enables `minimum_only=True`: minima are bit-exact
BF16, scales FP32. Do not credit the old minimum saving again. Stored original
scale bits may be packed as either an exact word dictionary or high16 words plus
an exact low16 dictionary. Count palette, packed index tail, two-byte selector,
raw fallback, and the shared FP32 decoding workspace. Choose the actual smallest
complete packet with deterministic ties; no rounding, new quantizer, changed
source codes, weights or protected values. A raw scale remains raw when encoding
does not save bytes. Generic CPU tests precede any GPU work.

The current COCO archives contain code bytes, not real scales. The first bounded
model control therefore archives actual minimum/scale arrays read-only at the
existing code-conversion boundary. It retains CPU tensors only and must match
every frozen vocabulary word. Only after both real captures pass are CPU costs
computed over all 32 layers, charging the largest required shared FP32 replay
buffer. These CPU costs are not measured GPU or whole-model savings. Runtime
header integration requires a subsequent clean source and an explicit operator/
model/accounting gate. Do not run a larger quality cohort because packet bytes
alone are positive.

## Complete, rather than partial, predicate fusion

Pinned PyTorch v2.8.0 `aten/src/ATen/native/TensorCompare.cpp:isclose` computes
`abs(self-other)` and `atol + abs(rtol*other)`, preserving dtype rounding at
every tensor operation. The old transport requires BOTH asymmetric predicates:
legacy FP32 closeness and actual BF16 verification. The new single-pass kernel
forms their union with the output-finiteness flag; the verification-only mode
uses the actual BF16 predicate. Separate operations/rounding, FMA disabled.
Dynamic `torch.nonzero`, index order, original BF16 correction words, protected
row rejection and deferred commit semantics stay unchanged. No threshold moves.

GPU controls cover all BF16 bit patterns in several pairings, boundary/tail sizes,
threshold perturbations, finite/protected/close failures, and complete sparse
patch equivalence. They also test raw/full-word/low16 scale decoding with target
sentinels and zero kernel spill/scratch requirements. Operator measurements use
five warmups, twenty CUDA-event repetitions and synchronized boundaries.

## Frozen first model comparison

- Pinned LLaVA-1.5-7B BF16 eager; same two consumed COCO images/prompts and 603
  logical source positions; 128 fixed greedy predictions, scored answer ends
  at the first EOS as before. Original first-32 KL/logit/token gate unchanged;
  these lossless changes additionally require every 128 vocabulary vector bitwise
  identical to the immutable parent storage-only reference.
- Same common tiled softmax (2048 complete query rows), direct additive compact
  code codec, logical positions, exact protection and native history execution
  controls in both arms. No approximate history or prototypes in this first test.
- Arms: `control_common`, `complete_predicate`. Initial and final cache storage
  must match exactly; fusion changes workspace scheduling, not persistent KV.
- Each writes manifest, raw JSONL, logits, timings and a time-aligned ownership
  ledger. Count initial/final and whole-request requested PyTorch KV/shared
  workspace; do not rename that as process HBM or allocator-reserved memory.
- Instrumented model requests are single trials, not clean paired latency.
  The correct later timing gate is multiplicity-adjusted upper ratio <=1.00;
  <=1.05 remains a reported trade-off only. Do not call operator speed a request
  speedup.

Finite `afterany` chain: CPU suite -> two one-A100 operator workers -> operator
review -> two one-A100 model/capture workers -> CPU analysis and actual header
costs. Failures are preserved, block the affected successor and are not converted
to success by retrying without a correction. One A100/12 CPUs/48 GiB, <=one hour
per worker, <=two concurrent workers under the aggregate-six controller;
`bii-gpu`, `bi_fox_dgx`; CPU jobs `bii`. Cache/build/temp paths remain scratch.

The common APIs apply identically to all compatible baseline representations.
Raw-cache baselines with no affine scale arrays receive no fictitious header
saving. A later method comparison must enable common transport on both storage-
only and hybrid/baseline arms, with exact result-origin/variant labels. No
superiority conclusion follows from these two consumed-input diagnostics alone.
