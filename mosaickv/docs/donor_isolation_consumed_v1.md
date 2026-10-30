# Frozen donor-removal isolation (consumed diagnostic only)

The complete-delimiter experiment at `1afbdf0` changed two things: it removed
saved donors and restored missing delimiter pieces. None of its four task/role
comparisons passed the unchanged 32-prefix guard. It does not identify which
part causes harm. This follow-up does not change reservations or allocation.

## Frozen intervention

Use exactly Description `000000532947` and MM-Vet `v1_197`, and both existing
Top-K and EviCastKV anchors. Bind the prior plan, worker captures and
transactions by SHA-256. Copy `removed_positions` verbatim; no reranking,
individual-route search, annotation access, or donor-set changes are allowed.

Run two modes for each role: all-zero head-aware masks and donor-only masks.
Both use the same attention capture flags, shape, original K/V, logical
positions, bounded incremental-history transport, and saved FullKV prefix.
Donor-only sets the saved donor logits to negative infinity in every associated
query head. It adds no restored positions and never masks mandatory or generated
tokens. Actual attention captures must confirm exactly zero donor probability.
The zero-mask replay must match prior anchor logits bitwise, and physical
initial/final/post-compression transaction-peak KV storage must stay identical.
FullKV generation/reinjection parity is repeated before the intervention.

This is an **attention-availability causal ablation**, not physical eviction or
a new compressed operating point. Dense per-head physical eviction would need
ragged packing/padding and would confound this control. Retain donor tensors;
report zero physical bytes saved and report additional additive-mask backing
separately. The CPU synthetic test checks masked attention against actual
gathered attention, but does not certify a real-model ragged eviction backend.

## Analysis and limitations

Compare FullKV KL, maximum logit deviation, greedy agreement and FP64 KL
shadow telemetry for the anchor, donor-only, and saved joint restore/donor
run over every one of the same 32 prefixes. Apply the original per-prefix
guard (absolute tolerance `1e-7`, relative `1e-6`), token agreement and absolute
KL `0.02` / logit difference `1.0` limits without exceptions for BF16 ties.
Preserve negative results. Means cannot override a prefix failure.

`donor-only - anchor` isolates donor attention access in a fixed physical
layout. `joint - donor-only` is a conditional restoration comparison, not an
additive attribution: autoregressive effects interact, and the joint run uses
physical repacking rather than masking. Source hashing, attention capture and
extra masks contaminate timing; these are not latency results or whole-request
HBM-peak evidence. No answer-quality or superiority conclusion follows.

## Execution

The script `evicastkv_donor_isolation.py` prepares a write-once plan from a
clean source, then submits two one-A100 / 12-CPU / 16-GiB / 20-minute workers
and one finite `afterany` CPU analysis. Register under the six-A100 aggregate
ceiling; use `bi_fox_dgx`, all caches under scratch, and the common environment.
Do not poll logs continuously. Analysis writes raw logits, prefix metrics,
storage/attention records, failure records and a bounded stop decision. Fresh
evaluation, selector changes, calibration and larger campaigns remain blocked.

CPU checks before submission: 26 focused donor/delimiter tests passed in the
common environment, including real tiny LLaVA forward passes with 1 and 2 KV
heads, 32-prefix replay, mask/gather algebra, bitwise zero-mask parity, source
immutability, storage equality, hook cleanup, and analyzer rejection of nonzero
donor attention. Ruff, shell syntax, and targeted MyPy (`--follow-imports=skip`
for the new typed module) passed. These are implementation checks, not
pretrained-model quality evidence; GPU outcomes belong in the immutable run.
