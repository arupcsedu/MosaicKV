# Consumed-input, same-byte generation-delimiter causal control

This is not a production selector, benchmark comparison, prior sweep, fresh
cohort, or Prof. Fox confirmation. It reuses Description `000000532947` and
MM-Vet `v1_197`, and their saved FullKV 32-token prefixes from early-prefix plan
`ad153ffe6e625c991e9acde8f46c27b168cc8d05bc8e7cafd4324fde9cb4e489`.
No annotation targets or future logits enter construction.

Apply the same three arms to the frozen Top-K and mixed-prior EviCastKV caches:
anchor, exact no-change repacking, and complete `ASSISTANT:` restoration.
Identify the shortest exact tokenizer-decoded suffix, at most 16 tokens; do not
hardcode the observed five-token decomposition. Preserve the existing mandatory
last source token, every route's selected count, original K/V, logical positions,
precision, budget, history transport and all generation controls. Delimiter
reservation is an explicitly evaluated intervention, not a changed mandatory
policy silently imposed on only one method.

For each route, restore all missing delimiter positions. Remove the same number
of nonmandatory, nondelimiter positions ranked by increasing
`mean_raw_prefill_window_attention * L2(source_value)`, breaking ties by original
position. Use an independent read-only capture of the final 16 prompt queries,
without prior mixing, and the same donor rule for both methods. This score is a
prospective surrogate, not proof that a donor is harmless. Fail rather than
increase capacity if the transaction is infeasible.

Require null repacking to reproduce all logits bitwise, and every intervention
to preserve initial, final and post-compression transaction-peak physical bytes.
Hash original source K/V before and after construction; retain complete per-head
added/removed positions, donor scores, per-position costs and effective selection
hashes. Runtime selector catalogs still describe the original anchor because
the scoped intervention is injected afterward: use the separate transaction
record and captured injected-position catalog as the authoritative intervention
selection. Never present the anchor catalog as the modified cache.

Check all 32 actual forward predictions against the same saved FullKV prefix.
Use the existing per-prefix guard: absolute tolerance 1e-7, relative 1e-6, no new
greedy disagreement where the anchor agreed. Also retain KL <= 0.02 and maximum
logit difference <= 1.0. Record FP64 KL as shadow telemetry, not a replacement
for the registered FP32 guard. An exact FullKV BF16 tie never waives token parity.
Meaningful safe improvement requires changed positions, every-prefix guard and
absolute limits, plus at least one strict KL improvement or recovered prediction.
A mean improvement cannot conceal a failed prefix. Evaluate the complete horizon
even when an early prefix fails, since this is a bounded diagnostic.

Use two one-A100 workers, 12 CPUs, 16 GiB host RAM, 20-minute limits and a finite
afterany CPU analyzer. Remain below six aggregate A100s. Extra source hashing,
attention capture and repacking contaminate timing; no latency advantage is
claimed. Whole-request KV peak remains uncertified. Even a diagnostic pass does
not authorize automatic production changes, calibration or larger evaluation.
