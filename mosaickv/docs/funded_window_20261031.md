# Fixed last-16 prompt-text completion, consumed diagnostic

Preserve the delimiter-only result, including the opposite effects across
Description/MM-Vet. It is not an all-task success. Test one separately frozen
scope: authoritative text positions in the final 16 ORIGINAL prompt positions,
including the verified ASSISTANT: suffix. Sixteen is the existing prompt-query
window, not a fitted threshold. It may contain only part of a long question;
do not describe this as complete-question preservation. No broader scope
sweep, fresh cohort or paper campaign is authorized by this protocol.

The rationale is attention competition: restoring only role markers may
increase their influence without restoring nearby instruction/question
evidence. Test the joint text window, without donors or new quantization.
The numerical/storage package is BYTE-IDENTICAL to 9988fd0: same source
projection, exact residuals, in-place integer cycles, compact mask, hard
own-anchor byte transaction, padding accounting and fallback rules. Only
the driver supplies a different fixed protected-position tuple. Every
previously retained decoded K/V value remains bitwise identical. Added text
uses original BF16 source. Dummy sorting keys are never logical positions.

This is still a padded-layout development ablation, NOT the frozen exact
paper method or a zero-padding gate pass. Every materialized padded row,
mask, source, phase, residual and map counts at the declared decode-owned
boundary. Source-only fallback is explicit; there is no FullKV fallback.
Whole-request KV-only peak is uncertified. Actual final/peak storage must
be measured; initial byte funding alone is insufficient.

Use the same two CONSUMED inputs and pinned BF16/eager LLaVA-1.5 model,
processor, prompts/media, original .5 anchor source cap, last-token mandatory
policy, block1, and boundary-eight-bit/middle-three-bit format. Offer the
same anchor/source/completion variants to EviCastKV, LOOK-M, Top-K, VL-Cache
and PrefixKV local adaptations. No original baseline or nominal-ratio result
is overwritten. These are rate/distortion points with own-anchor caps,
NOT a direct shared-realized-byte experiment or official reproduction.

Reuse all twenty completed anchor/source rows and their logits from the
completed delimiter retry, preserving original source/config/manifest fields
and verified hashes. Rebuild packets but do not repeat those trajectories.
Execute only TEN new completion inference rows. Source-only must remain
bitwise identical; the EviCastKV anchor must match the immutable old reference.
Before proposals, require FullKV clone/repack and all-valid-mask controls
(token agreement1, max logit difference<=.001), CUDA/CPU mask equality and
all-65536-BF16-pattern integer-cycle checks. Registered compressed guard
remains 32 tokens, KL<=.02, logit difference<=1, greedy agreement1. Do NOT
change thresholds or hide any negative task/metric.

Preserve computed tensors before postconditions. Record 128-token greedy
outputs, first-EOS FullKV-reference ROUGE-L, actual added positions, padding,
source immutability, memory and decode/construction timings. Fixed-order,
single-trial timings are exploratory, not randomized latency confirmation or
E2E gains. Restore hooks/cache state between requests and on exceptions.

Complete CPU suite from a clean SHA; two one-A100 workers with 12 CPUs and
30-minute limits; max six aggregate A100s; dependent CPU analysis. Independent
quality N=0. No additional scope tuning or large campaign follows a negative
result. Only meaningful quality preservation/improvement with numerical and
measured-memory correctness justifies a separately preregistered development
cohort; a diagnostic success is not statistical superiority.
