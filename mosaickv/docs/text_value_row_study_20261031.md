# Text-query whole-value precision: bounded development ablation

This is an explicitly lossy, online-profiled precision variant, not the frozen
exact-selector paper algorithm. No prototype, repair, training or label access
is introduced. The preceding scalar refit reduced storage but did not establish
quality preservation; whole-value rows were mixed on two consumed inputs.

Hypothesis: visual prefill queries in a short prompt's last 16 positions can
dominate max-query scoring. Limit funded value-row demand to text positions in
the processor's modality map. Keep individual queries separate; no pooled mean.
An empty text-query window fails closed. Keep K, selected source positions,
mandatory tokens, original logical positions, source capacity, generated-history
storage and the funded exception byte allowance unchanged. Every selector has
the same representation option. Retention 1 bypasses all lossy transformations.

Run only the same 16 consumed COCO images, not independent or held-out evidence.
Offer the option to Prompt Top-K and the local PrefixKV, VL-Cache, LOOK-M
append-only adaptations as well as EviCastKV. Preserve LOOK-M's actual merged
K/V. Reuse the preceding 160 hash-bound union3/scalar-refit observations and
FullKV quality reference without relabeling producer SHAs. Add 80 complete
requests, one warmup and one measured trial, 128 fixed timed tokens with answer
scoring only to first EOS. This is a rate/distortion diagnostic, not a matched
encoded-byte comparison. No original negative point is removed.

Require clean CPU suite, same-packet bitwise CPU/CUDA reconstruction and all
BF16 scatter-pattern controls, per-worker ordinary-generation/reinjection
parity, unchanged union3 logit-hash controls, repeated-request isolation, finite
timings and actual initial/final/decode-owned peak accounting. A four-image
EviCastKV canary must complete before other arms. At most six aggregate A100s,
one GPU/12 CPUs/30 minutes each. Finite afterany jobs promote only this bounded
study and recompute corpus CIDEr in every one of 10,000 paired replicates, with
Bonferroni adjustment over 14 candidate comparisons. This cannot authorize a
fresh or paper campaign. Cross-run latency is descriptive, not a paired repeated
trial proof. Whole-request allocator memory is not KV-only request peak.
