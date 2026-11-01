# Same-membership precision causal control

All 32 single-head causal interventions failed the unchanged MM-Vet guard.
Preserve that completed negative result; no fresh/larger cohort is authorized.
Do not raise restoration counts, weaken the guard, or tune against the answers.

Before changing selection again, isolate pruning from K/V quantization:

1. Reuse the pinned LLaVA-1.5 BF16/eager prefill and the EXACT same frozen
   EviCastKV mixed-prior source memberships at source-slot ratio 0.5.
2. Quantized control: original common head3_boundary8/source-only representation.
   All 128 logits must remain bitwise identical to its previous saved trajectory.
3. Higher-memory control: gather ORIGINAL BF16 K/V at those IDENTICAL positions.
   No merging, fitting, new evidence, donor evictions, biases or head sidecars.
4. FullKV clone must match the saved ordinary FullKV reference (maximum logit
   difference <= 0.001 and identical greedy tokens). No second prefill.
5. Measure 32 conditional-prefix KL/logit/token differences and 128 greedy
   predictions (score through first EOS), storage at all three boundaries and
   decode timing. Keep the existing KL<=0.02, logit<=1 and full token agreement.
6. Report the selected-original-versus-FullKV contrast as pruning error, and
   quantized-versus-selected-original as conditional precision error. They are
   nonlinear effects, not assumed additive. Preserve all failure rows.

The BF16 control deliberately uses MORE memory and does NOT satisfy the original
hard compressed cap. It is not a compression candidate, byte-matched comparison,
superiority result, or eligible paper performance row. All storage is measured,
not extrapolated from a ratio. FullKV is quality reference only. These same two
consumed identities have independent N=0. ROUGE-L against FullKV is diagnostic,
not official MM-Vet scoring or complete PrefixKV PPL evidence. Whole-request
memory peak and end-to-end latency remain uncertified.

If BF16-selected still fails, fixing low-bit storage alone cannot remove that
selection deficit. If it passes while quantized fails, target precision placement
rather than another evidence-allocation sweep. Neither outcome authorizes fresh
confirmation without a separately frozen and validated candidate.

Use one A100/12 CPUs/30 minutes per worker, two concurrent workers, clean source,
common environment, scratch-only caches, full CPU suite first and finite afterany
analysis. Graph/submodular, forecasting, prototype and repair mechanisms are
disabled in this explicitly labeled causal diagnostic.
