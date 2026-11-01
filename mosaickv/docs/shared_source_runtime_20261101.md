# Full-prompt shared-source precision: same-input GPU feasibility

The protected CPU study passed its frozen 20% reconstruction criterion for
3-bit per-channel (axis 2) source encoding on BOTH consumed inputs. CPU prefix
sizes were within the old caps. This is numerical feasibility, not answer
quality. Preserve all prior negative studies; no cohort or paper promotion.

This is an explicitly different EviCastKV representation variant, NOT the
graph-based mosaickv_exact method, official Palu/MLA/KVQuant, or a validated
three-tier compressor. Store one quantized normalized projection input shared
by K/V instead of independently storing pruned K/V. Reconstruct every ORIGINAL
prompt position using the frozen native projection weights and exact original
phase; no phase averaging, weight update, token renumbering or offload.

Frozen same-two-input experiment:

1. Same pinned LLaVA-1.5 BF16/eager, media, prompts, processor, seed and one
   prefill. Full-MHA only, not a claim of GQA/model-family portability.
2. Three arms: old source-only shadow; full-prompt 3-bit per-channel source;
   same source with globally exact rows excluded from range fitting. In the
   last arm exclude a source row ONLY when all KV heads have exact K/V overlays
   there. This conditional-extrema ablation changes no storage or decoder.
3. Protect EXACTLY the existing per-head first-four/last-eight selected rows
   plus selected text positions. Actual original post-RoPE GPU K/V rows are
   serialized losslessly, and restored bitwise after reconstruction. Old KV
   scalar-patch locations are not carried into a different source encoding;
   this explicit variant replaces that quantization-error mitigation.
4. All original positions remain represented. This retained-position fraction
   of one is NOT a retention_ratio=1 experiment: the byte-compressed source is
   explicitly lossy. The public retention-1 control bypasses quantization and
   uses untouched raw K/V, with identical 128 greedy tokens and <=0.001 logits.
5. Fused GPU projection reads encoded source into registers and writes directly
   to the existing strided layer K/V lease. No decoded full-source or extra
   full K/V global-memory intermediate. Count codes, min/scales, exact rows,
   row maps, shared phase and identity metadata. Reject over-cap formats;
   never raise the old cap or execute a FullKV fallback as the candidate.
6. Construction control compares each GPU materialization to CUDA native
   projection of CPU-decoded SAME codes, rtol=0.01/atol=2^-6 for ordinary
   unprotected BF16 GEMM reduction variation. Every protected scalar must be
   BITWISE exact. Test strided prompt leases and untouched history sentinels.
   This construction tolerance is not the FullKV tolerance or model guard.
7. Verify original prefill K/V unchanged through construction. Then drop all
   original cache, attention/query captures and full normalized-source copies
   before decode; retain only immutable encoded formats and frozen weights.
   Record the observed CUDA allocation decrease. Inactive diagnostic formats
   still exist, so hardware peaks are labeled diagnostic, not whole-request
   production evidence. No CPU residual is part of the cache/runtime.
8. Model guard remains 32 FullKV conditioning prefixes, max FP32 KL<=0.02,
   max logit difference<=1 and greedy agreement=1. Independently decode all
   supported arms for 128 predictions; score through EOS using reference
   ROUGE-L. This is NOT official MM-Vet judge scoring or PPL.
9. Repeat source A after source B, require identical fixed-prefix logits and
   immutable encoded bytes. No cross-request history or cached answer replay.
10. Report initial/final/decode-owned peak bytes, encoder time, decode CUDA and
    wall time, torch peak allocated/reserved, and inactive diagnostic bytes.
    No general latency claim from zero warmups/one fixed-order repetition.
    Lower initial bytes cannot conceal a higher peak from full-length leases.

FullKV remains an unmodified quality control. No query forecast, graph,
value-aware utility, submodular selection, prototype or residual repair is
enabled under a full-system claim. The old selector is used only to define
this ablation's protected positions. The representation principle is related
to [Palu](https://arxiv.org/abs/2407.21118) and
[MLA](https://arxiv.org/abs/2405.04434), not claimed as generically novel.
Any later comparison must offer common storage improvements to applicable
baselines and preserve their actual selected/merged values; this diagnostic
does not establish superiority over LOOK-M, PrefixKV or VL-Cache.

Two one-A100/12-CPU workers <=30 minutes after the complete CPU suite, finite
afterany analysis. Independent quality N=0. Even a positive model guard does
not authorize a fresh cohort, freeze, large campaign or paper claim. All three
storage boundaries and independent benchmark quality remain separate gates.
