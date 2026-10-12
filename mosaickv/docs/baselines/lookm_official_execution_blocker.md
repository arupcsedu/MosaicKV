# LOOK-M official-code execution blocker

Status: **official execution disabled; local results use `lookm_reimpl` only**.

The audited upstream repository is pinned at
`SUSTechBruce/LOOK-M@ecf0f51a9c416c2d85e47faf2638502f01a6d748` (MIT) under
`third_party/LOOK-M`. The upstream `HEAD` resolved to the same commit during
the 2026-10-11 source audit. MosaicKV does not patch this checkout.

## Reproducible import blocker

The pinned source imports `MixMerLlamaAttention_drop` at:

- `LLaVA-mix_merge_v1/llava/model/language_model/llava_llama.py:30`;
- `LLaVA-mix_merge_v1/llava/eval/model_vqa_loader.py:14`; and
- `LLaVA-mix_merge_v1/llava/eval/model_vqa.py:13`.

That class is absent from the pinned
`LLaVA-mix_merge_v1/llava/model/kv_token_merge/modify_llama.py`; the similarly
named available class is `WeightedLlamaAttention_drop`. Substituting it would
change the upstream merge mode and would be a local patch, not official
LOOK-M. The worker launcher additionally contains an author-local absolute
repository path in `workers/model_workers.py`, and the published path expects
the original `liuhaotian/llava-v1.5-7b` executable rather than the Hugging
Face conversion used by the unified runtime.

## Policy

1. Do not rename a repaired or adapted executable to `lookm_official`.
2. Do not alter the submodule to make it importable.
3. A future executable fork must live outside the upstream checkout, preserve
   its license/attribution, record a patch SHA, and use a label such as
   `lookm_patched_reimpl`.
4. Until the authors publish an importable, pinned revision or an auditable
   original-checkpoint path is available, the paper may report only
   `lookm_reimpl` measurements and must identify them as local
   reimplementation results.

This limitation does not invalidate exact-only MosaicKV results or the
separate byte-matched Prompt Top-K/VL-Cache-reimplementation lanes. It blocks
only an **official LOOK-M** reproduction claim.
