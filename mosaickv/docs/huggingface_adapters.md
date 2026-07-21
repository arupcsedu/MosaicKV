# Hugging Face cache adapters

## Scope and status

The runtime adapters implement the Phase A explicit full-cache path for:

- `llava-hf/llava-1.5-7b-hf`;
- `Qwen/Qwen2.5-VL-3B-Instruct` and `Qwen/Qwen2.5-VL-7B-Instruct`;
- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`; and
- `OpenGVLab/InternVL2_5-4B` as an optional remote-code adapter.

The backend-independent three-tier constructor can consume adapter capability
metadata, but all four adapters currently declare prototype merging and
residual repair unsupported. Because their cached keys are post-RoPE, the
constructor fails closed to selection-only exact cache state instead of
averaging incompatible rotary phases. Eager attention is the sole enabled implementation;
constructing an adapter around SDPA or FlashAttention-2 fails before execution.

The package remains importable without PyTorch. Loading an adapter is an
explicit operation in the common environment and requires an immutable model
revision. No adapter import downloads a model.

## Public contract

`HuggingFaceMultimodalAdapter` supplies:

- `prepare_inputs(messages, media)`;
- `prefill(prepared)`;
- `decode_one_token(token_id, state)`;
- `extract_past_key_values(value)` and `inject_past_key_values(snapshot)`;
- `capture_query_vectors()`;
- `get_modality_map(prepared)`;
- `get_logical_sequence_length(prepared_or_state)`;
- `get_cache_layout(cache_or_snapshot)`; and
- `supports_prototype_merge` and `supports_residual_repair`.

The candidate path never calls `model.generate()`. `prefill` performs one
multimodal forward with `use_cache=True`; `decode_one_token` performs exactly
one language-token forward. The next token is always `argmax(logits)`. The only
call to `model.generate()` is isolated in `compare_with_generate`, where it is
the untouched Hugging Face acceptance reference.

`DecodeState` deliberately keeps three different quantities:

- `active_cache_length`: physical K/V slots currently held;
- `logical_sequence_length`: original sequence positions represented so far;
- `next_decode_position`: absolute cache position for the next input token.

Compressed decoding currently fails closed when active and logical lengths
differ. This prevents a future cache-selection implementation from silently
using packed physical indices as logical positions.

## Cache and query representation

Adapters inspect the cache object returned at runtime. They support modern
objects exposing `to_legacy_cache`/`from_legacy_cache` and legacy tuple/list
caches, preserve the source class for reinjection, and record every observed K
and V shape, dtype, device, and sequence dimension. Unsupported cache types are
rejected; no common head count or concrete cache class is assumed. The audited
adapters each declare sequence dimension `-2` from their model-specific source
contract; the shared extractor does not hard-code that axis for future models.

All four audited language architectures store keys after rotary embedding:

- LLaVA-1.5 uses `LlamaAttention.forward`, which calls
  `apply_rotary_pos_emb` before `past_key_values.update`;
- Qwen2.5-VL uses `Qwen2_5_VLAttention.forward`, which calls
  `apply_multimodal_rotary_pos_emb` before `past_key_values.update`;
- LLaVA-OneVision uses `Qwen2Attention.forward`, which applies RoPE before
  `past_key_values.update`; and
- the pinned InternVL remote wrapper contains `Qwen2ForCausalLM`, with the
  same `Qwen2Attention` ordering.

Consequently, `cached_key_state` is `post_rope`. Query hooks attach to every
language layer's `q_proj`; the captured tensors are reshaped to
`[batch, heads, sequence, head_dim]` and are explicitly labeled
`q_proj_output_pre_rope`. Hooks are removed after each forward and never alter
weights.

Qwen2.5-VL's M-RoPE `rope_deltas` is copied into the decode state and restored
before each token step. InternVL prefill uses its public multimodal wrapper, but
decode calls the wrapper's public `language_model`: the pinned remote
`InternVLChatModel.forward` requires pixel inputs and reruns visual extraction,
so it is not a valid token-decode entry point.

## Media preparation

Native Transformers adapters accept standardized MosaicKV messages and PIL,
NumPy, or processor-compatible image/video payloads. The model processor owns
patching and placeholder expansion. Modality spans are derived from the
expanded language token IDs, not estimated from image size.

InternVL has no public `AutoProcessor` at the pinned revision. Its adapter is
therefore optional and requires images already transformed by the checkpoint's
pinned public preprocessing to tensors shaped `[num_patches, C, H, W]`.
Frame-based video uses `InternVLVideo(pixel_values, num_patches_list)`. The
adapter expands the official `<img><IMG_CONTEXT>...</img>` representation and
keeps image and video-frame spans distinct in its modality map.

## Correctness gates

Three gates each generate at least 16 tokens:

1. `compare_with_generate` compares the custom full-cache loop with greedy
   `GenerationMixin.generate` under neutral, fixed-length generation settings.
2. `compare_cache_reinjection` compares the untouched explicit loop with an
   otherwise identical loop whose complete prefill cache is extracted and
   reinjected at retention ratio `1.0`.
3. `compare_mosaickv_retention_one` blockizes that snapshot through the core
   `FullKVState`, gathers every block into `ExactTier`, reconstructs the cache,
   restores its logical/next-decode positions, and compares decoding with the
   untouched explicit loop.

All three report token agreement and maximum absolute logit difference. The pinned
checkpoint gate requires 100% token agreement and uses absolute logit tolerance
`1e-4`, which is stricter than the maximum FP16/BF16 tolerances in
`REPRODUCIBILITY.md`. The no-download, randomly initialized FP32 architecture
test uses `1e-6`. A failure is not converted into support by widening the
tolerance.

Run dependency-light tests:

```bash
/scratch/djy8hg/env/mosaickv/bin/python -m pytest mosaickv/tests/unit/test_hf_adapters.py
```

After explicitly creating the HF environment, run the no-download architecture
test on Slurm:

```bash
sbatch mosaickv/slurm/hf_adapter_smoke.sbatch
```

Run one authoritative checkpoint gate at a time. Cached weights are required by
default; downloads occur only with an explicit opt-in:

```bash
export MOSAICKV_HF_MODEL_ID=llava-hf/llava-onevision-qwen2-0.5b-ov-hf
export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv
sbatch --export=ALL mosaickv/slurm/hf_adapter_smoke.sbatch
```

For InternVL, also export `MOSAICKV_INTERNVL_PIXEL_VALUES` pointing to a tensor
artifact produced by the exact pinned public preprocessing. `HF_TOKEN`, if
needed, is inherited only through the environment and is never written by the
test or job script.

Checkpoint acceptance remains unsupported until this command passes for an
audited revision in the common environment and its complete validation record
is preserved. Source implementation, tiny architecture parity, and checkpoint
acceptance are separate evidence levels. The current common-lock test gate
passed all 29 affected adapter/environment tests without downloading weights;
this validates package compatibility only, not checkpoint acceptance.
