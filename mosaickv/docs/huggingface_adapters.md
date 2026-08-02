# Hugging Face cache adapters

## Scope and status

The runtime adapters implement the Phase A explicit full-cache path for:

- `llava-hf/llava-1.5-7b-hf`;
- `Qwen/Qwen2.5-VL-3B-Instruct` and `Qwen/Qwen2.5-VL-7B-Instruct`;
- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` and the pinned optional
  `llava-hf/llava-onevision-qwen2-7b-ov-hf`; and
- `OpenGVLab/InternVL2_5-4B` as an optional remote-code adapter.

The backend-independent three-tier constructor consumes adapter capability
metadata. Qwen2.5-VL, LLaVA-1.5, and LLaVA-OneVision declare prototype merging
and residual repair only for the audited Transformers 4.55.2 eager path. Each
captures its exact model-produced RoPE phase, de-rotates cached member keys,
merges in FP32 phase-neutral space, and re-rotates at the anchor. Any
version/API, padding, or position-map mismatch raises rather than averaging
incompatible phases. InternVL remains a post-RoPE exact-selection fallback.
The three native adapters accept eager, SDPA, and FlashAttention-2 for FullKV
execution.
Compression prefill remains eager because prompt-attention capture is not
exposed by SDPA or FlashAttention-2. The runtime may explicitly switch an
all-valid, no-repair exact cache to SDPA for decoding; arbitrary padded
per-head masks remain eager-only.
InternVL remains eager-only.

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

Compressed decoding uses explicit original positions when active and logical
lengths differ. Padded per-head layouts require the owned eager additive-mask
dispatcher; the hybrid SDPA decode gate rejects those layouts.

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

For Qwen2.5-VL, `build_prototype_key_transform` mirrors the installed
`get_rope_index`, `Qwen2_5_VLRotaryEmbedding`, and
`apply_multimodal_rotary_pos_emb` dataflow. Its transform provenance is
`transformers_4_55_2_qwen2_5_vl_get_rope_index_rotary_emb_apply_multimodal_rotary_pos_emb_v1`.
LLaVA-1.5 and OneVision capture the standard one-dimensional phase from the
installed Llama and Qwen2 rotary modules, respectively. Their provenances are
`transformers_4_55_2_llava_1_5_llama_rotary_emb_apply_rotary_pos_emb_v1` and
`transformers_4_55_2_llava_onevision_qwen2_rotary_emb_apply_rotary_pos_emb_v1`.
All inverses divide by `cos^2 + sin^2`, so attention scaling is not assumed to
be unit magnitude. Prototype and repair decode reject SDPA and
FlashAttention-2; those implementations remain exact-only.

Qwen2.5-VL's M-RoPE `rope_deltas` is copied into the prefill decode state and
restored before each token step. Its immutable decoded position range is
precomputed and reused instead of rebuilding three M-RoPE coordinates per
token. InternVL prefill uses its public multimodal wrapper, but
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

The prototype/repair gate additionally runs retention-one identity, a real
50% three-tier cache, and a forced residual restoration:

```bash
export MOSAICKV_HF_MODEL_ID=llava-hf/llava-onevision-qwen2-7b-ov-hf
export MOSAICKV_VERIFY_THREE_TIER=1
sbatch --export=ALL mosaickv/slurm/hf_three_tier_checkpoint.sbatch
```

The pinned OneVision-7B checkpoint stores FP16 tensors and declares FP16 in
its top-level config, but contains a conflicting nested BF16 text dtype.
`torch_dtype="auto"` therefore creates incompatible hidden/output-head dtypes;
uniform eager FP16 also produced non-finite validation logits. The A100 gate
pins every module to BF16 explicitly, and other precisions remain unsupported
until they pass their own finite-logit and parity gates.

For InternVL, also export `MOSAICKV_INTERNVL_PIXEL_VALUES` pointing to a tensor
artifact produced by the exact pinned public preprocessing. `HF_TOKEN`, if
needed, is inherited only through the environment and is never written by the
test or job script.

Source implementation, tiny architecture parity, and checkpoint acceptance are
separate evidence levels. The pinned Qwen2.5-VL-3B checkpoint passed the SDPA
FullKV gate in exploratory Slurm job `17346718`: generation, reinjection, and
retention-1 reconstruction each had 16/16 token agreement and maximum logit
difference `0.0`. Its dirty-tree provenance makes this implementation evidence,
not a canonical experimental row. Eager and FlashAttention-2 retain their own
independent gates.

The bounded v17 component-ablation run subsequently completed 44/44 compressed
prototype cells as `three_tier` with the registered transform provenance, plus
four exact-only no-prototype cells. This is exploratory dirty-tree checkpoint
evidence, not a clean support claim or paper result.

Exploratory A100 jobs `17855955` and `17856497` independently gated
LLaVA-1.5 FP16 and OneVision-7B BF16. Both reported 16/16 generation and cache
reinjection agreement with maximum logit difference zero, retention-one token
identity, real `three_tier` construction at retention 0.5, three forced repair
events, and a maximum re-decode count of one. Their dirty-tree provenance makes
them implementation evidence rather than paper measurements.
