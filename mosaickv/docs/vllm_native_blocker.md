# vLLM native MosaicKV blocker

Audit target: installed `vllm==0.11.2` source under
`/scratch/djy8hg/env/drc_rag_bench_env/lib/python3.11/site-packages/vllm`.
Audit date: 2026-07-20. The `--enable-mosaickv` feature is **unsupported** for
this version and fails before model weights load. The runtime does not emit a
MosaicKV result row and does not relabel ordinary vLLM inference as native.

## Source inspected

- `entrypoints/llm.py:94` defines public `LLM`; `LLM.generate` begins at
  `:383`. `v1/engine/async_llm.py:54` defines `AsyncLLM` (also exported as
  `AsyncLLMEngine`), whose streaming `generate` begins at `:351`. Neither
  request API accepts a post-prefill cache policy, logical-position map, or
  block-table mutation callback.
- `v1/core/kv_cache_manager.py:176` implements `get_computed_blocks` and
  `:218` implements `allocate_slots`. These schedule complete prefix blocks;
  there is no request-scoped arbitrary retention-policy hook.
- `v1/worker/block_table.py:15` defines `BlockTable`.
  `compute_slot_mapping` at `:125` obtains the physical block with
  `positions // block_size`; `commit_block_table` and `commit_slot_mapping` at
  `:183` and `:186` copy the coupled metadata to the GPU. Removing an interior
  entry would make its list index a new logical block ordinal.
- `v1/worker/gpu_model_runner.py:1163` prepares token positions, M-RoPE
  positions, block-table slot mappings, and attention metadata in one runner
  path. `execute_model` starts at `:2630`. The owned K/V tensors are held by
  `GPUModelRunner.kv_caches`; initialization/binding is internal at
  `:4918-4953`.
- `v1/attention/backends/flash_attn.py:90-99` declares the cache layout
  `[2, num_blocks, block_size, num_kv_heads, head_size]`; the forward path
  writes via `slot_mapping`. A tensor copy alone cannot update allocator,
  scheduler, and logical-position ownership.
- `distributed/kv_transfer/kv_connector/v1/base.py:144` defines
  `KVConnectorBase_V1`. It registers caches (`:219`), starts loads (`:237`),
  waits for a layer (`:255`), and saves a layer (`:269`). It transfers
  scheduler-authorized cache data; it does not change the local request's
  attention-visible logical block set.

The installed model registry does support the requested architectures:
`LlavaForConditionalGeneration` at `model_executor/models/registry.py:317`,
`LlavaOnevisionForConditionalGeneration` at `:326-328`, and
`Qwen2_5_VLForConditionalGeneration` at `:360-362`. Model registration is not
evidence that cache mutation is safe.

## Missing hook

MosaicKV whole-block selection needs a post-prefill, pre-first-decode callback
that atomically receives and returns all of the following for one request:

1. immutable original logical token positions, including Qwen M-RoPE axes;
2. physical block IDs and their allocation lifetime;
3. the attention-visible sparse logical-block order or an explicit position
   per cached token;
4. per-layer K/V tensors after prefill completion;
5. scheduler sequence length, next decode position, and prefix-cache ownership;
6. a commit/rollback operation synchronized with the model-runner CUDA stream.

vLLM 0.11.2 exposes no such atomic boundary. Its dense block table uses table
index as logical block ordinal. Compacting that table would renumber cached
positions for paged attention; leaving holes would point slot computation at
invalid or freed blocks. Preserving only the query token's RoPE position is
insufficient because cached K is already position encoded, causal masking and
M-RoPE metadata still depend on the original sequence.

## Required patch

A safe upstream patch should introduce a versioned `SparseKVBlockPolicy` at the
scheduler/runner boundary:

```text
PrefillKVView {
  request_id,
  kv_cache_group,
  physical_block_ids,
  logical_token_positions,
  mrope_positions,
  next_decode_position,
  read_only_layer_kv_views,
}

SparseKVSelection {
  retained_physical_block_ids,
  retained_logical_token_positions,
  original_logical_sequence_length,
}
```

The engine, not the plugin, must validate ownership and block alignment, build
attention metadata that accepts sparse original positions, retain prefix-cache
references until commit, and atomically publish the selection. The attention
backend contract must state whether it accepts sparse K positions. The patch
also needs a request-output field with exact retained KV bytes and selected
block IDs so scientific accounting does not inspect process-private tensors.

## Correctness risks

- interior-block compaction silently renumbers paged-attention positions;
- Qwen2.5-VL three-axis M-RoPE can diverge even if scalar sequence length looks
  correct;
- freeing blocks before metadata commit can create use-after-free or cross-
  request data exposure;
- prefix-cache hashes and reference counts can claim blocks that a plugin
  removed;
- tensor/pipeline parallel ranks may apply different selections;
- asynchronous cache copies can race the model-runner stream;
- CUDA graph captures bake in metadata and addresses, so Stage B must remain
  eager until a graph-safe contract exists.

Prototype injection and residual repair remain out of scope. Although internal
copy/swap operators can modify memory, no API validates arbitrary K/V
replacement against cache dtype/layout, allocator state, and active kernels.

## Implemented feature behavior

`mosaickv evaluate ... --backend vllm --enable-mosaickv` calls the versioned
capability guard before processor or engine construction. For vLLM 0.11.2 it
returns reason code
`audited_0_11_2_missing_sparse_logical_block_table_hook` and exits nonzero.
Any other vLLM version is `unaudited_vllm_version`. This is the Stage B outcome
until the upstream interface exists and retention-1 parity passes.
