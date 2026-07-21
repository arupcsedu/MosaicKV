# Backend capability matrix

> Common-environment reset (2026-07-21): the source/API findings below remain
> version-specific historical evidence for Transformers 4.57.6, vLLM 0.11.2,
> and SGLang 0.5.10.post1. Current execution uses the common lock
> (Transformers 4.49.0, vLLM 0.7.2, SGLang 0.4.3.post4) and is **unsupported**
> until its installed source is re-audited and its clean-tree gates pass. Do
> not transfer a support claim or source line number across these versions.

Audit date: 2026-07-20. “Native safe API” means a documented, supported engine
extension point whose ownership and synchronization rules permit MosaicKV to
operate without patching scheduler/runner internals. Merely being able to reach
a tensor from Python is not sufficient.

## Verdict

Hugging Face eager execution is the only viable first cache-modification
reference because the caller owns the model forward pass and returned cache.
Measured vLLM FullKV is executable and GPU-verified for pinned Qwen2.5-VL-3B,
but this does not expose a mutation hook. The correctness-first SGLang FullKV
HTTP wrapper is GPU-verified for pinned Qwen2.5-VL-3B/7B in the isolated
SGLang 0.5.10.post1 environment. Its controlled HF eager output comparison did
not establish token parity, so optimized SGLang settings remain disabled. The
installed vLLM and SGLang sources prove that both engines internally
represent the data MosaicKV would need, while neither public API exposes the
complete contract. A native integration requires version-pinned backend
changes plus correctness tests; it cannot be implemented as an ordinary
`LLM.generate` or `Engine.generate` plugin.

The SGLang source below was checked in the functional pinned prefix
`/scratch/djy8hg/env/mosaickv/lib/python3.11/site-packages`; it is
byte-identical for the cited files to the originally audited broken source
prefix. Runtime execution uses only the functional pinned prefix.

## Required-operation matrix

| Required operation | Hugging Face Transformers 4.57.6 | vLLM 0.11.2 public/native API | SGLang 0.5.10.post1 public/native API |
|---|---|---|---|
| Inspect KV after prefill | **Supported** in an owned eager forward via returned `Cache` | **Unsupported.** Public `LLM` has generation/chat/encode but no post-prefill KV callback. | **Unsupported.** Public `Engine` has generate/encode/cache flush but no post-prefill KV callback. |
| Select whole KV blocks | **No native block API.** A research adapter may group returned token tensors itself. | **Unsupported.** Internal `KVCacheManager` allocates/caches paged blocks, but exposes no external selection-policy hook. | **Unsupported.** Internal allocators manage token/page locations, but expose no external selection-policy hook. |
| Inject modified K/V | **Supported in an owned research adapter** by constructing/updating a compatible cache; it is not a high-level generation API | **Unsupported.** Internal cache tensors and copy/reshape operators exist, but arbitrary external writes are not a supported API. | **Unsupported.** Internal `set_kv_buffer` writes tensors, but its class comments reserve getters for attention backends and no public API owns synchronization. |
| Maintain original logical positions after compaction | **Supported only if the adapter passes explicit position/cache-position tensors and verifies each model's RoPE/M-RoPE semantics** | **Unsupported externally.** Internal `BlockTable` maps logical positions to physical slots; no public contract permits rewriting it. | **Unsupported externally.** `ForwardBatch.positions`, request-to-token mappings, and output cache locations are coupled runner internals. |
| Restore selected blocks during decode | **Supported in a custom eager decode loop** | **Unsupported for MosaicKV's dynamic repair contract.** KV connectors can transfer cache before a layer, but scheduler allocation and block tables must already agree. | **Unsupported as a public API.** HiCache can load evicted prefix-cache nodes, but is not a policy hook for arbitrary per-layer residual repair. |

## Hugging Face reference surface

The functioning source is under
`/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421/lib/python3.11/site-packages/transformers`.

- `cache_utils.py:84` defines `DynamicLayer`; its docstring at `:87` fixes the
  per-layer key/value shape as `[batch_size, num_heads, seq_len, head_dim]`.
  `DynamicLayer.update` at `:98` appends on the sequence dimension.
- `cache_utils.py:910` defines `DynamicCache`; its documentation at `:914`
  repeats the cache shape. `Cache` layer indexing makes the actual tensors
  accessible to an owned adapter.
- `models/llava/modeling_llava.py:308` defines
  `LlavaForConditionalGeneration`; its `language_model` property is at `:354`
  and its forward accepts/returns `past_key_values` at `:373`, `:424`, and
  `:447`.
- `models/qwen2_5_vl/modeling_qwen2_5_vl.py:590` defines the text attention;
  `q_proj` is at `:622`, query computation at `:645`, and cache update at
  `:658-660`. The wrapper's `language_model` property is at `:1396`.
- `models/llava_onevision/modeling_llava_onevision.py:661` defines the wrapper;
  `language_model` is at `:716` and `past_key_values` flows through `:738`,
  `:824`, and `:849`.
- `models/llama/modeling_llama.py:197` (`LlamaAttention`) exposes `q_proj` at
  `:210` and updates the cache at `:246`; `models/qwen2/modeling_qwen2.py:122`
  (`Qwen2Attention`) exposes `q_proj` at `:134` and updates at `:163`.

This is sufficient for the full-cache reference and a research cache adapter.
It is not a license to mutate cache tensors in place without tests: each model
must preserve its own `cache_position`, RoPE, and M-RoPE behavior, and retention
1.0 must reproduce full-cache output within the documented tolerance.

## vLLM source evidence

### Public boundary

`vllm/entrypoints/llm.py:94` defines `LLM`. Its relevant public methods are
`generate` (`:383`), `chat` (`:881`), and `encode` (`:965`); none exposes the
model runner, a post-prefill callback, cache tensors, a block table, or a
per-decode restoration callback. Therefore the five required native API
capabilities are unsupported at this boundary.

### Internal representation

- `vllm/v1/attention/backends/flash_attn.py:90` defines
  `get_kv_cache_shape`; at `:99` it returns
  `(2, num_blocks, block_size, num_kv_heads, head_size)`. Layout-specific
  strides are selected at `:105-110`. Forward unbinds the K/V axis at `:603`
  and writes through `slot_mapping` at `:615-625`.
- `vllm/v1/worker/gpu_model_runner.py:256` defines `GPUModelRunner` and stores
  `self.kv_caches` at `:351`. Logical token positions are held at `:456` and
  M-RoPE positions at `:500`. Slot mappings are computed from positions at
  `:1275`. KV tensor initialization/binding occurs in
  `initialize_kv_cache_tensors` at `:4918` and `bind_kv_cache` at `:4947`.
- `vllm/v1/worker/block_table.py:15` defines `BlockTable`. Its
  `compute_slot_mapping` (`:125`) maps request positions through physical
  block IDs, while commit methods at `:183` and `:186` publish block and slot
  tables. These are runner internals, not an extension API.
- `vllm/v1/core/kv_cache_manager.py:93` defines `KVCacheManager`.
  `get_computed_blocks` (`:176`), `allocate_slots` (`:218`), `get_blocks`
  (`:403`), `get_block_ids` (`:407`), and `cache_blocks` (`:411`) implement
  prefix-cache scheduling rather than arbitrary scientific cache selection.
- `vllm/_custom_ops.py:2121`, `:2143`, `:2178`, and `:2190` expose internal
  reshape/cache, FlashAttention cache, copy-block, and swap-block operations.
  Calling these alone would desynchronize manager metadata and is not safe.

### Closest extension seam: KV connector

`vllm/distributed/kv_transfer/kv_connector/v1/base.py:144` defines the abstract
`KVConnectorBase_V1`. It can `register_kv_caches` (`:219`), begin a load
(`:237`), wait per layer (`:255`), save a KV layer (`:269`), and wait for save
completion (`:291`). This is a supported-looking transfer seam for loading and
saving engine-owned paged caches. It does **not** supply MosaicKV's forecasting,
utility scoring, budget selection, changed logical block table, or uncertainty
repair policy. A connector prototype may validate transfer, but the native API
matrix remains unsupported for the complete operation.

## SGLang source evidence

### Public boundary

`sglang/srt/entrypoints/engine.py:143` defines `Engine`. Public methods include
`generate` (`:271`), `encode` (`:441`), `flush_cache` (`:772`), and memory
release/resume (`:1040`, `:1046`). None exposes per-request K/V, the
request-to-token mapping, post-prefill mutation, or a decode-step repair hook.

### Internal representation

- `sglang/srt/mem_cache/memory_pool.py:126` defines `ReqToTokenPool`; its
  `req_to_token` tensor and `write` at `:149` map logical sequence locations to
  physical token slots.
- `memory_pool.py:741` defines `MHATokenToKVPool`. `_create_buffers` beginning
  at `:845` creates one key and value tensor per layer with shape
  `[size + page_size, head_num, head_dim]` (value may use `v_head_dim`).
  `get_kv_buffer` is at `:992`, `set_kv_buffer` at `:995`, and
  `move_kv_cache` at `:1034`.
- The comment at `memory_pool.py:973` is decisive: `get_key_buffer`,
  `get_value_buffer`, and `get_kv_buffer` are intended only for the attention
  backend, “not for information purpose,” because they participate in
  layer-wise loading synchronization.
- `sglang/srt/mem_cache/allocator.py:117` defines
  `TokenToKVPoolAllocator`; `alloc` (`:144`), `free` (`:155`), CPU copy
  (`:167`), and load (`:170`) own slot lifetime. The paged form is
  `PagedTokenToKVPoolAllocator` at `:356`, with extend/decode allocation at
  `:403` and `:451`.
- `sglang/srt/model_executor/forward_batch_info.py:280` defines `ForwardBatch`.
  Its `out_cache_loc` (`:294`), `positions` (`:327`), `req_to_token_pool`
  (`:369`), and `token_to_kv_pool` (`:370`) show the coupled state that a
  correct mutation must preserve.
- `sglang/srt/layers/attention/flashinfer_backend.py:782`, `:788`, `:865`,
  `:893`, and `:900` show attention kernels writing and reading through those
  pools during extend and decode.

### Closest existing restoration mechanism: HiCache

`sglang/srt/mem_cache/hiradix_cache.py:55` defines `HiRadixCache`.
`write_backup` (`:610`), eviction (`:779`), and `load_back` (`:880`) implement
host/storage backup for radix-prefix-cache nodes. This is useful design evidence
for a future tiered-cache integration, but it restores SGLang-owned prefix
nodes. It does not expose arbitrary MosaicKV per-layer block selection,
prototype/residual values, preserved custom logical positions, or
uncertainty-triggered decode repair through the public engine. It is therefore
not a native solution to the requested contract.

### Measurement-only boundary and Stage B verdict

The Stage A wrapper uses only `/generate`, `/server_info`, and `/metrics`. It
records deterministic streaming latency, Radix cached-token observations,
process-tree GPU memory, exact server arguments, and model-derived logical KV
bytes without presenting those observations as a mutation API. The native
feature flag fails before server launch for `0.5.10.post1`; no HF-side
selection is replayed and labeled native. The inspected structures, missing
atomic hook, proposed upstream interface, and request-leakage risks are listed
in [the SGLang native blocker](sglang_native_blocker.md).

## Integration consequence

For vLLM and SGLang milestones, pin an exact backend commit, implement behind a
small versioned adapter, and add backend-native invariants covering block-table
ownership, allocation/free lifetime, CUDA-stream synchronization, graph capture,
prefix caching, tensor/pipeline parallelism, and M-RoPE. Do not claim backend
support until the five operations above are exercised end to end and retention
1.0 matches the backend's own full-cache reference.
