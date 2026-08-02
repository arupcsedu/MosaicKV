# Backend capability matrix

Audit target: the common environment at `/scratch/djy8hg/env/mosaickv`,
2026-07-25. A package import, a model-serving pass, token parity, and a native
cache-mutation pass are separate evidence levels.

| Component | Installed version | Package state | FullKV model gate | Native MosaicKV |
|---|---:|---|---|---|
| Hugging Face Transformers | 4.55.2 | Imports; CUDA smoke passed | Qwen2.5-VL-3B SDPA job `17346718` and FA2 job `17348171` passed 16-token ordinary/reinjection/retention-1 parity with maximum logit difference 0 | Exact selection for supported adapters; Qwen2.5-VL eager prototype/repair through audited M-RoPE transform; other post-RoPE/fused paths fail closed |
| vLLM V1 | 0.10.2 | Imports and loads Qwen2.5-VL-3B | Job `17351185` passed Stage A and its report has exact 16/16 HF token parity | Unsupported |
| SGLang | 0.5.1.post3 | Imports and launches Qwen2.5-VL-3B | Job `17351612` passed Stage A, request isolation, byte accounting, and exact 32/32 parity with the numerically corresponding HF-SDPA reference | Unsupported |

Rows remain non-canonical while the worktree is dirty. No pending or failed
job is a support claim.

## Required-operation matrix

| Required operation | Hugging Face eager | vLLM 0.10.2 V1 | SGLang 0.5.1.post3 |
|---|---|---|---|
| Inspect KV after prefill | Supported through returned `past_key_values` | Unsupported request-scoped API | Unsupported request-scoped API |
| Select whole KV blocks | Supported in the owned explicit loop | Unsupported atomic transaction | Unsupported atomic transaction |
| Inject modified KV | Supported through a compatible HF cache object | Unsupported | Storage write exists, ownership transaction unsupported |
| Preserve original logical positions | Explicit adapter state and position inputs | No sparse logical-position commit hook | No atomic mapping/position commit hook |
| Restore blocks during decode | Owned explicit loop | Unsupported | Unsupported |

“Unsupported” means no safe installed API was found; direct tensor mutation is
not treated as an integration contract.

## Hugging Face source boundary

The adapter consumes the returned Transformers cache, explicitly controls
prefill and one-token decode, and maintains active length, original logical
length, and next decode position separately. Qwen2.5-VL, LLaVA-1.5, and
LLaVA-OneVision declare eager, SDPA, and FlashAttention-2 for FullKV. Cache
compression prefill remains eager because prompt-attention capture is not
available safely in fused paths. An explicitly labeled eager-prefill/SDPA-decode
exact path is now implemented; it is admitted only for all-valid caches without
repair and records both implementations. Cached keys are
post-RoPE, so phase-unaligned prototype construction is rejected. Qwen2.5-VL
on the exact 4.55.2 eager path supplies a model-derived M-RoPE inverse/anchor
transform; the bounded v17 run exercised real three-tier and residual state.

## vLLM installed-source boundary

The audited V1 path uses:

- `vllm/v1/worker/gpu_model_runner.py:155` — `GPUModelRunner`;
- `gpu_model_runner.py:862` — `_prepare_inputs`;
- `vllm/v1/core/kv_cache_manager.py:84` — `KVCacheManager`;
- `vllm/v1/core/kv_cache_utils.py:159` — `KVCacheBlock`;
- `vllm/v1/attention/backends/utils.py:37` — `CommonAttentionMetadata`; and
- `vllm/v1/attention/backends/flash_attn.py:112` — `FlashAttentionMetadata`.

These internal structures do not expose an atomic request-scoped callback
between multimodal prefill and decode that can select sparse blocks, preserve
logical/M-RoPE positions, update prefix ownership, and safely restore blocks.
The measurement wrapper uses public generation plus read-only cache/timing
telemetry. See `vllm_native_blocker.md`.

## SGLang installed-source boundary

The audited path uses:

- `sglang/srt/entrypoints/engine.py:89` and `:140` — `Engine.generate`;
- `sglang/srt/mem_cache/memory_pool.py:50` — `ReqToTokenPool`;
- `memory_pool.py:165` and `:372` — `MHATokenToKVPool.set_kv_buffer`;
- `sglang/srt/mem_cache/allocator.py:118` — `TokenToKVPoolAllocator`;
- `sglang/srt/mem_cache/radix_cache.py:120` — `RadixCache`;
- `sglang/srt/model_executor/forward_batch_info.py:165` — `ForwardBatch`; and
- `sglang/srt/layers/attention/base_attn_backend.py:14` — `AttentionBackend`.

`set_kv_buffer` is a storage primitive, not an atomic transaction across
request mappings, allocator ownership, Radix entries, scheduler state, and
logical positions. See `sglang_native_blocker.md`.

## Execution rule

Only the common environment may be used. Every cache path must remain below
`/scratch/djy8hg`, and backend support requires a pinned-checkpoint GPU artifact
plus the backend-specific parity report.
