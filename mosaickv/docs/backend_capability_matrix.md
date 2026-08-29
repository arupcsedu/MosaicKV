# Backend capability matrix

Audit target: the common environment at `/scratch/djy8hg/env/mosaickv`,
2026-08-29. A package import, a model-serving pass, token parity, and a native
cache-mutation pass are separate evidence levels.

| Component | Installed version | Package state | FullKV model gate | Native MosaicKV |
|---|---:|---|---|---|
| Hugging Face Transformers | 4.55.2 | Imports; CUDA smoke passed | Qwen2.5-VL-3B SDPA job `17346718` and FA2 job `17348171`; LLaVA-1.5 eager job `17855955`; OneVision-7B BF16 eager job `17856497` passed checkpoint-specific gates | Exact selection for supported adapters; Qwen2.5-VL M-RoPE and LLaVA/OneVision standard-RoPE prototype/repair transforms on eager; fused compressed paths fail closed |
| vLLM V1 | 0.10.2 | Imports and loads Qwen2.5-VL-3B and LLaVA-1.5 | Job `17351185` passed Qwen Stage A; clean LLaVA job `19022147` passed scored-prefix, answer, score, and A-B-A parity on three tasks | Hash-bound common-selector `mosaickv_exact` request-wide projection passed clean development job `19022148`; prototype/repair unsupported |
| SGLang | 0.5.1.post3 | Imports and launches Qwen2.5-VL-3B | Job `17351612` passed Stage A, request isolation, byte accounting, and exact 32/32 parity with the numerically corresponding HF-SDPA reference | Version-pinned whole-block transport passed job `17871006`; paper method rows remain unsupported until the common-selector plan bridge passes |

Rows remain non-canonical while the worktree is dirty. No pending or failed
job is a support claim.

## Required-operation matrix

| Required operation | Hugging Face eager | vLLM 0.10.2 V1 | SGLang 0.5.1.post3 |
|---|---|---|---|
| Inspect KV after prefill | Supported through returned `past_key_values` | Unsupported request-scoped API | Unsupported request-scoped API |
| Select whole KV blocks | Supported in the owned explicit loop | Supported by the repository plugin using a SHA-bound common-selector request-wide projection | Supported by the spawn-safe request-wide transaction; current selector is validation-only |
| Inject modified KV | Supported through a compatible HF cache object | Unsupported | Storage write exists, ownership transaction unsupported |
| Preserve original logical positions | Explicit adapter state and position inputs | Supported by the version-pinned plugin: scheduler counters remain logical while attention receives compact physical tables | Supported by separate active `seq_lens` and logical `orig_seq_lens` plus patched decode/M-RoPE positions |
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
supplies a model-derived M-RoPE inverse/anchor transform; LLaVA-1.5 and
OneVision supply the corresponding standard Llama/Qwen2 rotary transform.
All transforms are version- and position-gated.

## vLLM installed-source boundary

The audited V1 path uses:

- `vllm/v1/worker/gpu_model_runner.py:155` — `GPUModelRunner`;
- `gpu_model_runner.py:862` — `_prepare_inputs`;
- `vllm/v1/core/kv_cache_manager.py:84` — `KVCacheManager`;
- `vllm/v1/core/kv_cache_utils.py:159` — `KVCacheBlock`;
- `vllm/v1/attention/backends/utils.py:37` — `CommonAttentionMetadata`; and
- `vllm/v1/attention/backends/flash_attn.py:112` — `FlashAttentionMetadata`.

The installed package exposes no public atomic request callback, but the
repository now supplies a fail-closed 0.10.2 general plugin spanning
`Scheduler.update_from_output`, scheduler-to-worker block-table replacement,
`GPUModelRunner._update_states`, and compact attention metadata in
`GPUModelRunner._prepare_inputs`. Prefix caching is disabled and logical/M-RoPE
positions remain unchanged. A100 job `17869843` passed identity token parity,
50% physical-block release, exact accounting, determinism, and A-B-A request
isolation. The later clean common-selector job `19022148` bound per-request HF
selector traces into vLLM plans, retained mandatory blocks and normalized
cross-head evidence, reduced active bytes by at least 30% relative to Prompt
Top-K on every tested row, and passed an A-B-A compressed-request probe. This
supports exact selection only; see `vllm_native_blocker.md`.

## SGLang installed-source boundary

The audited path uses:

- `sglang/srt/entrypoints/engine.py:89` and `:140` — `Engine.generate`;
- `sglang/srt/mem_cache/memory_pool.py:50` — `ReqToTokenPool`;
- `memory_pool.py:165` and `:372` — `MHATokenToKVPool.set_kv_buffer`;
- `sglang/srt/mem_cache/allocator.py:118` — `TokenToKVPoolAllocator`;
- `sglang/srt/mem_cache/radix_cache.py:120` — `RadixCache`;
- `sglang/srt/model_executor/forward_batch_info.py:165` — `ForwardBatch`; and
- `sglang/srt/layers/attention/base_attn_backend.py:14` — `AttentionBackend`.

The installed package has no public transaction spanning these structures, but
the repository now provides a fail-closed 0.5.1.post3 scheduler patch. It is
installed inside SGLang's spawned GPU process, compacts `ReqToTokenPool`, frees
dropped allocator locations, preserves `orig_seq_lens`, updates active
`seq_lens`, and owns the disabled-Radix/ChunkCache release paths. A100 job
`17871006` passed identity parity, 50% physical compaction, byte accounting,
allocator self-check, determinism, and A-B-A isolation. This validates the
transport only; see `sglang_native_blocker.md`.

## Execution rule

Only the common environment may be used. Every cache path must remain below
`/scratch/djy8hg`, and backend support requires a pinned-checkpoint GPU artifact
plus the backend-specific parity report.
