# Backend capability matrix

Audit target: the installed common environment at
`/scratch/djy8hg/env/mosaickv` on 2026-07-21.

| Component | Installed version | Package/CUDA smoke | Model-serving parity | Native MosaicKV |
|---|---:|---|---|---|
| Hugging Face Transformers | 4.49.0 | Passed | Checkpoint gate not rerun in common lock | Research eager path only |
| vLLM | 0.7.2 | Passed | Unsupported | Unsupported |
| SGLang | 0.4.3.post4 | Passed | Unsupported | Unsupported |

Package import only establishes environment viability. It does not establish
checkpoint loading, token parity, measurement correctness, or native cache
mutation support.

## Required-operation matrix

| Required operation | Hugging Face eager | vLLM public API | SGLang public API |
|---|---|---|---|
| Inspect KV after prefill | Supported through returned `past_key_values` | Unsupported | Unsupported |
| Select whole KV blocks | Supported in the owned research adapter | Unsupported | Unsupported |
| Inject modified KV | Supported through a model-compatible `Cache` | Unsupported | Unsupported |
| Preserve original logical positions | Supported only with explicit tested position inputs | Unsupported | Unsupported |
| Restore blocks during decode | Supported in the owned explicit decode loop | Unsupported | Unsupported |

“Unsupported” means no safe public API was found. Internal tensors or allocator
methods are not treated as extension contracts.

## Hugging Face source boundary

`transformers/cache_utils.py:341` defines `DynamicCache`, whose `update`
starts at line 398. LLaVA passes `past_key_values` into its language model at
`transformers/models/llava/modeling_llava.py:433` and returns them at line 474.
LLaVA-OneVision does the same at
`transformers/models/llava_onevision/modeling_llava_onevision.py:752` and line
793.

The Llama query projection is
`transformers/models/llama/modeling_llama.py:245`; RoPE is applied at line 275
before the cache update beginning at line 277. The Qwen2 query projection is
`transformers/models/qwen2/modeling_qwen2.py:145`; RoPE is applied at line 167
before the cache update. Qwen2.5-VL exposes `q_proj` at
`transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:732` and updates the
cache after rotary processing. Thus the current adapters record cached keys as
post-RoPE and fail closed on unsafe prototype averaging.

## vLLM source boundary

The public entry point is `vllm/entrypoints/llm.py:52`. `generate`, `chat`, and
`encode` begin at lines 277, 620, and 756. None exposes post-prefill cache
tensors or a request-scoped cache-mutation callback. Internal cache ownership
is split between `vllm/worker/cache_engine.py:20` and
`vllm/core/block_manager.py:21`. Their swap/copy operations do not provide a
public transaction that also preserves logical positions and scheduler state.

Native selection, prototype replacement, and decode-time restoration are
therefore unsupported for vLLM 0.7.2.

## SGLang source boundary

The public entry point is `sglang/srt/entrypoints/engine.py:73`, with
`Engine.generate` at line 112. It has no post-prefill KV callback.
`ReqToTokenPool` (`sglang/srt/mem_cache/memory_pool.py:46`),
`TokenToKVPoolAllocator` (line 119), the layer KV buffers (abstract
`set_kv_buffer` at line 109), and `RadixCache`
(`sglang/srt/mem_cache/radix_cache.py:79`) have separate ownership rules.
There is no public atomic operation that updates these structures together
while preserving multimodal logical positions.

Native selection, prototype replacement, and decode-time restoration are
therefore unsupported for SGLang 0.4.3.post4.

## Execution rule

Only the common environment may be used. The supported verification command
does not load model weights:

```bash
source mosaickv/scripts/cache_env.sh
sbatch --reservation=bi_fox_dgx mosaickv/slurm/env_smoke.sbatch
```

Backend support must remain unsupported until a clean-tree pinned-checkpoint
run passes its backend-specific parity gate.
