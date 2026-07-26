# SGLang native MosaicKV blocker

Audit target: `sglang==0.5.1.post3` in the common environment.

## Verdict

Native MosaicKV is unsupported. The feature gate remains fail-closed and no
native, prototype, repair, or simulated SGLang row may be emitted.

## Installed source inspected

- `sglang/srt/entrypoints/engine.py:89` and `:140` — `Engine.generate`;
- `sglang/srt/mem_cache/memory_pool.py:50` — `ReqToTokenPool`;
- `memory_pool.py:165` and `:372` — `MHATokenToKVPool.set_kv_buffer`;
- `sglang/srt/mem_cache/allocator.py:118` — `TokenToKVPoolAllocator`;
- `sglang/srt/mem_cache/radix_cache.py:120` — `RadixCache`;
- `sglang/srt/model_executor/forward_batch_info.py:165` — `ForwardBatch`; and
- `sglang/srt/layers/attention/base_attn_backend.py:14` —
  `AttentionBackend`.

The layer KV write primitive does not atomically update request-to-token
mappings, allocator ownership, Radix entries, scheduler state, and multimodal
logical positions. The public engine has no post-prefill cache callback.

## Required upstream interface

A safe integration needs a scheduler-owned transaction that freezes one
request at the prefill/decode boundary; exposes its logical positions and KV
locations; validates and commits retained locations without renumbering;
updates mappings, allocator ownership, and Radix entries together; and supports
a synchronized restore operation. Prototype replacement additionally requires
per-layer K/V layout validation.

Ad hoc mutation can leave Radix nodes pointing to freed slots, make the
scheduler reuse live memory, corrupt M-RoPE positions, or leak request data.
Until an upstream transaction or reviewed version-pinned native patch exists,
whole-block selection, prototype replacement, and residual repair remain
unsupported.
