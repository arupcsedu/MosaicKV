# SGLang native MosaicKV blocker

Audit target: `sglang==0.4.3.post4` in the common environment.

## Verdict

Native MosaicKV is unsupported. No native SGLang row may be emitted and the
feature gate must remain fail-closed.

## Installed source inspected

- `sglang/srt/entrypoints/engine.py:73` and `:112` — public engine and
  generation entry point;
- `sglang/srt/mem_cache/memory_pool.py:46` — `ReqToTokenPool`;
- `sglang/srt/mem_cache/memory_pool.py:119` —
  `TokenToKVPoolAllocator`;
- `sglang/srt/mem_cache/memory_pool.py:180` — layer KV pool;
- `sglang/srt/mem_cache/memory_pool.py:109` — abstract KV write primitive;
- `sglang/srt/mem_cache/radix_cache.py:79` — `RadixCache`; and
- `sglang/srt/managers/scheduler.py:1927` — cache-flush ownership boundary.

The public engine exposes no post-prefill cache callback or request-scoped
block-selection transaction. The KV write primitive updates layer storage but
does not atomically update request-to-token mappings, allocator ownership,
Radix entries, scheduler state, and multimodal logical positions.

## Missing hook

A safe integration needs a scheduler-owned transaction that:

1. freezes one request at the prefill/decode boundary;
2. exposes its logical positions and owned KV locations;
3. validates and commits retained locations without renumbering positions;
4. updates request mappings, allocator ownership, and Radix entries together;
5. optionally replaces K/V values with layer/layout validation; and
6. restores blocks at a synchronized decode boundary without exposing them to
   another request.

## Correctness risks without the hook

Ad hoc mutation can leave Radix nodes pointing to freed slots, make the
scheduler reuse live KV memory, corrupt multimodal M-RoPE positions, or leak
one request's cache into another. A successful kernel call alone would not
prove ownership or position correctness.

## Proposed upstream interface

An upstream interface should expose a versioned request cache snapshot plus
atomic `commit_kv_selection` and `restore_kv_blocks` scheduler operations. The
commit must validate request identity, logical positions, allocated slots,
Radix ownership, layer layout, and exact byte accounting before publication.

Until that interface or a reviewed version-pinned native patch exists,
whole-block selection, prototype replacement, and residual repair remain
unsupported.
