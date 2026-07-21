# vLLM native MosaicKV blocker

Audit target: `vllm==0.7.2` in the common environment.

## Verdict

Native MosaicKV is unsupported. No native row may be emitted and
`--enable-mosaickv` must remain fail-closed.

## Installed source inspected

- `vllm/entrypoints/llm.py:52` — public `LLM` entry point;
- `vllm/entrypoints/llm.py:277`, `:620`, and `:756` — public generation,
  chat, and encoding methods;
- `vllm/worker/cache_engine.py:20` — physical KV cache allocation and
  swap/copy operations;
- `vllm/core/block_manager.py:21` — scheduler-owned logical/physical block
  management;
- `vllm/worker/model_runner.py:997` — GPU runner base and per-step KV inputs;
  and
- `vllm/attention/backends/abstract.py:72` and `:91` — backend cache-shape and
  block-copy interfaces.

The public entry point exposes no post-prefill callback, request block table,
arbitrary KV replacement transaction, logical-position override, or
decode-time restoration callback. Internal cache copy/swap primitives do not
transfer scheduler ownership and position metadata atomically.

## Missing hook

A safe integration needs a request-scoped transaction after prefill and before
the first decode step that:

1. exposes immutable logical positions and owned physical block IDs;
2. permits selecting complete blocks without renumbering positions;
3. validates and commits the replacement block table atomically;
4. optionally replaces K/V values with explicit dtype/layout validation;
5. supports a synchronized decode-time restoration transaction; and
6. invalidates or updates prefix-cache ownership without affecting another
   request.

## Correctness risks without the hook

Directly mutating internal tensors can desynchronize the scheduler block
manager, cache engine, attention slot mapping, prefix-cache ownership, and
multimodal RoPE positions. Likely outcomes include cross-request leakage,
stale-cache reads, incorrect next-token positions, premature block reuse, or
silent output corruption.

## Proposed upstream interface

An upstream interface should provide a versioned `PrefillCacheView` and an
atomic `commit_block_selection(request_id, retained_blocks,
logical_positions)` operation. A separate synchronized
`restore_blocks(request_id, blocks)` operation should be permitted only at a
decode boundary. Both operations should validate attention-backend support and
return exact active-byte accounting.

Until such an interface or a reviewed version-pinned native patch exists,
whole-block selection, prototype injection, and residual repair remain
unsupported.
