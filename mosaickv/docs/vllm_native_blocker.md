# vLLM native MosaicKV blocker

Audit target: `vllm==0.10.2`, V1 engine, in the common environment.

## Verdict

Native MosaicKV is unsupported. `--enable-mosaickv` remains fail-closed and
no native, prototype, repair, or simulated backend row may be emitted.

## Installed source inspected

- `vllm/v1/worker/gpu_model_runner.py:155` — `GPUModelRunner`;
- `gpu_model_runner.py:862` — `_prepare_inputs`;
- `vllm/v1/core/kv_cache_manager.py:84` — `KVCacheManager`;
- `vllm/v1/core/kv_cache_utils.py:159` — `KVCacheBlock`;
- `vllm/v1/attention/backends/utils.py:37` — `CommonAttentionMetadata`; and
- `vllm/v1/attention/backends/flash_attn.py:112` —
  `FlashAttentionMetadata`.

V1 can allocate/reuse physical blocks and report prefix hits, but it exposes
no request-scoped post-prefill transaction that jointly updates the scheduler,
block table, attention slot mapping, prefix ownership, and sparse original
logical positions.

## Required upstream interface

A safe integration needs a scheduler-owned prefill/decode-boundary transaction
that exposes immutable logical positions and owned physical blocks, validates a
whole-block selection, commits its block table and position map atomically,
and invalidates or updates prefix-cache ownership. Prototype replacement would
add strict K/V dtype/layout validation; repair would require a synchronized
restore transaction.

Without that boundary, direct mutation risks stale reads, early block reuse,
M-RoPE renumbering, cross-request leakage, and silent output corruption. A
reviewed version-pinned native patch implementing this transaction—or an
equivalent upstream public API—is required before the native flag can be
enabled.
