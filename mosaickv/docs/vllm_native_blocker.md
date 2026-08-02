# vLLM native MosaicKV transaction status

Audit target: `vllm==0.10.2`, V1 engine, Qwen2.5-VL-3B BF16, eager
execution on an A100. Re-audited and experimentally gated: 2026-08-02.

## Verdict

The former atomic-transaction blocker is resolved by a repository-owned,
version-pinned vLLM general plugin. Whole request-token blocks can be released
after complete multimodal prefill while logical and M-RoPE decode positions
remain unchanged. The implementation fails closed for every other vLLM
version, prefix caching, non-eager execution, speculative decoding, multiple KV
cache groups, and unsupported attention metadata.

This is an **experimental native transport gate**, not yet a paper-authorized
`mosaickv_exact` row. Its deterministic mandatory/even-coverage selector exists
only to exercise the engine transaction. A paper row must first supply a block
plan produced by MosaicKV's common forecast/graph/utility selector. Prototype
injection and residual repair remain unsupported in vLLM.

## Implemented transaction

The plugin entry point is
`mosaickv.backends.vllm_native_plugin:register`, registered under
`vllm.general_plugins`. For the audited installed source it:

1. intercepts `vllm.v1.core.sched.scheduler.Scheduler.update_from_output` at
   the complete-prefill boundary;
2. releases unselected request-owned `KVCacheBlock` objects through
   `BlockPool.free_blocks`, replaces their logical table slots with vLLM's
   null block, and leaves `Request.num_computed_tokens` unchanged;
3. sends a complete replacement block table through
   `Scheduler._make_cached_request_data`;
4. replaces the persistent worker row in
   `GPUModelRunner._update_states` rather than appending duplicate blocks;
5. retains ordinary logical token and M-RoPE position construction in
   `GPUModelRunner._prepare_inputs`, then supplies FlashAttention with a compact
   physical block table and active sequence length; and
6. filters null placeholders when the request is freed, because vLLM does not
   maintain null-block reference counts.

Prefix caching is disabled, so no prefix owner can retain a freed physical
block. CUDA graphs, cascade/AOT scheduling, speculative decoding, and
multi-group caches are rejected before a compressed decode.

## A100 evidence

Exploratory Slurm job `17869843` passed the fail-closed gate on two distinct
synthetic multimodal requests:

- ordinary vLLM FullKV and native retention `1.0` produced identical greedy
  token IDs for both requests;
- each prompt contained 18 source blocks and retention `0.5` retained 9;
- measured active KV storage fell from 11,132,928 bytes at identity to
  5,824,512 bytes at retention `0.5` for each request;
- both compressed requests generated 16 valid tokens;
- repeated trials selected identical blocks; and
- an A-B-A probe reproduced the first request's tokens after a distinct
  intervening request.

The raw exploratory artifacts are below
`/scratch/djy8hg/runs/mosaickv/native-vllm-17869843`. The worktree was dirty,
so these observations validate implementation behavior but are not canonical
paper measurements.

## Remaining boundary

vLLM still has no equivalent public transaction API. The repository plugin is
therefore deliberately pinned to 0.10.2 internals and must be re-audited after
any engine upgrade. Current connector and sparse-KV proposals do not replace
this boundary: <https://github.com/vllm-project/vllm/issues/44223> and
<https://github.com/vllm-project/vllm/issues/5751>.

The next scientific gate is a common-selector plan bridge. Until that bridge
passes parity and selection-equivalence tests, the experiment matrix continues
to reject compressed vLLM paper rows rather than relabeling the transport
fallback as MosaicKV.
