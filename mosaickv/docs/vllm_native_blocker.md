# vLLM native MosaicKV transaction status

Audit target: `vllm==0.10.2`, V1 engine, eager execution on an A100.
Re-audited and experimentally gated for Qwen2.5-VL and LLaVA-1.5 through
2026-08-29.

## Verdict

The former atomic-transaction blocker is resolved by a repository-owned,
version-pinned vLLM general plugin. Whole request-token blocks can be released
after complete multimodal prefill while logical and M-RoPE decode positions
remain unchanged. The implementation fails closed for every other vLLM
version, prefix caching, non-eager execution, speculative decoding, multiple KV
cache groups, and unsupported attention metadata.

The common-selector blocker is resolved for **exact whole-block selection**.
The HF runtime writes a request-wide projection of the common
forecast/graph/utility decision; a SHA-256-bound catalog transports that plan
to vLLM; and the plugin rejects any mismatch in model, revision, dataset,
method, task, logical length, block size, retention, mandatory blocks, or
catalog/source-trace hash. The old mandatory/even-coverage policy remains only
as an explicitly labeled transport validator.

This does not make prototype injection or residual repair native vLLM
features. They remain unsupported and must not be reported as vLLM
`mosaickv_proto` or `mosaickv_full` rows.

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

Clean-source job array `19022147`, pinned to
`0a0d6e546a0798a6540b967802b2f70947ed1391`, subsequently verified LLaVA-1.5
FullKV across HF eager and ordinary vLLM on MMStar, TextVQA, and DocVQA. All
six scored completion prefixes, decoded answers, and official sample scores
matched; all three persistent engines passed A-B-A isolation. Five of six
complete fixed-length timing streams also matched. The sole full-stream
difference occurred after both backends had already emitted EOS, so it is
recorded as a post-terminal timing diagnostic and is not part of the decoded
completion.

Clean-source job array `19022148`, pinned to the same SHA, exercised FullKV,
Prompt Attention Top-K, and the common-selector vLLM `mosaickv_exact`
projection on two retrospective examples from each of MMStar, TextVQA, and
DocVQA. MosaicKV's minimum per-sample active-memory reduction relative to
Prompt Top-K was 33.33%, 40.00%, and 33.33%, respectively. Mean official task
scores were 0.5 versus 0.5, 0.0 versus 0.0, and 0.5 versus 0.0. A two-request
MosaicKV A-B-A probe passed. These are small, retrospective development
results; they establish implementation behavior, not paper-level efficacy.

Clean-source job `19022355`, pinned to
`e4b29a4d9c278ba2ede4dc0370a281e0464472a3`, reran the model-independent
native transaction gate. Both synthetic multimodal requests matched ordinary
vLLM across all 16 fixed-length token IDs at retention 1.0. Retention 0.5 kept
9 of 18 physical blocks, both compressed requests completed, and the A-B-A
isolation check passed. The verifier correctly marked the transport fallback
as not authorized for a paper method row; common-selector evidence comes from
the separate `19022148` run.

## Remaining boundary

vLLM still has no equivalent public transaction API. The repository plugin is
therefore deliberately pinned to 0.10.2 internals and must be re-audited after
any engine upgrade. Current connector and sparse-KV proposals do not replace
this boundary: <https://github.com/vllm-project/vllm/issues/44223> and
<https://github.com/vllm-project/vllm/issues/5751>.

The common-selector plan bridge now passes its development gates. A paper row
still requires a larger fixed paired cohort, confidence intervals, complete
clean provenance, and the same official-score and measured-active-byte gates.
Only `mosaickv_exact__vllm_requestwide_projection` is supported. The matrix
must continue rejecting vLLM prototype, repair, simulated-native, prefix-cache,
CUDA-graph, and unsupported-kernel combinations.
