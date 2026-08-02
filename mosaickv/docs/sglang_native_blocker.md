# SGLang native MosaicKV transaction status

Audit target: `sglang==0.5.1.post3`, Qwen2.5-VL-3B BF16,
`torch_native` attention on an A100. Re-audited and experimentally gated:
2026-08-02.

## Verdict

The former scheduler/allocator/Radix blocker is resolved by a repository-owned,
spawn-safe, version-pinned scheduler patch. Whole request-token blocks can be
released after multimodal prefill while attention uses the compact active
length and decode positions retain the original logical/M-RoPE length.

This is an **experimental native transport gate**, not yet a paper-authorized
`mosaickv_exact` result. The mandatory/even-coverage selector is a deterministic
transaction validator only. Paper rows require a plan from MosaicKV's common
forecast/graph/utility selector. Prototype replacement and residual repair
remain unsupported in SGLang.

## Implemented transaction

`mosaickv.backends.sglang_native_patch.install` is invoked by the controlled
server launcher. SGLang starts its GPU scheduler with multiprocessing `spawn`,
so the engine process target is replaced with a repository wrapper that
installs the patch inside the child before constructing `Scheduler`.

For the audited source, the patch:

1. intercepts
   `SchedulerOutputProcessorMixin.process_batch_result_prefill` at complete
   prefill;
2. compacts `ReqToTokenPool.req_to_token` in selected logical-block order and
   frees dropped physical token locations through
   `TokenToKVPoolAllocator.free`;
3. changes `ScheduleBatch.seq_lens` to the active physical length while
   retaining `orig_seq_lens` as the logical length;
4. patches `ForwardBatch.init_new` so one-dimensional and Qwen M-RoPE decode
   positions derive from `orig_seq_lens` while attention reads active lengths;
5. frees only compact active ownership through both `ChunkCache` and disabled
   `RadixCache` request-release paths; and
6. rejects retraction/recompute so stale logical mappings cannot be reused.

Radix/prefix caching, overlap scheduling, CUDA graphs, speculative decoding,
non-`torch_native` attention, allocator pages other than one, and chunked
request prefills fail explicitly. The server remains session-free and request
IDs remain globally unique.

## A100 evidence

Exploratory Slurm job `17871006` passed the fail-closed gate on two distinct
synthetic multimodal requests:

- ordinary SGLang FullKV and native retention `1.0` produced identical greedy
  token IDs for both requests;
- each prompt contained 18 source blocks and retention `0.5` retained 9;
- measured active KV storage fell from 11,132,928 bytes to 5,824,512 bytes per
  request;
- both compressed requests generated 16 valid tokens;
- repeated trials selected identical blocks;
- an A-B-A probe reproduced the first request after a distinct request; and
- SGLang's idle allocator self-check passed after every compressed request.

The raw exploratory artifacts are below
`/scratch/djy8hg/runs/mosaickv/native-sglang-17871006`. The worktree was dirty,
so the run validates implementation behavior but is not a canonical paper
measurement.

Two preceding exploratory failures are retained as engineering evidence:
`17869844` exposed that parent class patches do not cross scheduler `spawn`, and
`17870743` exposed that disabled Radix selects `ChunkCache`, whose ordinary
release would free zeroed logical slots twice. The final patch and gate cover
both defects; neither failed run is a support claim.

## Remaining boundary

SGLang still exposes no equivalent public atomic transaction. Its custom
attention interface does not coordinate allocator ownership and request/Radix
metadata:
<https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md>.
The repository patch is therefore tied to 0.5.1.post3 and must be re-audited
after every upgrade.

Until the common-selector plan bridge passes selection equivalence, the
experiment matrix rejects compressed SGLang paper rows rather than relabeling
the transport fallback as MosaicKV.
