# vLLM backend

## Paper backend scope

The current paper scope uses vLLM and does not use SGLang. The controlled
one-model verification target is
`llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`
on MMStar, TextVQA, and DocVQA. `slurm/vllm_llava15_three_tasks.sbatch`
runs two deterministically selected examples per task through both HF FullKV
and vLLM FullKV, then checks official lmms-eval scoring, repeated-token
determinism, A-B-A request isolation, timing and cache telemetry, logical KV
byte accounting, immutable provenance, and retention-1 HF token/answer/score
parity. This is a verification workload, not a paper result table.

The vLLM result schema reports active KV bytes at the
`packed_prefill_active_kv_tensor_storage` boundary: live prefill blocks times
the configured vLLM block size and the pinned model's layer/head/dtype
geometry. Live final-block padding is included. The free pool and reserved
engine capacity are excluded; logical prefill bytes, padding bytes, and the
trace-only post-decode size are recorded separately.

## Stage A: FullKV measurement wrapper

The common environment pins `vllm==0.10.2`. The wrapper uses the V1
`AsyncLLMEngine` with eager execution and CUDA graphs disabled. The Slurm
launcher exposes the common environment's executables to spawned workers,
loads GCC 11.4 for Torch 2.8 native extensions, and disables the unnecessary
FlashInfer stochastic-sampler JIT for deterministic greedy decoding. The
wrapper records:

- streamed TTFT, inter-token latency, request latency, and throughput;
- V1 iteration prefill/decode timing;
- `RequestOutput.num_cached_tokens` and prefix hit ratio;
- read-only frontend multimodal processor-cache counter deltas from
  `AsyncLLM.processor.mm_processor_cache._cache.stat()`;
- process GPU memory sampled with `nvidia-smi`; and
- exact engine arguments, immutable model revision, raw repeated trials, and
  trace provenance.

For LLaVA-1.5 only, the repository's version-pinned general plugin routes the
cache-free CLIP vision encoder's existing vLLM `MultiHeadAttention` branch to
PyTorch SDPA. Decoder paged attention remains vLLM FlashAttention. This avoids
xFormers 0.0.32.post1 importing external FlashAttention 2.8.3, which exceeds
xFormers' declared audited maximum of 2.8.2; the version check is not ignored
and no environment package is modified.

The same MosaicKV message builder, processor/chat template, prompt, media,
checkpoint-compatible slow image processor, and deterministic generation
parameters are used. Qwen `min_pixels=200704` and `max_pixels=1605632` are
passed through vLLM's supported `mm_processor_kwargs` and recorded in engine
metadata. Stage A emits only `full_kv`, retention `1.0`,
`native_mosaickv=false` rows.

Prefix caching is disabled by default for the correctness/reference path. In
exploratory job `17349373`, vLLM loaded the pinned Qwen2.5-VL-3B weights and
completed generation, but its second identical request (a prefix-cache hit)
changed greedy tokens. `--enable-vllm-prefix-caching` remains an explicit
diagnostic option and the wrapper fails if cold/cached tokens differ; it may
not produce a reference row until that gate passes.

Exploratory uncached job `17351185` passed Stage A: both trials generated the
same 16 token IDs, active telemetry was complete, the second request recorded
a multimodal-processor-cache hit, and the verifier accepted the artifact. Its
controlled report at
`/scratch/djy8hg/runs/mosaickv/vllm-fullkv-0.10.2-fix9/parity/ci-red.json`
passed all input/model/generation checks and exact 16/16 HF token parity. The
older mismatch was caused by vLLM using checkpoint resize defaults while HF
used the lmms profile, not by model-weight or KV-cache divergence.

Commands:

```bash
source mosaickv/scripts/cache_env.sh
sbatch mosaickv/slurm/vllm_fullkv.sbatch
sbatch mosaickv/slurm/vllm_hf_fullkv_parity.sbatch
sbatch mosaickv/slurm/vllm_llava15_three_tasks.sbatch
```

All caches and outputs must be below `/scratch/djy8hg`; `HF_TOKEN` is accepted
only through the process environment.

## Stage B: native MosaicKV

The repository now supplies a vLLM general plugin for exactly 0.10.2. It
commits request-wide whole-block release after complete prefill, replaces the
scheduler-to-worker block table, gives FlashAttention a compact physical table
and active length, and retains the original logical/M-RoPE decode positions.
Prefix caching, speculative decoding, CUDA graphs, cascade/AOT scheduling,
non-FlashAttention metadata, and multiple cache groups fail explicitly.

Exploratory A100 job `17869843` passed ordinary-FullKV versus native-retention-1
token parity on two distinct multimodal requests, deterministic repeated
selection, exact physical-byte accounting, a retention-0.5 release from 18 to
9 prompt blocks, and an A-B-A request-isolation probe. The gate command is:

```bash
MOSAICKV_NATIVE_BACKEND=vllm \
MOSAICKV_ALLOW_DIRTY_EXPLORATORY=1 \
sbatch mosaickv/slurm/native_backend_parity.sbatch
```

The common selector bridge is now implemented. HF forecast/graph/utility
traces carry a request-wide block projection; write-once catalog builders bind
each entry to the model, revision, task, dataset revision, method, source trace,
and SHA-256; and vLLM consumes the per-request selected and mandatory block IDs
through `SamplingParams.extra_args`. Request-wide aggregation caps repeated
per-head votes, normalizes utility within layer/head routes, reserves the best
available image/OCR evidence block, and fills the remaining hard byte ceiling
deterministically. The plugin records the committed selection and catalog
provenance in both its scheduler audit and the model trace.

Clean LLaVA development arrays `19022147` and `19022148` passed scored-prefix
HF/vLLM FullKV parity, official score parity, measured active-byte accounting,
and persistent-engine A-B-A isolation. The compressed two-example-per-task
gate used 33.33--40.00% fewer active KV bytes than Prompt Top-K without a lower
mean official task score on MMStar, TextVQA, or DocVQA. Because the examples
were retrospective and very small, these observations are implementation
evidence rather than paper results.

The fixed follow-up protocol uses 32 paired samples per task and 10,000 paired
bootstrap resamples. Selector catalogs may contain different physically
attainable retention ratios across requests while retaining one common block
size. The hash-bound catalog ratio, not the engine's fallback ratio, controls
each scheduler transaction and is recorded as the row's effective retention
ratio. This allows all 32 requests to share one persistent engine and avoids
sample-wise model reloads. See
`configs/experiments/paper_quality_memory_llava_vllm_v1.yaml` and
`slurm/vllm_llava15_paper_quality_memory_v1.sbatch`.

No HF fallback or logical-only byte counter is labeled native. The supported
method name is `mosaickv_exact__vllm_requestwide_projection`; prototype
injection and repair remain unsupported. See `vllm_native_blocker.md` for the
exact transaction, fail-closed boundary, and evidence.
