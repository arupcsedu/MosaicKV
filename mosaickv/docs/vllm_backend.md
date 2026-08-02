# vLLM backend

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

The native fallback selection policy is only a transport validator. It cannot
be emitted as a paper method row, and the versioned experiment matrix continues
to reject it until a plan from the common MosaicKV selector is bridged and
selection equivalence is tested. No HF fallback or logical-only byte counter is
labeled native. Prototype injection and repair remain unsupported. See
`vllm_native_blocker.md` for the exact transaction and evidence.
