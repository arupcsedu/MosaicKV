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

Native whole-block selection is unsupported for the audited V1 source. The
flag fails before model loading, and no simulation is substituted. The exact
classes inspected and missing atomic sparse-position transaction are recorded
in `vllm_native_blocker.md`.
