# SGLang backend

## Stage A: FullKV measurement wrapper

The common environment pins `sglang==0.5.1.post3`. The correctness profile
launches the installed HTTP server with overlap scheduling and CUDA graphs
disabled, Torch-native attention, deterministic greedy sampling, a persistent model,
and explicitly controlled warmup. It records:

- streamed TTFT, inter-token/decode latency, request latency, and throughput;
- server prefill/queue/e2e metadata when exposed;
- Radix cached-token counts and Prometheus cache statistics;
- exact logical active-KV bytes from model geometry;
- process GPU memory and the exact server command; and
- deterministic repeated trials plus an A-B-A request-isolation probe and a
  warm-B versus post-`/flush_cache` cold-B token comparison.

Installed 0.5.1 routes and stream semantics are version-gated:
`/get_server_info` supplies scheduler/memory-pool metadata, and
the requested `meta_info.output_token_logprobs` supplies cumulative,
prefix-validated authoritative token IDs. `BatchStrOut.output_ids` are only
detokenizer deltas and are not used as the quality token sequence. Stage A
emits only `full_kv`, retention `1.0`, `native_mosaickv=false` rows.

SGLang 0.5.1 hard-codes Qwen resize limits in
`sglang/srt/multimodal/processors/qwen_vl.py::resize_image`. The repository-local
`mosaickv.backends.sglang_server` launcher preserves SGLang's ordinary import
order, then sets the audited HF/lmms limits before `TokenizerManager` is
constructed. The installed package is not edited. Both values and the exact
launcher command are recorded in engine metadata.

Exploratory A100 job `17351612` completed two synthetic samples with identical
tokens across repeated trials, exact active-byte accounting, and a passing
A-B-A plus warm-B/cold-B request-isolation check. The two reports under
`/scratch/djy8hg/runs/mosaickv/sglang-fullkv-0.5.1-fix8/parity-sdpa/` pass all
input/model/generation/text checks and exact 32/32 token parity against HF
SDPA. HF eager differed at one near-tie token on the white image; the controlled
HF SDPA job `17352472` reproduced SGLang exactly, identifying attention-kernel
numerics rather than preprocessing or state leakage as the cause. Triton
attention remains a separate optimized-production follow-up and may not inherit
the Torch-native verdict.

Commands:

```bash
source mosaickv/scripts/cache_env.sh
sbatch mosaickv/slurm/sglang_fullkv.sbatch
sbatch mosaickv/slurm/sglang_hf_fullkv_parity.sbatch
```

All caches and outputs must be below `/scratch/djy8hg`; `HF_TOKEN` is accepted
only through the process environment.

## Stage B: native MosaicKV

The controlled launcher now installs a spawn-safe scheduler patch for exactly
0.5.1.post3. It compacts the request-to-token mapping after complete prefill,
releases dropped allocator slots, preserves `orig_seq_lens` for logical and
M-RoPE decode positions, supplies active lengths to attention, and releases
only compact ownership through both disabled-Radix and ChunkCache paths.

Exploratory A100 job `17871006` passed ordinary-FullKV versus
native-retention-1 token parity on two distinct requests, deterministic 18-to-9
block selection at retention 0.5, exact active-byte accounting, SGLang's
allocator self-check, and A-B-A isolation. Run the gate with:

```bash
MOSAICKV_NATIVE_BACKEND=sglang \
MOSAICKV_ALLOW_DIRTY_EXPLORATORY=1 \
sbatch mosaickv/slurm/native_backend_parity.sbatch
```

The fallback selector validates native transport only. It cannot be emitted as
a paper method row until a common MosaicKV selector plan is bridged and its
selected blocks are shown equivalent. Prototype injection and repair remain
unsupported. See `sglang_native_blocker.md` for exact internals and retained
failure evidence.
