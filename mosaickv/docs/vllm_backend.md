# vLLM backend

The backend has two deliberately separate states:

- Stage A is implemented as measured **vLLM FullKV** inference. It performs no
  cache pruning, merging, quantization, offloading, or repair.
- Stage B is fail-closed. vLLM 0.11.2 has no safe sparse-logical-block commit
  hook, so `--enable-mosaickv` is unsupported and produces no result row. See
  [the native blocker](vllm_native_blocker.md).

## Supported Stage A models

The version-pinned wrapper accepts:

| Model | vLLM registry architecture | Media | Runtime status |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | `Qwen2_5_VLForConditionalGeneration` | image, multi-image, video | **GPU verified** at the pinned revision |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `Qwen2_5_VLForConditionalGeneration` | image, multi-image, video | Source-registered; checkpoint run not yet verified |
| `llava-hf/llava-1.5-7b-hf` | `LlavaForConditionalGeneration` | image, multi-image | Source-registered; vLLM checkpoint run not yet verified |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | `LlavaOnevisionForConditionalGeneration` | image, multi-image, video | Source-registered; checkpoint run not yet verified |

Registry presence is checked against installed vLLM 0.11.2 source. A model is
not reported as runtime-verified until its GPU command succeeds.

## Controlled inputs and outputs

`EvaluationHarness` constructs the same ordered `MultimodalMessage` objects
used by the HF runtime. The vLLM adapter renders them with the checkpoint's
pinned `AutoProcessor` chat template, passes the unchanged PIL/video payloads
as `multi_modal_data`, and uses the same immutable processor revision. The
LLaVA-1.5 no-template fallback is byte-for-byte the same
`USER: <image> ... ASSISTANT:` construction as the HF adapter.

Generation is greedy (`temperature=0`, `top_p=1`) with the same seed and
`max_new_tokens`. Results are ordinary common-harness rows with
`backend=vllm`, `method=full_kv`, and `retention_ratio=1.0`. The wrapper repeats
an identical request for cache diagnostics and requires every trial's token ID
sequence to match. A mismatch becomes a failed sample rather than being
silently averaged.

Offline execution resolves the immutable Hub snapshot once and passes the same
local directory to both Transformers and vLLM. This avoids optional processor
file probes being misreported as missing checkpoints and prevents the two
loaders from resolving different revisions.

Raw JSONL and trace output need only the pinned vLLM environment. Pass an
explicit `--parquet-output` when running in an environment that also contains
PyArrow; otherwise the manifest records aggregate metrics as `not_applicable`
instead of installing an undeclared dependency or fabricating an artifact.

## Measurements

Every sample has an atomic JSON trace containing every request trial:

- host-observed TTFT, token timestamps, inter-token latencies, decode time,
  request latency, and token throughput;
- vLLM `IterationStats` TTFT, inter-token latency, prefill, and decode values;
- prompt/generated token counts and finish reason;
- `RequestOutput.num_cached_tokens` and its prompt-token hit rate;
- `MultiModalCacheStats` query/hit counts for vLLM's multimodal preprocessor
  cache;
- process-tree GPU memory sampled from `nvidia-smi`, including resident
  baseline, peak, and peak delta;
- exact engine controls: vLLM version, block size, eager execution/no CUDA
  graph, prefix caching, tensor parallelism, memory utilization, maximum model
  length, and attention-backend configuration.

The accepted Qwen run used vLLM automatic attention selection. Its engine log
selected FlashAttention for the language attention path while retaining a
compatible model-specific vision path. Forcing `VLLM_ATTENTION_BACKEND` to
FlashAttention is unsupported for this checkpoint: job `17158394` failed
during vision profiling because that forced build requires a head dimension
divisible by 32. The manifest therefore records `attention_implementation` as
`vllm_auto`, not as a fabricated single backend.

vLLM 0.11.2 does not expose a per-request GPU encoder-output-cache hit counter.
The trace therefore records repeated-request timing and the distinct
multimodal preprocessor cache counters, and explicitly marks encoder-cache hit
attribution unavailable. It does not infer a hit from latency. Likewise,
active K/V bytes are left null because public request output does not expose
the allocated per-request block set; total GPU process memory is not mislabeled
as active K/V.

The common result row uses the first cold/probe trial for TTFT, latency, and GPU
memory. All trials remain in the trace so cache-warm behavior and prefix-cache
hit rates are auditable without collapsing observations.

## GPU validation record

Validation was performed on 2026-07-20 with one A100-SXM4-80GB under Slurm.
The artifacts are development evidence only because the worktree was dirty;
their manifests correctly set `canonical_eligible=false`.

- Recovery job `17157973` installed all 159 exact vLLM lock entries, passed
  `pip check`, imported the complete vLLM surface, and passed a CUDA 2x2 matrix
  multiplication with CUDA 12.8 and driver 595.71.05. Its subsequent model
  step failed on the pre-fix offline snapshot lookup and emitted no result row.
- Standalone environment job `17158688_1` completed successfully and recorded
  `support_verified=true` for all 159 pins/imports and the A100 CUDA smoke.
- Final vLLM job `17158441` completed one image/prompt FullKV sample, retained
  both raw trials, generated 16 tokens deterministically, recorded TTFT,
  inter-token latency, throughput, request latency, GPU memory, prefix cache,
  multimodal cache, and the encoder-cache observability boundary, and passed
  `verify_vllm_backend.py --require-gpu-measurements`.
- Controlled HF eager FullKV job `17158501` used the same model revision,
  prompt/media/preprocessing/tokenization hashes, BF16 precision, seed, and
  generation-parameter hash. Its 16 token IDs and decoded text exactly matched
  the vLLM result. The generated comparison is
  [`results/vllm_hf_fullkv_parity.json`](../results/vllm_hf_fullkv_parity.json).

The accepted vLLM [manifest](/scratch/djy8hg/runs/mosaickv/vllm-fullkv/vllm-fullkv-17158441/manifest.json)
and [raw trace](/scratch/djy8hg/runs/mosaickv/vllm-fullkv/traces/vllm-fullkv-17158441/ci-red-7fb7e52384d6.json)
remain outside the repository run tree. No synthetic validation latency is a
paper result.

## Commands

Create and GPU-verify the pinned environment explicitly:

```bash
export MOSAICKV_PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
export MOSAICKV_ENV_ROOT=/scratch/djy8hg/env
export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv
./mosaickv/scripts/create_envs.sh vllm
sbatch --reservation=bi_fox_dgx mosaickv/slurm/env_smoke.sbatch
```

One local GPU request using cached weights:

```bash
export HF_HOME=/scratch/djy8hg/cache/mosaickv/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv/vllm
/scratch/djy8hg/env/mosaickv/bin/mosaickv evaluate \
  --config mosaickv/configs/vllm_fullkv.yaml \
  --task synthetic_smoke --subset-size 1 --cache-probe-repeats 2 \
  --vllm-max-model-len 4096 --local-files-only \
  --output-dir /scratch/djy8hg/runs/mosaickv/vllm-fullkv
```

Slurm uses the same command and stores the validator output:

```bash
sbatch mosaickv/slurm/vllm_fullkv.sbatch
```

The native feature gate can be checked without a GPU or model load:

```bash
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/verify_vllm_backend.py --vllm-version 0.11.2
```
