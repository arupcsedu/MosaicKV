# FullKV reference and measurement protocol

## Contract

`mosaickv.fullkv.FullKV` is the Hugging Face ground-truth path. It performs an
ordinary eager prefill followed by explicit greedy, token-by-token decoding. It
does not call `model.generate()` and it never prunes, selects, merges,
quantizes, transfers, replaces, or repairs cache state. Its method metadata is
fixed to `fullkv`, backend to `huggingface`, and retention ratio to `1.0`.

The command preflight rejects a FullKV configuration unless:

- backend and attention implementation are `huggingface` and `eager`;
- `temperature = 0`, `do_sample = false`, and retention ratio is exactly `1.0`;
- forecasting, graph construction, selection, prototypes, residuals, and repair
  are all disabled.

The active cache object passes through an explicit identity checkpoint after
prefill. Both its Python identity and tensor-payload byte count must remain
unchanged. Compression and repair are therefore recorded as structural zero
costs with zero event invocations; these values describe absent operations and
are not inferred performance measurements.

`FullKVEvaluationModel` exposes the same path through the repository's
`LocalEvaluationModel` interface for local and lmms-eval quality evaluation. It
returns the standard evaluation metrics while retaining the complete raw
FullKV trial records through its `raw_measurements` property. Requests that
change the configured output length or greedy-decoding controls fail closed.

## Timing boundaries

All GPU intervals use `torch.cuda.Event(enable_timing=True)`. Every start and
stop boundary calls `torch.cuda.synchronize(device)`, and each raw row records
the number of synchronization calls and phase event invocations. A measured
trial separates:

- image/video encoder execution, captured by a forward hook when invoked;
- multimodal projector execution, captured independently when invoked;
- language-model prefill;
- the no-op FullKV compression phase;
- TTFT, from the synchronized start of model execution through prefill and the
  no-op compression boundary;
- each individual decode step after the first predicted token;
- the absent repair phase; and
- total GPU latency plus a host-wall-clock observation over the same region.

Media decoding, processor/tokenizer work, model loading, telemetry queries, and
artifact writing are outside the GPU total-latency boundary. An encoder or
projector field is `null` when that module was not invoked; missing observations
are never replaced with fabricated numbers.

## Memory and machine state

Peak allocator memory is reset immediately before each measured trial, after a
synchronization, and recorded with both `torch.cuda.max_memory_allocated` and
`torch.cuda.max_memory_reserved`. Active KV payload is computed by summing
`tensor.numel() * tensor.element_size()` for every key and value tensor. FullKV
has no CPU residual state, so its residual payload is structurally zero.

Before and after every trial, read-only `nvidia-smi` queries capture GPU UUID,
name, P-state, graphics/SM/memory clocks, power draw and limit, driver version,
CUDA visibility, and all compute processes. Processes other than the current
PID on visible devices are explicitly listed as background processes. Failed
queries remain visible in `query_error`.

## Workload format

Commands consume strict, local JSONL. Paths may be absolute or relative to the
workload file. Network URLs are not accepted.

```json
{"sample_id":"example-1","prompt":"What is shown?","media":[{"kind":"image","path":"images/1.png"}]}
{"sample_id":"example-2","prompt":"Summarize the clip.","media":[{"kind":"video","frame_paths":["frames/1.png","frames/2.png"]}]}
```

An optional `system_prompt` is accepted. InternVL uses its checkpoint-specific,
already-preprocessed tensors: an image descriptor has `tensor_path`; a video
descriptor has `tensor_path` and `num_patches_list`. File contents, selected
prompts, preprocessing specification, and actual processor token IDs are
hashed into the immutable run manifest.

## Commands

All commands require a pinned 40-character model revision, a cache directory
outside the home directory, explicit artifact paths, and a run ID. Model access
uses `HF_TOKEN` only through the process environment. Downloads are disabled
unless `--allow-download` is explicitly supplied. Before using the packaged
configuration template, replace its dataset-revision placeholder and cache
budget with the immutable workload identity and preregistered comparison
budget; the command rejects placeholder revisions.

```bash
COMMON='--config configs/fullkv_onevision.toml --workload /scratch/$USER/mosaickv/workload.jsonl --cache-root /scratch/$USER/mosaickv/cache'

# One deterministically selected sample, one measured trial.
mosaickv fullkv-debug $COMMON --run-id debug-001 \
  --raw-output /scratch/$USER/mosaickv/runs/debug.raw.jsonl \
  --aggregate-output /scratch/$USER/mosaickv/runs/debug.aggregate.json \
  --log-output /scratch/$USER/mosaickv/runs/debug.log.json \
  --manifest /scratch/$USER/mosaickv/runs/debug.manifest.json

# Exactly 20 selected samples.
mosaickv fullkv-smoke $COMMON --run-id smoke-001 \
  --raw-output /scratch/$USER/mosaickv/runs/smoke.raw.jsonl \
  --aggregate-output /scratch/$USER/mosaickv/runs/smoke.aggregate.json \
  --log-output /scratch/$USER/mosaickv/runs/smoke.log.json \
  --manifest /scratch/$USER/mosaickv/runs/smoke.manifest.json

# Every row in the workload.
mosaickv fullkv-run $COMMON --run-id dataset-001 \
  --raw-output /scratch/$USER/mosaickv/runs/dataset.raw.jsonl \
  --aggregate-output /scratch/$USER/mosaickv/runs/dataset.aggregate.json \
  --log-output /scratch/$USER/mosaickv/runs/dataset.log.json \
  --manifest /scratch/$USER/mosaickv/runs/dataset.manifest.json

# One sample with three warmups and ten measured trials by default.
mosaickv fullkv-latency $COMMON --run-id latency-001 \
  --raw-output /scratch/$USER/mosaickv/runs/latency.raw.jsonl \
  --aggregate-output /scratch/$USER/mosaickv/runs/latency.aggregate.json \
  --log-output /scratch/$USER/mosaickv/runs/latency.log.json \
  --manifest /scratch/$USER/mosaickv/runs/latency.manifest.json
```

`--warmups`, `--trials`, `--bootstrap-samples`, and `--confidence-level` are
configurable on every mode. Raw JSONL preserves one row per sample and trial,
including failures. Derived JSON reports median, p5, p95, mean, population
standard deviation, and a deterministic percentile-bootstrap confidence
interval for every available measurement. With two or more completed trials
per sample, `deterministic_token_match` reports exact greedy-token agreement;
a mismatch makes the command fail.

## Slurm

The packaged job accepts `MOSAICKV_FULLKV_MODE`, `MOSAICKV_CONFIG`,
`MOSAICKV_WORKLOAD`, `MOSAICKV_RUN_ID`, `MOSAICKV_OUTPUT_DIR`, and
`MOSAICKV_CACHE_ROOT`. It does not download weights unless
`MOSAICKV_ALLOW_MODEL_DOWNLOAD=1` is exported.

```bash
sbatch --export=ALL,\
MOSAICKV_HF_PYTHON=/scratch/$USER/env/mosaickv_hf/bin/python,\
MOSAICKV_FULLKV_MODE=fullkv-latency,\
MOSAICKV_CONFIG=$PWD/configs/fullkv_onevision.toml,\
MOSAICKV_WORKLOAD=/scratch/$USER/mosaickv/workload.jsonl,\
MOSAICKV_RUN_ID=latency-001 \
slurm/fullkv.sbatch
```

The tiny CUDA architecture gate is `slurm/hf_adapter_smoke.sbatch`. It uses no
downloaded weights and validates repeated 16-token deterministic output, exact
KV payload accounting, absent FullKV transformations, and synchronized phase
boundaries. It is validation evidence, not a model-quality or performance
result.
