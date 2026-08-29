# Paper quality–memory campaign: LLaVA-1.5 HF and vLLM

Status: **execution authorized**. Canonical execution still requires a clean
immutable source commit and reviewed, hash-bound sample ledgers.

## Scope

- Model: `llava-hf/llava-1.5-7b-hf` at revision
  `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, FP16.
- Tasks: MMStar, TextVQA, and DocVQA, each at its pinned dataset revision.
- Scale: 256 deterministic examples per task, or 768 unique examples.
- Backends: HF eager and vLLM eager.
- Within each backend: FullKV, Prompt Attention Top-K, and MosaicKV.
- HF reports the runtime's actual effective MosaicKV label. LLaVA currently
  uses the safe exact-selection fallback when prototype assignments are
  incompatible; the analysis must not call such a row a prototype/repair row.
- vLLM uses `mosaickv_exact` request-wide whole-block projection. It makes no
  prototype-merging or residual-repair claim.

The primary comparison is always backend-local. Active KV bytes from HF and
vLLM are never compared as though their physical cache layouts were identical.
Official task scores are reported separately and are not macro-averaged.

## Four-A100 time and storage estimate

The estimate uses completed jobs 19028932 (HF, eight examples per task) and
19023196 (vLLM, 32 examples per task), including selector planning and model
startup rather than model-token throughput alone.

- Expected compute: approximately 26 A100 GPU-hours.
- Expected wall time with four A100s continuously allocated: approximately
  7 hours.
- Per-worker Slurm limit: 3 hours. The whole array is expected to finish in
  about 7 hours, with an 8-hour planning bound. Queue wait is not included.
- Layout: 12 paired sample shards (three tasks × four 64-example shards), 24
  backend jobs, one A100 per job, with at most four concurrent jobs.
- Expected new result/trace storage: less than 1 GB. Existing model and dataset
  caches under `/scratch/djy8hg/cache/mosaickv` are not duplicated.

## Gates

For each backend and task, MosaicKV must use at least 30% fewer active KV bytes
than Prompt Top-K for every paired sample and have a nonnegative paired mean
official-score difference. All 256 identities must be complete and identical
across methods and backends. Every result records clean-source, model, dataset,
configuration, environment, cache-accounting, and timing provenance. vLLM must
also record successful persistent-engine request-isolation evidence. Latency is
reported but is not a rejection gate.

## Execution sequence after a reviewed setup commit

All paths below are under `/scratch`; no cache is written to a home directory.
From the reviewed immutable source commit, set:

```bash
export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv
export MOSAICKV_PAPER_LEDGER_ROOT=/scratch/djy8hg/runs/mosaickv/paper-hf-vllm-v2-input-ledgers
export MOSAICKV_PAPER_INPUT_ROOT=/scratch/djy8hg/runs/mosaickv/paper-hf-vllm-v2-inputs
export MOSAICKV_PAPER_OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/paper-hf-vllm-v2-results
```

Materialize the three master ledgers on CPU:

```bash
sbatch mosaickv/slurm/materialize_paper_quality_hf_vllm_llava15_v2.sbatch
```

After all three ledgers are present and reviewed, create immutable shard/job
specifications and submit the capped four-A100 array:

```bash
bash mosaickv/scripts/submit_paper_quality_hf_vllm_llava15_v2.sh
```

The launcher prints the GPU array job ID. Submit analysis only after all 24
array elements finish:

```bash
export MOSAICKV_PAPER_INDEX=${MOSAICKV_PAPER_INPUT_ROOT}/jobs.jsonl
export MOSAICKV_PAPER_ANALYSIS_ROOT=/scratch/djy8hg/runs/mosaickv/paper-hf-vllm-v2-analysis
sbatch mosaickv/slurm/analyze_paper_quality_hf_vllm_llava15_v2.sbatch
```

The analysis writes paired raw and summary Parquet files plus a JSON and
Markdown gate report. Failed gates remain visible; the analyzer never removes
or substitutes failed samples.
