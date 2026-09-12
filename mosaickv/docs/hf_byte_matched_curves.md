# HF byte-matched selector curves

This experiment addresses the review concern that the original MosaicKV and
Prompt Top-K results were measured at different active-KV operating points.
It is a selector-policy comparison, not another comparison of nominal token
retention ratios.

## Controlled comparison

The protocol is
[`configs/experiments/hf_byte_matched_curves_v1.yaml`](../configs/experiments/hf_byte_matched_curves_v1.yaml).
For every paired example, MosaicKV exact selection and Prompt Top-K use the
same:

- LLaVA-1.5-7B model and immutable revision;
- FP16 precision and HF eager attention/decode;
- processor, prompt, image, preprocessing, and tokenization;
- 64-token deterministic greedy generation;
- four-token cache blocks and last-token mandatory policy; and
- hard active-KV cap: 16, 22, 30, 40, or 60 MiB.

The worker evaluates both methods through one loaded adapter. The analyzer
rejects the campaign unless the realized active KV byte count is exactly equal
for every `(task, sample_id, budget)` pair. A shared configured cap without a
shared realized byte count is not accepted as byte matched.

The curve subset contains 256 deterministic examples from each of MMStar,
TextVQA, and DocVQA. IDs are selected by hashing only task, sample ID, and the
fixed selection seed. The selector never reads answers, scores, latency, or
memory outcomes. These examples come from the existing 9,000-example campaign,
so the curve is a fixed characterization experiment rather than a new held-out
generalization claim.

## Execution

Use only the common environment and scratch-backed caches:

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh
export MOSAICKV_HF_BYTE_CURVE_INPUT_ROOT=/scratch/djy8hg/runs/mosaickv/hf-byte-curves-<git-sha>/inputs
export MOSAICKV_HF_BYTE_CURVE_OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/hf-byte-curves-<git-sha>/results
export MOSAICKV_HF_BYTE_CURVE_ANALYSIS_ROOT=/scratch/djy8hg/runs/mosaickv/hf-byte-curves-<git-sha>/analysis
bash mosaickv/scripts/submit_hf_byte_matched_curves.sh
```

Submission requires a clean tracked and untracked worktree. The launcher first
runs one 16-example shard from each task. Only if all three pass exact byte
matching does Slurm release the remaining 45 shards. At most four one-A100 jobs
run concurrently; every job requests 12 CPUs and has a two-hour limit.

The dependent analyzer writes:

- `rows.parquet`;
- `summary.parquet`;
- `paired_comparisons.parquet` with paired bootstrap intervals;
- `byte_match_validation.parquet`;
- `figure3_byte_matched_curves.tex`;
- `byte_matched_table.tex`;
- `report.md`; and
- `validation.json`.

No value from these outputs may enter a measured-results table until the GPU
campaign and fail-closed analyzer both complete successfully.
