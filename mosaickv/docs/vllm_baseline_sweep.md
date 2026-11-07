# Matched-byte vLLM baseline sweep

This is a **development-only**, retrospective backend sweep, not a paper-native
reproduction, untouched evaluation, or claim of quality improvement. No packages
are installed or changed and no existing HF worker or artifact is modified.

## Preregistered scope

- LLaVA-1.5-7B, revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, FP16.
- MMStar accuracy, TextVQA consensus score, DocVQA ANLS; report separately.
- Eight ID-only, previously evaluated examples per task, identical at every cap.
- Shared prefill physical KV caps: 40/80/120 MiB. LLaVA's 16-token serving blocks
  occupy 8 MiB, so these caps are exactly attainable for sufficiently long inputs.
  All methods must use the largest feasible storage below the cap, checked per sample.
- FullKV quality reference; Prompt Top-K, LOOK-M, PrefixKV, VL-Cache, MosaicKV exact.
- Deterministic 64-token output, seed 20271001; zero warmups, one measured trial.
  Timing is exploratory, not a publication latency result. FullKV is excluded
  from latency comparisons.
- Nine one-A100 shards; 12 CPUs, 96 GiB host memory, two hours maximum each.
  Reservation `bi_fox_dgx`, partition `bii-gpu`; at most four A100s globally,
  including already-running HF jobs, enforced by the admission controller.

## Method semantics

Compressed rows use `__vllm_requestwide_projection`. An HF selector produces a
hash-bound offline plan; the audited vLLM 0.10.2 plugin actually drops whole
physical request blocks, preserves logical positions, and appends decode tokens.
This is **not** online native forecasting/planning or an exact per-layer port.
All serving arms share first/last mandatory whole blocks and 16-token granularity.
The source HF selectors use one-token blocks and common boundary protection.

LOOK-M source pivots include text-prior scoring, but their merged K/V values are
not injected into vLLM. PrefixKV's pinned offline profile is deterministically
rescaled to the byte cap; layer-wise lengths and decode eviction are not executed
by vLLM. The launcher explicitly separates the HF `effective_profile_ratio`
decode policy from serving's append-only behavior. VL-Cache's layer/head choices
are projected to a shared request table. These are local variants, never official
reproductions. Prototype/repair and secondary baseline ports are not claimed.

## Gates and artifacts

Each shard generates all six HF source arms in a separate process, exits that
process, then starts each vLLM method in its own engine process. It also runs
ordinary FullKV versus retention-1 native identity and A–B–A isolation in every
engine. Checks cover complete raw rows, task references, prompt/media hashes,
fixed token lengths, native selection telemetry, maximal physical-cap utilization,
exact per-sample byte matching, and retention-1 token parity. Compressed quality
is measured, including negative results; execution success never implies quality
parity. The CPU analysis produces paired score differences and bootstrap 95% CIs.

All artifacts live in a new scratch root: preserved source patch/untracked files,
full source inventory, pinned inputs/configs/catalogs, per-arm JSONL/Parquet,
manifests, traces, error logs, terminal status, and `analysis/summary.parquet`.
Failed shards remain in `analysis/report.json` and are never silently included as
complete comparisons. No cross-task score average is produced.

The source is a detached snapshot of the existing HEAD plus only the sweep
implementation. It excludes ongoing PrefixKV evidence ablations and user paper
edits. It remains dirty and explicitly noncanonical until a commit is authorized.
Dirty HF source reuse is opt-in, preflight-only, and requires the identical source
fingerprint/file inventory; the canonical campaign guard is unchanged.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/prepare_vllm_baseline_sweep.py \
  --root /scratch/djy8hg/runs/mosaickv/vllm-baseline-sweep-NEW \
  --parent-root /scratch/djy8hg/runs/mosaickv/hf-byte-matched-2c7a266-256-v1
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/prepare_vllm_baseline_sweep.py \
  --root /scratch/djy8hg/runs/mosaickv/vllm-baseline-sweep-NEW --submit
```
