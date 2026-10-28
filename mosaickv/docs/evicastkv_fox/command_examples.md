# Executable campaign commands

Run from the isolated campaign worktree, not the dirty main checkout. All caches,
logs and outputs stay under `/scratch/djy8hg`; no environment is installed.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY=/scratch/djy8hg/env/mosaickv/bin/python
RUN=/scratch/djy8hg/runs/evicastkv/fox-review-20261028-v4

$PY -m pytest mosaickv/tests/unit/test_fox_campaign.py -q \
  -o cache_dir=/scratch/djy8hg/cache/mosaickv/pytest/fox-v1
$PY mosaickv/scripts/evicastkv_campaign.py prepare --root "$RUN"
$PY mosaickv/scripts/evicastkv_campaign.py dry-run --root "$RUN" --stage smoke
$PY mosaickv/scripts/evicastkv_campaign.py validate --root "$RUN" --stage smoke
$PY mosaickv/scripts/evicastkv_campaign.py pilot --root "$RUN" --stage smoke
```

Preparation is write-once. The pilot submits two one-A100 workers and a finite
`afterany` CPU completion job. The callback checks denominators, errors, hashes,
parity and physical bytes. It submits the registered next stage only on a pass;
otherwise it writes a blocking decision. It never continuously polls logs and
never treats a failed Slurm dependency as a successful scientific gate.

The user's current explicit instruction supersedes the prompt package's default
manual stage-by-stage launch rule, **not** correctness or freshness requirements.
No push is authorized. The GPU ceiling is four in aggregate for this campaign,
one GPU/12 CPUs/64 GiB per worker, two-hour limit, `bii-gpu`, `bi_fox_dgx`,
account `bii_dsc_community`. Current user jobs are checked before admission.

Additional implemented command modes:

```bash
$PY mosaickv/scripts/evicastkv_campaign.py development --root "$RUN" --stage scoring
$PY mosaickv/scripts/evicastkv_campaign.py analyze --root "$RUN" --stage scoring
$PY mosaickv/scripts/evicastkv_campaign.py confirmation --root "$RUN"
$PY mosaickv/scripts/evicastkv_campaign.py resume --root "$RUN"
```

Do not manually run `analyze` while its dependent analysis job is still pending;
derived outputs are write-once. `confirmation` refuses execution without the
required independent evidence. `resume` deliberately refuses blind resubmission;
selective retry needs a separately bound retry specification, not output mutation.

## Measurement meaning

The version-4 E2E outer wall clock includes media loading, runtime input preparation,
prefill, compression, decode and output detokenization, ending at an explicit
output-completion callback. Model loading, scoring and post-output trace publication
are outside it. Full return wall time and post-output instrumentation time are
recorded separately. Earlier attempts included trace publication and must not be
pooled as the same timing definition. The runtime's narrower model measurement is
retained separately. Process-memory samples cover the whole instrumented request,
including post-output work, and are labeled separately from live-KV bytes.
Token-observer timestamps measure a real local CPU consumer handoff, not network
streaming latency. First-logit readiness is separately recorded. CUDA work is
synchronized at declared request/phase boundaries; no CUDA_LAUNCH_BLOCKING is set.

GPU allocator peaks are not process HBM. GPU telemetry and visibility limitations
must be reported separately. Five warmups and three counterbalanced repeats are
required for registered timing rows; one quality execution is not a repeated
latency benchmark. Diagnostic clones never enter timed workers.
