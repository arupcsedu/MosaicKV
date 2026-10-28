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

## CPU-only statistical reproduction

The sparse analysis implementation uses the installed official PTB tokenizer
and ngram extraction. It caches ngram counts, **not fixed-DF CIDEr scores**.
Each of the 10,000 resamples recomputes reference document frequencies, TF-IDF
norms, clipping and the Gaussian length penalty, with duplicate image occurrences
counted separately. The official scorer remains the point-estimate authority.
Every actual cohort is checked against that scorer on the complete original
corpus and eight paired resamples before the sparse bootstrap is accepted
(absolute tolerance 1e-10 CIDEr points). This analysis tolerance does not alter
any model, logit, cache, quality or memory gate. No package is installed.

```bash
$PY mosaickv/scripts/evicastkv_statistics_recheck.py reproduce \
  --root /scratch/djy8hg/runs/evicastkv/fox-review-20261028-v6-coco-scoped \
  --output NEW_EMPTY_DERIVED_DIRECTORY
```

This executes two independent regenerations from saved rows and checks exact
derived hashes plus agreement with any existing official-reducer statistics.
Canonical predictions/statistics remain untouched. Add `--include-packets`
only after all producer jobs have terminated, so both report snapshots use the
same immutable set of raw rows. The finite `follow` callback schedules itself
only after already registered completion jobs; it never polls or starts GPU work.

The mechanism study is sharded into two development inputs per worker (32
workers for 64 inputs), without changing any condition, input, coefficient,
statistical family or output policy. Its greater per-input treatment count must
not inherit the eight-input quality-validation shard size. Each worker still
requests one A100, 12 CPUs, 64 GiB and at most two hours; at most four are admitted
concurrently. This is scheduling, not a new inference batching policy.

## Full offline CPU regression audit

The broader unit suite includes fixture checks against pinned upstream source.
In a **separate audit worktree**, initialize the existing gitlinks before the
audit; this retrieves licensed source only, not packages, datasets or weights:

```bash
git submodule update --init --jobs 3 -- third_party/LOOK-M third_party/PrefixKV \
  third_party/StreamingLLM third_party/PyramidKV third_party/SnapKV third_party/ElasticCache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
$PY -m pytest mosaickv/tests/unit -q -p no:cacheprovider \
  --junitxml=/scratch/djy8hg/runs/evicastkv/NEW_AUDIT_DIRECTORY/unit_results.xml
```

Do not initialize or alter source in an actively running experiment worktree.
Historical `shared_byte_comparison_v2.yaml` retains its original source digest;
the Fox instrumented/optimized checkout is intentionally a different source.
Its unit policy checks do not authorize that old campaign. A separate regression
requires the actual preparer to reject source drift **before** catalog access or
output creation and verifies that the historical contract is not rewritten.
There is no refreshed historical seal, relaxed fingerprint, or reused paper run.

## Qualify saved first-token diagnostics separately

`evicastkv_boundary_audit.py` joins every completed A row with its hash-bound B
snapshot, rejects missing/duplicate/changed records and writes
`first_token_sensitivity.csv` plus `GATE_boundary_B.json` under a new directory.
Every B method inherits its input's independent FullKV chunking control.
Execution completion is not a numerical pass; an invalid FullKV control leaves
B blocked without rewriting the valid primary A gate. B's changed observations
and capacity boundary cannot be interpreted as a single causal factor.

```bash
$PY mosaickv/scripts/evicastkv_boundary_audit.py \
  --root /scratch/djy8hg/runs/evicastkv/fox-review-20261028-v6-coco-scoped \
  --boundary-export PREVIOUS_PACKET/boundary_A_B_diagnostics.json \
  --boundary-export-sha REVIEWED_SHA256 --output NEW_EMPTY_BOUNDARY_AUDIT_DIRECTORY
```

This command never runs GPU inference, scores an invalid B corpus, changes a
numerical threshold, or turns an A/B diagnostic into primary task evidence.
