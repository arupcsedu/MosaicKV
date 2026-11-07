# Modest memory reductions: staged validation

This is a new budget-calibration experiment, not a reinterpretation of the
negative retrospective COCO analysis or the failed MileBench gate. It changes
no selector, model, prompt, media processing, precision, attention backend,
mandatory policy, or 64-token generation control.

The versioned registration is
[`modest_memory_v1.yaml`](../configs/experiments/modest_memory_v1.yaml).
The running campaign uses a separately frozen JSON copy, not a mutable view of
that YAML file.

## Stage 1: identity-only availability audit

The audit projects only `question_id` from the 14 pinned COCO validation Arrow
streams. Images and caption annotations are not decoded during cataloging.
It inventories top-level/two-level project plans, materialization/ID manifests
and their transitive ledger references. It does not traverse scheduler logs or
per-attempt result trees, nor use answers/scores to choose identities. The
90-second scan budget fails closed rather than waiting for hours.

The completed audit found:

| Pool | Total | Prior materialized/reserved identities excluded | Available |
|---|---:|---:|---:|
| COCO Caption validation | 40,504 | 4,096 | 36,408 |
| MileBench ActionPrediction | 200 | 128 | 72 |
| MileBench ActionSequence | 200 | 200 | 0 |
| MileBench CLEVR-Change | 200 | 200 | 0 |
| MileBench Spot-the-Diff | 200 | 200 | 0 |

This is an audit of the specified project metadata, not a claim about experiments
outside that scope. Reserved/unexecuted materialized identities are also
excluded, so “excluded” does not mean every identity was evaluated. The audit
read 1,191 metadata files in approximately 34 seconds.

Deterministic hash ranking reserves two nonoverlapping COCO cohorts: 128
calibration identities and 512 confirmation identities. Both exclude all 4,096
prior identities. Only calibration images/references are materialized now.
The confirmation reservation remains ID-only.

MileBench execution is blocked: there are no unreserved examples in either
registered ROUGE-L subtask. Additional compatible subtasks would need a
separate preregistration, pinned media/annotation provenance and both metric
families. No old examples are relabeled untouched, and its failed gate remains
failed.

## Stage 2: COCO calibration

- Model: `llava-hf/llava-v1.6-mistral-7b-hf` at
  `2424fdd47412fccc66d91719126b420e9fbd7065`.
- Dataset: `lmms-lab/COCO-Caption`, validation revision
  `c3213b54e173ae86517d481de17f31439d6f8c72`.
- Inference source: clean existing worktree at
  `edb6edfaf50f4539c9c1969371fefecf0eabccbf`.
- Every compressed method at every budget: Prompt Top-K, `vl_cache_reimpl`,
  `mosaickv_exact`; initial physical caps 28/32/35/40 MiB.
- FullKV once per image as a quality reference, never a compressed-method
  latency comparator. No historical predictions reused on the new identities.
- HF eager, FP16, block size one, last-token mandatory policy, deterministic
  64-token generation, five warmups, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before
  Python starts. Original K/V, logical positions and append-only growth remain
  unchanged; prototypes/residuals/repair remain disabled.
- 13 arms x 128 images = 1,664 planned prediction rows, in 104 one-method,
  16-image shards. Every worker has one A100, 12 CPUs and a two-hour limit.
- Four A100s are admitted in aggregate, below the user's six-A100 ceiling.
  The unchanged frozen worker's contract requires four; it is not silently
  edited to increase admission. The shared controller counts other active jobs
  owned by this user and preserves running jobs.
- Per-sample completion markers support selective resume. Failures remain
  explicit. GPU analysis depends on no CPU preparation job; preparation is done.

All 104 worker specifications passed CPU verification before submission.
Per-worker GPU ordinary-generation and retention-one parity checks remain
mandatory at execution. “CPU verified” does not mean GPU/quality gates passed.

### Registered selection and stopping rules

1. Compute FullKV's complete calibration corpus CIDEr. That absolute value is
   the quality target, chosen without access to confirmation outcomes.
2. Across both baseline methods, choose the smallest measured baseline cache
   meeting that target. Equal-budget ties prefer Prompt Top-K, then
   `vl_cache_reimpl`; this deterministic rule is registered before inference.
3. Among strictly smaller MosaicKV caps, select the largest whose calibration
   CIDEr is at least the selected competitor's. Every measured memory boundary
   must also be strictly smaller on every paired sample.
4. Stop if no baseline or no smaller candidate qualifies. Do not choose a larger,
   weaker competitor simply to produce a positive comparison. Do not change
   the algorithm using this cohort's labels.
5. Estimate confirmation size from paired complete-corpus bootstrap variance,
   observed positive difference, 80% planned power and corrected alpha. This is
   an approximation; selecting a calibration winner can overestimate its effect.
   If more than the reserved 512 identities are required, stop for another
   ID-only availability/reservation step rather than underpowering the test.
6. Create `analysis/calibration/frozen_candidate.yaml` only when execution,
   physical storage, preliminary quality and 512-sample power readiness pass.
   A freeze is not a paper-quality pass. No confirmation jobs are submitted by
   the calibration analyzer.

The CPU analyzer recomputes complete corpus CIDEr in all 10,000 paired bootstrap
draws, including repeated-identity document frequencies. It validates the
accelerated computation against the unchanged pinned official evaluator.

## Stage 3: independent confirmation, still blocked

Before any confirmation inference, freeze the candidate configuration, selected
competitor, absolute quality target, all neighboring/cheaper baseline points,
sample size, multiplicity family, timing subset/order, and stopping rule.

The required confirmation gates remain:

- strictly lower measured initial/final/peak KV backing storage;
- multiplicity-adjusted paired CIDEr lower bound at least zero, with no margin;
- paired end-to-end ratio upper bound at most one;
- selected baseline meets the frozen quality target and no cheaper measured
  baseline qualifies;
- complete rows/provenance, parity, no FullKV fallback, no sample dropping.

Use five warmups and 20 timed repetitions on a fixed 16-image timing subset,
with randomized method order and consistent GPU controls. These repeated timing
trials are **not** replaced by the calibration's one quality trial per image.
No optional stopping or post-outcome sample-size adjustment is permitted.

## Current artifacts and commands

Root:
`/scratch/djy8hg/runs/mosaickv/modest-memory-v1-20261022/`.

- `availability-audit.json`: write-once audit and separate ID reservations.
- `coco-calibration/`: preserved preparation failure, before inference, caused
  by tuple-versus-JSON-array comparison. The corrected check compares normalized
  JSON representations; no configuration control was changed.
- `coco-calibration-retry1/plan.json`: immutable 104-worker plan;
  SHA-256 `fb1752fa1c7f34bdfad9fbfdb0c1dc2f7d060e673491989b9e16c2886494d994`.
- `coco-calibration-retry1/protocol.json`, `contract.yaml`, `configs/`, `ledgers/`:
  frozen controls and inputs.
- `tooling/`: hash-bound CPU analysis and Slurm launch snapshots.
- `outputs/`, `logs/`: append-only predictions, traces, manifests and failures.
- `scheduler/`: admission state, separate from immutable scientific controls.
- `analysis/calibration/report.json`: dependent analysis once workers finish.

Submitted jobs:

| Job | Purpose | Dependency |
|---|---|---|
| `21831484_0–103` | 128-image, 13-arm COCO calibration | Materialization and CPU verification already complete |
| `21831997` | Corpus metrics, power estimate and conditional freeze | `afterany:21831484`; incomplete rows block freezing |
| `21831486` | Shared four-A100 admission controller | Running on `bii` |

The initial controller tick rejected the launcher's pre-existing user holds.
The correction explicitly adopts **only this newly submitted array**; the
controller is now running and two GPU workers were observed started. Two more
were released but waiting for resources. The immutable scientific plan/hash
did not change. `submission.json` preserves the initial incident and
`submission-recovery.json` records the recovery; it is not hidden by rewriting
the old submission record.

An exact-runtime CPU import smoke then caught two analysis-only compatibility
issues: the old inference package has no `Point` typing alias, and its collector
accepts three positional arguments rather than later subset keywords. The
correction changes neither metric arithmetic nor inference. The old pending CPU
analyzer `21831485` was canceled and replaced by `21831997`, using a new immutable
`analysis-tooling-v2/` bundle. GPU jobs, source, plan and samples are untouched;
the original tooling bundle is preserved. `analysis-replacement.json` records
this correction. Frozen-runtime imports and collector invocation now pass.

Verification: 44 focused planning/scheduler CPU tests and scoped Ruff checks
pass. GPU parity and empirical gates are not yet claimed passed.

CPU planning/tests use the common environment and scratch caches:

```bash
source mosaickv/scripts/cache_env.sh
/scratch/djy8hg/env/mosaickv/bin/python -m pytest \
  mosaickv/tests/unit/test_modest_memory.py -q
```

Submission requires importing the unchanged frozen inference package, not the
dirty main checkout:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH=/sfs/weka/scratch/djy8hg/worktrees/mosaickv-coco-shared-retry1-20261014/mosaickv/src:/sfs/weka/scratch/djy8hg/worktrees/mosaickv-coco-shared-retry1-20261014/mosaickv/scripts
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/submit_modest_memory.py \
  --root /scratch/djy8hg/runs/mosaickv/modest-memory-v1-20261022/coco-calibration-retry1
```

Without `--submit`, the launcher is read-only verification. A second submission
to the same root is refused. The current array is already submitted; do not
submit another full copy. Retry only compatible missing/failed workers.

Planning estimate, not a measurement: approximately 3–5 hours for calibration
with four continuously allocated A100s, excluding queue time. Confirmation time
and feasibility depend on the calibration result and power calculation.
