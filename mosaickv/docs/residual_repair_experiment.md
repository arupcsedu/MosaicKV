# Experiment 4: detailed residual repair

## Scope

This bounded experiment evaluates real decode-time residual repair for
`Qwen/Qwen2.5-VL-3B-Instruct` under the Hugging Face eager runtime. It is an
exploratory implementation validation, not a paper-scale estimate. The immutable
study definition is
`configs/experiments/repair_detailed_small.yaml`.

The fixed examples are the versioned Experiment-3 ledgers: two MMStar ordinary
image examples, two DocVQA OCR/document examples, two ChartQA examples, and one
Video-MME example. Every policy receives the same sample IDs, prompt/media path,
processor, model revision, BF16 precision, deterministic greedy generation,
eight-token output cap, retention ratio 0.2, block size 16, prototype geometry,
mandatory-token policy, eager attention backend, and physical packed-prefill KV
byte budget.

## Bounded sweep

Each task is one Slurm array element. It loads the model once and runs 15 cells:

- FullKV and no-repair references;
- entropy, prototype-risk, combined-risk, and evaluation-only oracle policies;
- combined-risk aggressive, middle, and conservative threshold pairs;
- maximum restoration counts `R = 1, 2, 4`;
- lossless, FP16, BF16, FP8, and INT8 residual representations; and
- pinned versus pageable CPU residual placement.

Except for the FullKV reference, this is a one-variable-at-a-time sweep around
the combined-risk middle configuration. Thresholds are log-scale exploratory
values and are not fitted to task labels. A paper-scale run must choose them on
a disjoint calibration split.

## Runtime evidence

Every decode step records its trigger status and reason, entropy, prototype
risk, provisional/final token IDs, maximum absolute logit change, transfer and
re-decode latency, restored bytes, and budget state. Triggered events additionally
record:

- triggering prototype ID, layer, KV head, anchor, attention mass, dispersion,
  risk, modality composition, and logical span;
- each restored block's modality, original logical positions, media/page/frame
  metadata, dtype placement, and bytes; and
- evicted and superseded prototype IDs.

The trace records both requested and actual residual placement. A pageable row
fails aggregation if its payload is silently pinned, and vice versa. Active KV
bytes must match across every compressed policy for a paired sample.

## Metrics

The analysis uses paired sample-policy cells. Repair precision is the number of
beneficial triggered cells divided by triggered cells. A benefit either recovers
the paired FullKV answer or strictly improves the official task score. Repairable
failures are no-repair answers that differ from FullKV and that the
evaluation-only oracle can recover or improve; repair recall is the fraction of
those failures repaired by a policy.

Repair latency is synchronized residual-transfer plus re-decode time. The report
also gives p95 repair latency, restored bytes, paired task-score gain, and paired
end-to-end latency cost. The oracle's independent FullKV diagnostic trajectory is
excluded from online latency.

## Commands

All commands must use the common environment and scratch-rooted caches:

```bash
source mosaickv/scripts/cache_env.sh
PYTHONPATH=mosaickv/src /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/prepare_repair_experiment.py \
  --output-directory /scratch/djy8hg/runs/mosaickv/repair-experiment-matrix-v3 \
  --run-output-root /scratch/djy8hg/runs/mosaickv/repair-experiment-v3

sbatch \
  --export=ALL,MOSAICKV_REPAIR_INDEX=/scratch/djy8hg/runs/mosaickv/repair-experiment-matrix-v3/jobs.jsonl,MOSAICKV_ALLOW_EXPLORATORY_DIRTY=1 \
  mosaickv/slurm/repair_experiment.sbatch

PYTHONPATH=mosaickv/src /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/analyze_repair_experiment.py \
  --index /scratch/djy8hg/runs/mosaickv/repair-experiment-matrix-v3/jobs.jsonl
```

The dirty-tree override is permitted only for explicitly labeled exploratory
runs. Canonical paper rows require a clean worktree and a new immutable matrix.

## Completed exploratory run

Stable matrix v2 completed on Slurm jobs `17788043`, `17788044`, `17788172`,
and `17788024`; all exited zero. The critical path was 1:14:19. The matrix has
60 immutable configuration cells, 105 completed paired result rows, 686
decode-step records, and no failed samples. All 60 manifests agree on Git
`9e8b77c0cfe9af62d6a1a6ce5840a01543f17b72` and dirty-patch SHA
`41fa7a6d2cfdf2b68ac621c60df04f9c2460858055ef204df7f13e5fffd36d4f`.

The earlier v1 shakedown observed source-patch drift while jobs were active and
is excluded from all derived results. Its files remain under the scratch run
root for auditability, but the checked-in analysis reads only the immutable v2
index SHA `7ba195d515ad1651a053ee4aff63a708d715aac72984fa8642f7fe69a271be3b`.

The measured result is deliberately mixed: entropy-only and evaluation-only
oracle repair each improve the benchmark score on the sole oracle-repairable
compressed failure, whereas the primary combined policy does not. The
primary combined policy changes that ChartQA answer to another wrong number.
The conservative combined threshold improves the ChartQA case but damages a
DocVQA answer, leaving zero net score gain. See the paired
[analysis](../analysis/residual_repair.md) and the non-cherry-picked
[qualitative appendix](../analysis/residual_repair_qualitative.md).

## Quality-fix follow-up

The first sweep exposed a correctness bug in partial restoration: promoting a
strict subset of one prototype's members removed the entire parent prototype,
so every unpromoted member in that cluster disappeared from the active cache.
It also ranked members by prototype weight, favoring evidence already closest
to the prototype, and allowed prototype risk to mutate a very high-confidence
token. The follow-up implements three isolated corrections:

- algebraic partial-prototype downdates that preserve unpromoted members;
- weighted K/V reconstruction-error member ranking with cross-cluster coverage;
  and
- a minimum-entropy guard on the prototype-risk trigger.

The immutable follow-up definition is
`configs/experiments/repair_quality_fix_small.yaml`. Slurm array `17802244`
and children `17802245`--`17802247` completed all 44 cells with exit code zero;
the critical path was 54:02. The resulting 77 paired rows and 490 decode-step
records have no failed samples, invalid timings, budget violations, or missing
trigger provenance. All 44 manifests agree on Git
`9e8b77c0cfe9af62d6a1a6ce5840a01543f17b72` and dirty-patch SHA
`15c30f973f9196dac44b0bb9c7257e21b149b6799999490ef432d724710253c2`.

The guarded prototype-risk policy is the best measured online operating point
in this bounded run. It repaired the one oracle-repairable compressed failure,
raised the aggregate score sum from 4/7 to 5/7, caused no incorrect answer
changes, used 16 repair events, and added 3.983 seconds paired mean end-to-end
latency. The guarded combined policy also reached 5/7 and no longer damaged the
DocVQA case, but used 25 events and added 10.748 seconds.

Every tested restoration count (`R=1,2,4`) recovered the failure. Increasing
`R` raised mean restored bytes per event from 6.75 KiB to 27.52 KiB without a
further score gain; `R=1` also had the lowest paired cost among those combined
variants at 9.685 seconds. FP8 and INT8 retained the quality recovery and
reduced mean residual storage from 53.60 MiB to 26.80 MiB, but remained slower
than the lossless row. The transferred blocks are only tens of KiB, so the
extra conversion and full step re-decode dominate the saved transfer time.
Compressed residual dtypes are therefore a CPU-capacity feature, not a latency
claim, under this transfer granularity.

The complete [quality-fix analysis](../analysis/residual_repair_quality_fix.md),
[qualitative appendix](../analysis/residual_repair_quality_fix_qualitative.md),
and machine-readable results under `results/repair_quality_fix/` retain all
favorable, neutral, and failure cases. This run is still exploratory dirty-tree
evidence; a paper claim requires a clean commit and a disjoint calibration and
evaluation study.
