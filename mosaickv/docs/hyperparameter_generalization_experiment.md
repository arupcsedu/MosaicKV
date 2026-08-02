# Experiment 5: bounded hyperparameter generalization

## Scientific question

This experiment asks whether one architecture-neutral MosaicKV configuration,
selected on a disjoint calibration slice, transfers across model scale and
model family without retuning. It separately asks which backends can execute an
equivalent implementation. Backend availability is not treated as evidence for
or against algorithmic generalization.

The versioned definition is
`configs/experiments/hyperparameter_generalization_small.yaml`. This is a
tiny-sample implementation study bounded to 3.5 hours, not a paper-scale
generalization estimate.

## Calibration and freeze boundary

Calibration uses two versioned DocVQA examples with
`Qwen/Qwen2.5-VL-3B-Instruct`. Three candidates vary only forecast mode and
draft length. Retention ratio 0.2, block size 16, graph size, utility,
selection, prototype geometry, repair policy, precision, decoding, and output
length remain fixed.

The winner rule is declared before execution:

1. maximize mean official task score;
2. maximize mean FullKV token agreement;
3. maximize mean selected-block recall against the evaluation-only oracle;
4. minimize mean forecast regret;
5. minimize mean end-to-end latency; and
6. use candidate ID as a deterministic final tie-break.

`freeze_generalization_hyperparameters.py` refuses incomplete rows or
diagnostics, verifies one calibration Git/patch provenance, and writes an
immutable frozen YAML. Evaluation cannot be materialized without that file.
The evaluation preparer verifies the frozen SHA and rejects overlap between
every calibration and held-out ledger.

## Frozen evaluation

The architecture-neutral row is `mosaickv_exact` for Qwen2.5-VL-3B,
Qwen2.5-VL-7B, and LLaVA-1.5-7B on the same two MMStar, two DocVQA, and two
ChartQA examples. Every model-task pair has its own FullKV reference. Quality
loss is paired within model and task; incompatible benchmark metrics are never
macro-averaged.

Qwen additionally receives an `architecture_adapted_full` row using the same
frozen core hyperparameters plus its already audited M-RoPE prototype transform
and guarded repair policy. These rows measure architecture adaptation and are
not pooled with uniform exact-selection generalization. LLaVA remains
exact-only because its post-RoPE keys lack an audited inverse/anchor transform.

## Backend boundary

One held-out Qwen3/MMStar example tests pure FullKV eager, SDPA, and
FlashAttention-2. Compressed rows use eager prefill, because scoring and query
capture require it, followed by eager, SDPA, or FlashAttention-2 decode. The
prefill/decode distinction is recorded in every row.

At the time of immutable v1, the installed vLLM and SGLang versions had only
separately verified FullKV parity artifacts, so no native compressed row was
generated. That historical exclusion remains correct. Later A100 jobs
`17869843` and `17871006` passed version-pinned native whole-block transport
transactions for vLLM 0.10.2 and SGLang 0.5.1.post3. Those transport gates do
not retroactively authorize Experiment-5 rows: the common forecast/graph/utility
selector plan still requires a serving-engine equivalence gate.

The optional LLaVA-OneVision-7B checkpoint is unsupported in this bounded run:
it is not cached and its exact 7B adapter/checkpoint gate is absent. No download
is attempted.

This statement describes the immutable v1 run. It was resolved afterward for
future matrices: the checkpoint was cached under the scratch cache, registered
at revision `0d50680527681998e456c7b78950205bedd8a068`, and exploratory BF16
A100 job `17856497` passed FullKV/reinjection parity, retention-one identity,
real prototypes, and residual repair. The historical v1 results and their
unsupported-combination table are not retroactively rewritten.

## Commands

All commands use the common environment and scratch-rooted caches:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH=/scratch/djy8hg/workdir/MosaicKV/mosaickv/src

/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/prepare_generalization_experiment.py calibration \
  --spec mosaickv/configs/experiments/hyperparameter_generalization_small.yaml \
  --destination /scratch/djy8hg/runs/mosaickv/generalization-calibration-matrix-v1 \
  --output-root /scratch/djy8hg/runs/mosaickv/generalization-calibration-v1

sbatch --export=ALL,MOSAICKV_GENERALIZATION_INDEX=/scratch/djy8hg/runs/mosaickv/generalization-calibration-matrix-v1/jobs.jsonl,MOSAICKV_ALLOW_EXPLORATORY_DIRTY=1 \
  mosaickv/slurm/hyperparameter_generalization.sbatch
```

Only after calibration completes, freeze and materialize evaluation. The
checked-in analysis records the exact commands, immutable index SHA, frozen
SHA, Slurm jobs, failures, and measured outputs.

## Executed exploratory run

Calibration Slurm job `17844008` completed in 2m55s and selected `hybrid4`.
The read-only frozen artifact has SHA-256
`53e1eeebbfd08b9c90b2100878590c39c4fe17bffb570d765f853d6dfb60c598`.
Held-out array `17844973` completed all nine tasks with exit code zero; the
longest task was Qwen2.5-VL-7B at 13m01s. The evaluation index SHA-256 is
`f6a8a5110fcc2a1138fbf6ce5d8f4100bba77421d6b72c207b5fb9055bbec4bb`.
Supplemental FullKV same-prefix logit job `17848538` completed in 23 seconds.
Across all vocabulary logits for eight steps, SDPA versus eager had maximum
absolute deviation 0.921875, mean absolute deviation 0.105575, and mean KL
0.000181; FlashAttention-2 versus eager had 0.843750, 0.112691, and 0.000441,
respectively. Both retained 100% same-prefix token agreement. Compressed-path
logit deviation was not captured and remains explicitly unavailable.

The run produced 54 completed sample rows, 24 model/task/role aggregates,
nine backend-support rows, and six explicit unsupported combinations. Eight
of nine architecture-neutral exact-selection cells had zero task-local score
loss while reducing active KV memory by 80.3%--81.6%; Qwen2.5-VL-3B ChartQA
was the exception with a 0.5 loss on two examples. LLaVA DocVQA and ChartQA
parity is floor-limited because FullKV also scored zero. These tiny dirty-tree
results validate protocol plumbing but are not paper-scale evidence.

Generated artifacts are:

- `results/generalization/sample_results.parquet`
- `results/generalization/model_task_generalization.parquet`
- `results/generalization/backend_parity.parquet`
- `results/generalization/unsupported_combinations.parquet`
- `tables/generalization_by_model_task.tex`
- `tables/backend_parity.tex`
- `figures/quality_loss_model_family.pdf`
- `analysis/hyperparameter_generalization.md`
- `analysis/architecture_specific_adaptations.md`
