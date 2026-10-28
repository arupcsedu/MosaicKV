# Additional official MileBench subtasks: preparation, not confirmation

The user selected additional official MileBench subtasks rather than COCO-only
confirmation. This is a **new benchmark scope**, not a repair that makes the
original four-task confirmation fresh. Keep the original ActionPrediction,
ActionSequence, CLEVR-Change and Spot-the-Diff results, exclusions, numerical
failures and registered contrast family intact.

## Proposed scope

| Official subtask | Proposed question type | Separate metric | Official category |
| --- | --- | --- | --- |
| ActionLocalization | Multiple choice | Accuracy | Realistic Temporal |
| CharacterOrder | Multiple choice | Accuracy | Realistic Temporal |
| ALFRED | Open ended | ROUGE-L F1 | Realistic Semantic |
| IEdit | Open ended | ROUGE-L F1 | Realistic Semantic |

These are existing official task/scorer routes, selected using input structure,
availability and montage compatibility, not model outcomes. The source is
`FreedomIntelligence/MileBench`, pinned to
`53c7a58051ef88bacf76541d91f03f5ba2d71e7d`. The annotation previews expose 200
IDs per proposed task. This is a source-pool count, **not a measured quality
result or a certified untouched count**. Input choice structure is checked;
the original annotation's `meta_data.question_type` remains a separate gate.
If any task fails media feasibility, do not silently replace it or weaken
freshness. Register a scope amendment before materialization/outcome access.

The proposed confirmation has 64 inputs per task (256 total), with eight
additional correctness inputs per task. These are planning counts. No sample
identities have been selected for either cohort by this tool. Underlying media
groups must be disjoint between those cohorts, across tasks and from history.

## Bounded ID/media-path audit

`evicastkv_milebench_scope.py` reads exactly these Parquet columns:

```
sample_id
task_instance.images_path
task_instance.combined_1_images
task_instance.choice_list
```

It does not decode prompts, context or reference answers, load a model, download
image archives, construct ledgers, select confirmation identities or submit
jobs. HTTP range reads can transfer adjacent encoded bytes; do not claim that
the transport cannot contain an unrequested column's bytes. The guarantee is
column-projected decoding and no use or persistence of those values. The source
revision/file path and projected-content hash are recorded. The complete remote
Parquet file is **not** claimed to have been downloaded or hashed.

All four catalogs and the availability report are write-once, bound to a clean
source SHA, protocol hash and historical inventory hash. Even an apparently
unused ID is only an **upper bound** on availability. The current inventory is
not exhaustive: unresolved historical roots and frame/video/document identity
coverage remain visible in the report. Task-specific path names or a different
montage are not proof of different underlying media.

Before calling any input fresh, classify unresolved history, verify official
question types, check original-frame content and decoded pixels, and resolve
shared video/document/source-image groups. New tasks may reuse the same source
media as old tasks. Adversarial annotations on old media are not fresh.

## Numerical and statistical boundary

The consumed-input numerical diagnostic is a separate chain. Neither this
scope nor an attention-precision diagnostic can waive its failed thresholds.
Any common arithmetic change requires new identical-policy controls for every
method/reference, followed by new quality/timing evidence. Boundary-B results
remain diagnostic unless their original numerical control passes; an additional
same-chunk identity does not replace that control.

Keep the pinned LLaVA-1.5-7B checkpoint, FP16 weights and stored K/V, eager HF,
64 emitted decisions, one-token blocks, mandatory first/last positions, exact
selection, no online draft/prototypes/repair, and 16/40 MiB initial caps. FullKV
is an uncompressed quality reference. The remaining methods are Prompt Top-K,
`lookm_reimpl` (never official LOOK-M), and `mosaickv_exact`/EviCastKV. Attention
arithmetic and a reviewed selected configuration are unresolved and block
confirmation. No old/new-arithmetic timings may be pooled.

Accuracy and ROUGE-L must remain separate. Bootstrap paired source-media
clusters and recompute equal-subtask means within each metric family. Report
the four-task mix explicitly; it is not the full MileBench aggregate. A new
scope-specific test family must be frozen before scoring. Preserve the original
six-contrast family and its unexecuted original MileBench tests; do not replace
missing tests with the new subtasks after seeing outcomes.

## Commands

From the clean scope-preparation worktree:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_milebench_scope.py \
  --protocol "$PWD/mosaickv/configs/experiments/evicastkv_fox/milebench_additional_subtasks_v1.json" \
  --history-inventory /scratch/djy8hg/runs/evicastkv/fox-review-20261028-numerical-recovery-v1/history-inventory.json \
  --output /scratch/djy8hg/runs/evicastkv/fox-review-20261028-milebench-additional-v1
```

This command has no experiment launch or downstream authorization option. All
catalog/media verification, numerical, configuration-freeze and preregistration
gates must pass before a separate confirmation execution can be enabled.
