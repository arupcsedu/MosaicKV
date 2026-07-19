# Unified evaluation harness

## Status and boundary

The harness is implemented and CPU-tested. It does not implement MosaicKV, load a public
dataset, or establish benchmark support for any model/backend pair. Public task execution
remains **unverified** until the selected locked environment, model adapter, dataset revision,
and GPU smoke command pass.

Public benchmark prompt construction and scoring stay with `lmms-eval==0.7.2`. MosaicKV adds
only deterministic development-subset selection, a standardized local-model request boundary,
per-sample systems/fidelity telemetry, append-only storage, and provenance linkage. The adapter
uses `lmms_eval.api.model.lmms` and passes an instantiated adapter to
`lmms_eval.evaluator.simple_evaluate`; it consumes the six-argument `generate_until` request
contract used by lmms-eval 0.7.2:

```text
(context, generation_kwargs, doc_to_visual, doc_id, task, split)
```

No local fallback scorer exists for a public benchmark. If lmms-eval is absent or returns an
unrecognized per-sample metric, the sample is recorded as failed.

## Task registry

| MosaicKV task | Development lmms task | Dataset | Per-sample metric | Media | Scoring owner |
|---|---|---|---|---|---|
| `mmstar` | `mmstar` | `Lin-Chen/MMStar` | `average` | image | lmms-eval |
| `mmvet` | `mmvet` | `lmms-lab/MMVet` | `gpt_eval_score` | image | lmms-eval external judge |
| `textvqa` | `textvqa_val_lite` | lmms-eval lite task | `exact_match` | image | lmms-eval |
| `docvqa` | `docvqa_val_lite` | lmms-eval lite task | `anls` | image | lmms-eval |
| `chartqa` | `chartqa_lite` | lmms-eval lite task | `relaxed_overall` | image | lmms-eval |
| `video_mme` | `videomme` | `lmms-lab/Video-MME` | `videomme_perception_score` | video | lmms-eval |
| `synthetic_ci` | not applicable | packaged local fixture | `exact_match` | synthetic RGB tuple | MosaicKV CI only |

MM-Vet 0.7.2 calls an external LLM judge. Although its requested temperature is zero, an
external service is not guaranteed to be bitwise deterministic. The registry therefore marks
that scorer non-deterministic; MM-Vet rows must record the judge identity and must not be used as
canonical evidence until a reproducible judge policy is preregistered.

Video-MME is rejected before execution unless the local model declares `supports_video=True`.
That declaration is only a routing precondition, not evidence of working video support.

## Deterministic subsets and messages

Selection sorts stable sample IDs by
`SHA256("mosaickv-subset-v1\\0" + seed + "\\0" + sample_id)` and takes the requested count.
Selection is independent of source iteration order. The chosen source ID is retained in an
`_mosaickv_sample_id` dataset column before lmms-eval reindexes the subset.

Every local model receives a tuple of role-tagged messages. The user message contains media parts
in dataset order followed by one text part. Media payloads are passed through unchanged so the
same backend preprocessing can be used for full-cache, MosaicKV, ablations, and baselines.

## Local model integration

A local full-cache or MosaicKV implementation implements
`mosaickv.evaluation.model.LocalEvaluationModel` and returns `ModelGeneration`. Unavailable
telemetry must remain `None`; it must never be estimated or filled with synthetic values.

```python
from mosaickv.evaluation.lmms_adapter import run_lmms_development_evaluation

summary = run_lmms_development_evaluation(
    run_id="immutable-run-id",
    task_names=("mmstar", "textvqa"),
    model=local_model,
    raw_output="runs/immutable-run-id/raw.jsonl",
    parquet_output="runs/immutable-run-id/samples.parquet",
    manifest_path="runs/immutable-run-id/manifest.json",
    seed=17,
    subset_size=32,
)
```

The function loads the registered lmms-eval task objects, replaces each evaluation split with
the seeded subset, calls `simple_evaluate`, and joins lmms-eval’s per-sample score with the
MosaicKV telemetry side channel. Model-generation failures receive an empty response only so
lmms-eval can finish the batch; the raw MosaicKV row remains `status="failed"` with a null task
score and an error. If lmms-eval itself fails, every selected pending sample is materialized as a
failure row.

## Result and storage contract

Each JSONL row includes `run_id`, `task`, `status`, and `error` in addition to every field requested
for the research result schema. Timing values are seconds and memory/cache values are bytes.
`reference` is plain text for one answer or a canonical JSON array encoded as text for multiple
answers. Missing backend observations are JSON nulls.

`JsonlResultStore` is append-only and lock-protected. Resume keys are `(run_id, sample_id)`.
Re-appending an identical row is a no-op; a conflicting row raises `ResultConflictError`.
`merge_jsonl` applies the same rule, so a merged file has at most one row for each composite key.

`write_parquet_aggregate` writes a Zstandard-compressed, deduplicated Parquet materialization of
the per-sample rows. “Aggregate” means a merged analysis dataset, not an averaged results table;
statistical reduction must remain a separate, versioned transformation with run/sample lineage.
Parquet requires `pyarrow` from the locked HF evaluation environment.

## CPU synthetic check

The packaged synthetic task uses four one-pixel RGB fixtures and a deterministic weight-free
model. It validates orchestration only and is never a measured benchmark result.

```bash
cd /scratch/djy8hg/workdir/MosaicKV
/scratch/djy8hg/env/mosaickv_dev/bin/mosaickv evaluate \
  --task synthetic_ci \
  --run-id synthetic-ci-001 \
  --raw-output /scratch/djy8hg/results/mosaickv/synthetic-ci-001/raw.jsonl \
  --manifest /scratch/djy8hg/results/mosaickv/synthetic-ci-001/manifest.json \
  --seed 17 \
  --json
```

To also create Parquet, execute the same command in the explicitly created HF evaluation
environment and add:

```text
--parquet-output /scratch/djy8hg/results/mosaickv/synthetic-ci-001/samples.parquet
```

List registered tasks without loading datasets or models:

```bash
mosaickv evaluate --list-tasks --json
```
