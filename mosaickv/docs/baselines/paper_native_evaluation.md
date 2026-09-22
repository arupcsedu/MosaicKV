# Paper-native baseline evaluation

## Purpose

The existing MosaicKV campaign measures MMStar accuracy, TextVQA VQA score,
DocVQA ANLS, active KV bytes, and latency for FullKV, Prompt Top-K, and
MosaicKV. Those rows remain valid and unchanged. They answer the common-budget
question, but they do not establish that the local PrefixKV, VL-Cache, or
LOOK-M reimplementations reproduce the experiments in their papers.

This document defines three separate reproduction tracks. Their metrics are
not combined into one macro score. A method may enter MosaicKV's primary table
only after its native track passes; otherwise it is labeled a local
reimplementation diagnostic.

The machine-readable contract is
[`published_paper_native_metrics_v1.yaml`](../../configs/experiments/published_paper_native_metrics_v1.yaml).
Execution is authorized for its enabled methods. The common-runtime
PrefixKV/LOOK-M reimplementations use the audited HF conversion at revision
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`; their protocol IDs and method
labels explicitly include `reimpl`. The incompatible pinned official launchers
remain recorded but have `execution_enabled: false`. The LLaVA-NeXT checkpoint
used by VL-Cache remains pinned to
`2424fdd47412fccc66d91719126b420e9fbd7065`. Authorization is not a measured
support claim: every run must still pin its exact dataset artifact and sample
ledger, and it must retain a failure row if loading or execution fails.

## What each paper measures

| Track | Paper model and data | Native quality metrics | Paper operating points | Upstream code |
|---|---|---|---|---|
| PrefixKV | legacy LLaVA-1.5-7B; LLaVA-Description (1,000) and MM-Vet (218) | teacher-forced corpus PPL; mean ROUGE-L F1 against frozen FullKV generations | 10% through 90% retained cache plus FullKV | pinned at `597f1ab032704951550f93bcc8a23f1454b80aa4` |
| VL-Cache | LLaVA-v1.6-Mistral-7B; COCO Caption, DocVQA, MathVista | CIDEr, ANLS, and accuracy, respectively | 1%, 5%, 10%, 20%, 40%, 60%, 80%, 100% | no official release assumed; always `vl_cache_reimpl` |
| LOOK-M | legacy LLaVA-1.5-7B; MileBench | dataset Accuracy or ROUGE-L, followed by equal-dataset subtask means | 5%, 20%, 35%, 50%, 100% retained cache | pinned at `ecf0f51a9c416c2d85e47faf2638502f01a6d748` |

Primary sources: [PrefixKV](https://arxiv.org/abs/2412.03409),
[VL-Cache](https://arxiv.org/abs/2410.23317), and
[LOOK-M](https://aclanthology.org/2024.findings-emnlp.235/). The executable
details were checked against the pinned PrefixKV `eval_ppl.py` and
`eval_rouge.py`, and LOOK-M `evaluate.py` and `score.py` files.

## Two result categories

MosaicKV now treats the following as separate artifact families:

| Category | Measured origin | Purpose | Required controls |
|---|---|---|---|
| `paper_reproduction` | `locally_reproduced` | Reproduce a paper in its native regime | paper checkpoint, dataset/split, native metric, nominal budget convention, measured active bytes, processor and supported environment |
| `unified_comparison` | `locally_measured` | Compare methods in one common runtime | identical weights, processor/chat/media/prompt/tokenization, generation/output length, precision, backend, block and mandatory-token policy, environment lock, samples, and active-byte budget |

Raw execution rows, metric observations, manifests, and analyses carry both
fields. The paired analyzer requires `--result-category`, rejects mixed
cohorts, and always requires paired active-byte matching for a unified
comparison. It also validates the controlling manifests before aggregation.

The existing MMStar/TextVQA/DocVQA FullKV, Prompt Top-K, and MosaicKV cohort
remains the primary unified-comparison table. Paper-native PrefixKV, LOOK-M,
VL-Cache, and secondary-baseline runs extend the evidence in separate tables;
they neither overwrite those raw artifacts nor enter the primary cohort.

Published values use a third origin, `paper_reported`, but never enter the
measured metric stream. They live in source-hashed JSON objects. The
write-once comparison artifact nests `paper_reported`,
`locally_reproduced`, and `reproduction_delta`, so a paper number cannot be
mistaken for a MosaicKV execution. The delta is always local minus paper.

The versioned protocol binds each paper lane to a budget convention and a
supported environment, including exact model/processor revision, backend,
attention implementation, precision, active-byte accounting SHA, cache block
size, mandatory-token policy, and common environment-lock SHA. A run with a
different dataset ID/split, mutable dataset revision, or incompatible control
is rejected before model loading.

## Implemented metric path

Paper-native metrics use a separate immutable schema. A metric row records the
metric name, direction, unit, scope, implementation/version, evaluator
identity, reference-artifact hash, and manifest path. This prevents a TextVQA
score, ANLS, CIDEr, PPL, or ROUGE value from being silently stored in the same
anonymous `task_score` column.

The following components are implemented:

- `evaluation.metrics`: strict sample/corpus schemas, registry, scorers,
  append-only JSONL, Parquet output, and metric-aware paired bootstrap;
- `evaluation.prefixkv_metrics`: a teacher-forced NLL interface, immutable
  FullKV answer artifacts, token-weighted corpus `teacher_forced_ppl`, and
  per-sample ROUGE-L from PrefixKV/LOOK-M's pinned `rouge==1.0.1` dependency
  (`perplexity` remains a read-only alias for older artifacts);
- `evaluation.milebench`: all 29 datasets in LOOK-M's released taxonomy,
  dynamic dispatch from the declared question type, the two official needle
  special cases, and metric-separated equal-dataset group aggregation;
- `evaluation.tasks`: executable LLaVA-Description, paper-style MM-Vet,
  MileBench, COCO Caption, DocVQA, and MathVista routes;
- `evaluation.paper_native_lanes`: a hash-bound materialized-input runner for
  PrefixKV and MileBench, strict method/protocol guards, frozen FullKV
  reference handling, official MileBench hierarchy output, and measured
  active-byte validation;
- `evaluation.lmms_adapter`: task-owned scoring plus optional paper-native
  metric JSONL/Parquet output;
- `analyze_paper_native_metrics.py`: paired confidence intervals that rerun
  the appropriate metric reducer on each bootstrap draw.

CIDEr is corpus-valued and is recomputed from the complete prediction and
reference cohort with lmms-eval's COCO `PTBTokenizer` followed by
`pycocoevalcap.Cider`; it is not averaged from per-example CIDEr numbers. ANLS
delegates to lmms-eval's DocVQA implementation. MathVista delegates answer
extraction, normalization, and equality to lmms-eval's pinned task module;
`Levenshtein==0.27.5` is pinned explicitly because lmms-eval imports it but
does not declare it. PPL is `exp(sum NLL / sum target tokens)`, matching the pinned
PrefixKV driver, which flattens all target-token NLLs before taking the mean.
For PPL, every sample's NLL sum and token count are retained so paired
bootstrap resampling remains mathematically valid.

## Metric-aware paired statistics

Statistical comparisons are fail-closed and pair methods by the exact same
sample IDs. The default analysis uses 10,000 paired bootstrap replicates, a
fixed seed, and a 95% percentile confidence interval. Each replicate reruns
the metric's actual reducer instead of averaging already aggregated scores:

- Accuracy, ANLS, and ROUGE-L F1 are the arithmetic mean of resampled sample
  contributions.
- Teacher-forced PPL is recomputed as
  `exp(sum(resampled NLL) / sum(resampled target-token count))`; per-sample
  PPL values are never averaged.
- CIDEr is recomputed from resampled raw captions and references through the
  pinned COCO tokenizer and CIDEr reducer. Repeated bootstrap draws receive
  evaluator-local occurrence IDs, so repetitions are counted rather than
  collapsed by the evaluator dictionaries.
- MileBench is stratified by official dataset. Every replicate first rebuilds
  each dataset mean and then reruns the official equal-dataset subtask, NH,
  and IR hierarchy. Accuracy and ROUGE-L hierarchy entries remain separate.

Every result reports `candidate_minus_reference` and its paired confidence
interval. When the full-cohort reference value is nonzero, it also reports the
relative difference `(candidate - reference) / abs(reference)` and the same
paired interval expressed in units of that reference. Relative values are
recorded as `null` when the reference is zero. The sign is always candidate
minus reference; use `metric_direction` to interpret whether that sign is
beneficial because PPL is lower-is-better. No analysis combines PPL, CIDEr,
ANLS, ROUGE-L, and accuracy into one average.

MileBench sample rows carry the official dataset and declared question type in
`metadata_json`; the analyzer refuses hierarchy reconstruction when those
fields are absent. CIDEr predictions and references stay in raw result JSONL,
while the metric stream contains its corpus observation. Consequently paired
CIDEr analysis requires the corresponding raw result files explicitly rather
than duplicating captions in Parquet.

## Materialized PrefixKV and MileBench ledger

The two non-lmms routes consume one immutable JSON object. Relative media
paths are resolved against the ledger directory; each media object is hashed
before inference. Home-directory paths, broken files, mismatched hashes,
duplicate sample IDs, unknown MileBench datasets, and incomplete task-specific
metadata fail before model execution.

```json
{
  "schema_version": 1,
  "task": "llava_description",
  "dataset_id": "PrefixKV/LLaVA-Description",
  "dataset_revision": "<immutable dataset/artifact SHA>",
  "split": "official_1000",
  "samples": [{
    "sample_id": "detail-000001",
    "prompt": "Describe the image in detail.",
    "references": ["human annotation"],
    "teacher_forcing_target": "human annotation",
    "media": [{"kind": "image", "path": "images/000001.jpg", "sha256": "<sha256>"}]
  }]
}
```

MileBench replaces `teacher_forcing_target` with `milebench_dataset`,
`milebench_question_type`, and (for multi-choice samples) `choices`. The
runner writes generated predictions to raw JSONL, raw measurements to a
Parquet table, sample metric observations to one Parquet table, and corpus
aggregates to another. A completed paper-native row must contain measured
`active_kv_bytes`; nominal retention is retained only as an additional
operating-point label.

## PrefixKV reproduction rules

1. Freeze FullKV generations for the exact model, tokenizer, chat template,
   prompts, media, and generation configuration. Store and hash them with
   `FullKVReferenceArtifact`.
2. Compute ROUGE-L F1 for every compressed answer against that frozen FullKV
   answer. Human annotations are not the ROUGE reference in this protocol.
3. Separately teacher-force the dataset's human answer and record every target
   token's NLL through the same compressed cache path. Aggregate PPL over all
   target tokens, not over per-sample PPL values.
4. Generate the PrefixKV layer profile from a disjoint calibration set. Record
   both retained ratio and the upstream code's complementary forget ratio.
5. Reproduce one official PrefixKV point using the pinned legacy runtime before
   treating `prefixkv_reimpl` as paper-faithful. HF-converted LLaVA weights are
   not interchangeable with the legacy checkpoint.

The common HF runtime implements `teacher_forced_nll`: it performs the normal
prefill and cache compression once, scores the first human target token from
the prefill logits, and then feeds the remaining target tokens through the
same compressed decode state. It rejects rather than truncates a target longer
than `generation.max_new_tokens`. `evaluate_prefixkv_native_metrics` binds
those additive NLL statistics to the paired generated rows and frozen FullKV
answer artifact. vLLM and SGLang do not expose this protocol and must be marked
unsupported for PrefixKV-native PPL rather than substituting generated-answer
likelihoods.

## VL-Cache reproduction rules

Use the paper's LLaVA-v1.6-Mistral checkpoint family. Run the exact same sample
IDs for FullKV, each comparison method, and MosaicKV at every budget. Report:

- COCO Caption: corpus CIDEr in score points (`pycocoevalcap` value times 100);
- DocVQA: official lmms-eval ANLS;
- MathVista testmini: official answer extraction and accuracy in percentage
  points.

The local `vl_cache_reimpl` remains explicitly a reimplementation because no
official source is assumed. Its first validation is trend reproduction across
the paper's full retention sweep, not equality to a single published number.
The 7B LLaVA-v1.6 track must not be mixed with LLaVA-1.5 MosaicKV rows.

## LOOK-M reproduction rules

MileBench uses task-dependent metrics. Multi-choice, needle, and most
open-ended datasets use accuracy or ROUGE-L as specified by the official
evaluator. Dataset scores are averaged equally inside each subtask. The
Natural-Habitat summary is the mean of N-1 and N-2; Image Retrieval aliases
I-1. Heterogeneous Accuracy and ROUGE values are never collapsed into one
undocumented overall score.

First run the pinned official LOOK-M repository with its legacy LLaVA model,
then run `lookm_reimpl` under identical controls. Compare selected positions,
active tensor bytes, tokens, native MileBench metrics, and latency. A result
from the unified runtime is always labeled `lookm_reimpl`, never official.

## Recurring comparator status

H2O, StreamingLLM, PyramidKV, SnapKV, and RoCo are now separately implemented
under their required `*_reimpl` labels. StreamingLLM, PyramidKV, and SnapKV
are audited against pinned, licensed submodules. H2O and EasyKV/RoCo were
audited at immutable commits but their trees expose no top-level license, so
their upstream source is not redistributed. Their formulas, interpretations,
and executable boundaries are recorded in
[`published_kv_spec.md`](published_kv_spec.md). Before any of these rows can
enter a measured paper table it still needs:

1. a paper and source mapping;
2. an immutable official commit and preserved license, or a separately named
   `*_reimpl` implementation;
3. retention-one identity and exact byte-accounting tests;
4. a native-benchmark reproduction point;
5. the same model, processor, prompts, media, decoding, precision, backend,
   warmups, and active-byte budget used for MosaicKV.

The six local methods now pass structural identity, determinism, and exact
byte-accounting tests and are execution-enabled in their declared protocol
lanes. This is implementation readiness, not a measured reproduction claim.
Elastic Cache is available only as the explicitly local `elastic_cache_reimpl`,
audited against the pinned MIT source. It must not be labeled official code.

Pinned official PrefixKV and LOOK-M are invoked through the fail-closed
`run_official_baseline.py` control-plane wrapper. Their author dependency
stacks are isolated because changing those versions would invalidate the
official-code label. A local common-runtime row remains `prefixkv_reimpl` or
`lookm_reimpl`; it is never promoted to official by configuration.

## Commands

Validate the protocol without running models:

```bash
source /scratch/djy8hg/workdir/MosaicKV/mosaickv/scripts/cache_env.sh
python mosaickv/scripts/validate_published_protocols.py \
  mosaickv/configs/experiments/published_paper_native_metrics_v1.yaml \
  --require-metrics-ready
```

Validate both metric readiness and the authorized execution matrix:

```bash
python mosaickv/scripts/validate_published_protocols.py \
  mosaickv/configs/experiments/published_paper_native_metrics_v1.yaml \
  --require-metrics-ready --require-executable
```

This command validates authorization, immutable model SHAs, dependencies,
metric routes, and method inclusion. It does not claim that an external
checkpoint or dataset is already cached, nor that a GPU smoke test passed.

HF/vLLM/SGLang evaluation commands accept `--metric-output`,
`--metric-observations-parquet-output`, and
`--metric-aggregates-parquet-output`. The two Parquet destinations must be
provided together so sample observations and corpus aggregates cannot be
silently mixed. The older `--metric-parquet-output` flag remains as a combined
export for compatibility. For a paired sample-level comparison:

```bash
python mosaickv/scripts/analyze_paper_native_metrics.py \
  --metrics /scratch/djy8hg/runs/mosaickv/RUN/metrics.jsonl \
  --output /scratch/djy8hg/runs/mosaickv/RUN/mosaickv-vs-full.json \
  --task docvqa --model MODEL --backend hf --metric anls \
  --result-category paper_reproduction \
  --reference-method full_kv --candidate-method mosaickv_exact \
  --candidate-ratio 0.1
```

For corpus CIDEr, provide every raw result shard containing the paired
cohorts:

```bash
python mosaickv/scripts/analyze_paper_native_metrics.py \
  --metrics /scratch/djy8hg/runs/mosaickv/RUN/metrics.jsonl \
  --raw-results /scratch/djy8hg/runs/mosaickv/RUN/raw-full.jsonl \
  --raw-results /scratch/djy8hg/runs/mosaickv/RUN/raw-candidate.jsonl \
  --output /scratch/djy8hg/runs/mosaickv/RUN/cider-paired.json \
  --task coco_caption --model MODEL --backend hf --metric cider \
  --result-category paper_reproduction \
  --reference-method full_kv --candidate-method vl_cache_reimpl \
  --candidate-ratio 0.1
```

For MileBench, use `--metric milebench`. The output contains a separate paired
comparison for every official metric-bearing hierarchy key; it never emits a
heterogeneous overall macro.

For PrefixKV or LOOK-M native routes, additionally bind the versioned
protocol and materialized ledger. A PrefixKV FullKV run uses
`--fullkv-reference-output`; every compressed run uses both
`--fullkv-reference-input` and `--fullkv-reference-sha`. For example:

```bash
mosaickv evaluate \
  --task llava_description \
  --model llava-hf/llava-1.5-7b-hf \
  --model-revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --dataset-revision DATASET_OR_ARTIFACT_SHA \
  --method full_kv --retention-ratio 1.0 \
  --published-protocol configs/experiments/published_paper_native_metrics_v1.yaml \
  --published-protocol-id prefixkv_hf_reimpl_native_metrics_v1 \
  --paper-native-ledger /scratch/djy8hg/data/prefixkv/llava_description.json \
  --paper-native-ledger-sha LEDGER_SHA256 \
  --fullkv-reference-output /scratch/djy8hg/runs/prefixkv/fullkv_reference.json \
  --metric-output /scratch/djy8hg/runs/prefixkv/metrics.jsonl \
  --metric-observations-parquet-output /scratch/djy8hg/runs/prefixkv/observations.parquet \
  --metric-aggregates-parquet-output /scratch/djy8hg/runs/prefixkv/aggregates.parquet
```

Pass all paired raw result shards to `analyze_paper_native_metrics.py` to add
the nominal ratios and measured active-byte operating points to its output.
For an equal-budget claim, add `--require-byte-match` (default tolerance 1%).
The analyzer rejects unmatched sample identities or missing byte accounting.
For `--result-category unified_comparison`, active-byte matching is mandatory
and cannot be disabled.

To compare one source-hashed published value with one locally reproduced
aggregate without contaminating measured tables:

```bash
python mosaickv/scripts/compare_paper_reproduction.py \
  --paper-reported /scratch/djy8hg/runs/mosaickv/RUN/paper_reported.json \
  --locally-reproduced /scratch/djy8hg/runs/mosaickv/RUN/locally_reproduced.json \
  --output /scratch/djy8hg/runs/mosaickv/RUN/reproduction_comparison.json
```

The paper object must include its PDF or source-artifact SHA-256, exact
table/figure/page locator, and a preregistered absolute tolerance. The local
object must list the run IDs, manifest paths, manifest SHA-256 values, and the
scored/failed/attempted denominators. The attempted count must equal scored
plus failed. It must also provide the path and SHA-256 of the model-specific
FullKV retention-1 parity artifact. That artifact must compare at least 16
tokens for both ordinary generation and extraction/reinjection, achieve exact
token agreement, and remain within its declared logit tolerance. Each
referenced paper-lane manifest must contain evaluator
name/version/commit, the FullKV reference-output hash, and the pinned upstream
or local baseline commit. All model, dataset, metric, budget, and environment
identity fields must match or comparison fails. The command returns exit code
2 after writing a negative artifact when the absolute reproduction delta is
outside the preregistered tolerance; such an artifact cannot satisfy the
paper-reproduction acceptance gate.

The repository implements and CPU-tests these gates but does not convert an
unexecuted GPU reproduction into positive evidence. A paper table may add a
new baseline only after the model-specific retention-1 parity run,
official-metric fixtures, and at least one paper operating point all pass.

The anomalously low legacy PrefixKV TextVQA/DocVQA rows are quarantined from
primary paper tables and figures. They remain in raw diagnostic artifacts but
are not interpreted as PrefixKV performance. PrefixKV may re-enter a primary
comparison only after the native PPL/ROUGE lane and an official-checkpoint
parity point pass.

All model and dataset caches must remain under `/scratch/djy8hg`; `HF_TOKEN`
is accepted only from the process environment.
