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
It is deliberately `execution_authorized: false`: dataset and checkpoint
revisions that the upstream artifacts do not provide must be resolved before
measured rows are produced.

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
  FullKV answer artifacts, token-weighted corpus PPL, and per-sample ROUGE-L;
- `evaluation.milebench`: the released LOOK-M normalization, multiple-choice,
  needle, and ROUGE routes plus equal-dataset subtask aggregation;
- `evaluation.tasks`: COCO Caption and MathVista routes in addition to the
  existing DocVQA route;
- `evaluation.lmms_adapter`: task-owned scoring plus optional paper-native
  metric JSONL/Parquet output;
- `analyze_paper_native_metrics.py`: paired confidence intervals that rerun
  the appropriate metric reducer on each bootstrap draw.

CIDEr is corpus-valued and is recomputed from the complete prediction and
reference cohort with `pycocoevalcap`; it is not averaged from per-example
CIDEr numbers. PPL is `exp(sum NLL / sum target tokens)`, matching the pinned
PrefixKV driver, which flattens all target-token NLLs before taking the mean.
For PPL, every sample's NLL sum and token count are retained so paired
bootstrap resampling remains mathematically valid.

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

The papers also report H2O, StreamingLLM/Local, PyramidKV, Elastic Cache,
SnapKV, and RoCo variants. They are registered in the protocol as
`unsupported_unpinned`, not silently approximated by Prompt Top-K. Before one
can enter a measured table it needs:

1. a paper and source mapping;
2. an immutable official commit and preserved license, or a separately named
   `*_reimpl` implementation;
3. retention-one identity and exact byte-accounting tests;
4. a native-benchmark reproduction point;
5. the same model, processor, prompts, media, decoding, precision, backend,
   warmups, and active-byte budget used for MosaicKV.

Until then, FullKV and Prompt Top-K remain the verified common-runtime
comparators. They are not renamed as published methods.

## Commands

Validate the protocol without running models:

```bash
source /scratch/djy8hg/workdir/MosaicKV/mosaickv/scripts/cache_env.sh
python mosaickv/scripts/validate_published_protocols.py \
  mosaickv/configs/experiments/published_paper_native_metrics_v1.yaml
```

The stricter command intentionally fails while unresolved or unsupported
entries remain:

```bash
python mosaickv/scripts/validate_published_protocols.py \
  mosaickv/configs/experiments/published_paper_native_metrics_v1.yaml \
  --require-executable
```

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
  --reference-method full_kv --candidate-method mosaickv_exact \
  --candidate-ratio 0.1
```

All model and dataset caches must remain under `/scratch/djy8hg`; `HF_TOKEN`
is accepted only from the process environment.
