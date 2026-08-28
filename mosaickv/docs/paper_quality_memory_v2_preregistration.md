# Frozen-candidate paper evaluation preregistration

## Decision boundary

MosaicKV paper candidate v1 is frozen at Git commit
`99a412bf0d2c639a095ff6efc5ca02e435b5e889`. Its configuration, selection
logic, hyperparameters, and decode behavior may not change after this boundary.
The exact hashes are recorded in
`configs/ledgers/paper_candidate_v1_frozen_99a412b.yaml`. Any behavioral
change creates a new candidate and cannot be reported as this preregistered
evaluation.

The freeze follows a positive but very small retrospective development check:
two samples per task passed the active-memory and exact-answer gates. Those
results are not paper evidence and are recorded only to explain why this
candidate, rather than another variant, was frozen.

## General-VQA replacement

RealWorldQA replaces MMStar. The installed `lmms-eval==0.7.2` package provides
a complete `realworldqa` task using the public `lmms-lab/RealWorldQA` test
split and deterministic exact-match scoring. It does not require an external
judge. The dataset revision is pinned to
`907c4e5228fd1703c710ed937601cb5f89ab8d5c`.

A preliminary path-and-content audit found zero RealWorldQA references in the
repository and prior MosaicKV run artifacts. This is necessary but not
sufficient evidence of untouched status. Before materialization, a canonical
identity-only exclusion scan must show no prior use. Any ambiguity or overlap
fails closed.

MMBench was not selected because the installed task package does not contain
the referenced default template needed by the desired task route, and its
available development evaluation route relies on external judge scoring.
RealWorldQA therefore gives the fixed environment a self-contained,
deterministic general-VQA route.

TextVQA validation and DocVQA validation remain in the evaluation for
task-level OCR and document replication. Their prior split exposure is
explicitly disclosed; neither will be described as untouched or used to
upgrade the RealWorldQA confirmatory result. No incompatible task metrics will
be averaged.

## Fixed comparison

The evaluation contains exactly one model and three methods:

- `llava-hf/llava-1.5-7b-hf` at revision
  `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, FP16, HF eager attention;
- `full_kv`, `prompt_attention_topk`, and the frozen
  `mosaickv_full__paper_candidate_v1`;
- 128 examples each from RealWorldQA test, TextVQA validation, and DocVQA
  validation, for 384 task examples and 1,152 paired method-example rows.

Samples are selected by a preregistered SHA-256 order over canonical dataset
identities before loading prompts, media, answers, references, or labels.
Ledgers are write-once and cannot be replaced after outcomes are inspected.

All paired arms use identical model weights, precision, processor, chat
template, media preprocessing, prompt, tokenization, generation parameters,
maximum output length, mandatory-token policy, block granularity, and backend.
Generation is greedy with a fixed seed and at most 64 new tokens.

## Unchanged gates

The nearest baseline is Prompt Attention Top-K. MosaicKV passes only if both
conditions hold separately for every task:

1. every paired MosaicKV sample uses strictly more than 30% fewer measured
   active key-and-value tensor-storage bytes than Prompt Attention Top-K; and
2. MosaicKV's mean official task score is at least the paired Prompt Attention
   Top-K score, with a zero-point margin.

RealWorldQA and TextVQA use exact match; DocVQA uses ANLS. Paired bootstrap
95% confidence intervals with 10,000 resamples are reported but do not weaken
the zero-margin quality gate. Residual and total cache storage, answer and
token agreement, latency, and failure counts are secondary diagnostics.

The sample size is fixed before outcomes are visible. It is 64 times the
two-sample development check, but no formal power claim is made without an
independent estimate of paired score variance.

## Authorization state

This document is a preregistration draft, not an execution authorization. No
catalog, sample ledger, model inference, Slurm job, or result was created.
RealWorldQA task routing and identity tooling must first be implemented and
CPU-tested without changing the frozen candidate. The protocol then requires
separate review of the exclusion manifest, ledgers, disjointness report, and
retention-1 parity before GPU execution can be authorized.
