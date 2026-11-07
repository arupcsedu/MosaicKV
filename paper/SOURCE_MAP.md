# Manuscript evidence map — 2026-10-22

The paper is a presentation of existing immutable experiments, not a new
execution source. Its dirty documentation checkout is never substituted for
an experiment's clean SHA. No raw run record is edited.

| Cohort | Numerical source | Execution source |
| --- | --- | --- |
| COCO, 256 examples, all four caps | `/scratch/djy8hg/runs/mosaickv/coco-shared-32-edb6edf-retry1-20261014/analysis/lower-memory-frontier-21636391/report.json` | `edb6edfaf50f4539c9c1969371fefecf0eabccbf` |
| MileBench E6, 192 examples | `/scratch/djy8hg/runs/mosaickv/milebench-isoquality-e6-fdeca17-20261020/analysis/larger/` | `fdeca177c4a3eaf77df53780b3586a4dc462365c` |
| Earlier MileBench, 256 examples | `/scratch/djy8hg/runs/mosaickv/milebench-lookm-retry-c4d733b-20261019-v2/analysis/recovered-larger/` | `c4d733b7` family, with compatible immutable retries recorded by the source report |
| PrefixKV, 1,218 examples | `/scratch/djy8hg/runs/mosaickv/prefixkv/full-campaign-e4de390-20261019-v6/analysis/final/` | `e4de39001ea8edc74630f1864636bbc924d5486a`; candidate algorithm `95887454ec5428376e42b47db04269c3896913f7` |

The PrefixKV original analyzer produced complete metrics, paired comparisons,
sample rows, and LaTeX before failing to import matplotlib for an optional plot.
The paper exporter independently rechecks the 8,526 successful rows, unique
identities, task counts, and shared-byte boundaries. Its generated plot uses
those measured values. We do not manufacture a passed original report.

## Export and presentation

`scripts/export_results.py` checks source reports and their bound artifacts,
then writes measured tables, PGFPlots data, full paired-comparison CSVs, and
`generated/provenance.json`. The latter stores complete SHA-256 bindings.
Exported paper figures never use external published numbers as local results.
Main tables use mean physical storage; exact shared-byte tests are per sample.
MileBench summary-file sample counts refer to the whole cohort; the manuscript
explicitly reports 96 applicable examples per E6 metric and 128 per earlier
cohort metric. The cohorts are never pooled.

## Design and implementation anchors

- Frozen shared-cap contract and policy: `mosaickv/docs/shared_byte_comparison_v2.md`
  and `mosaickv/configs/experiments/shared_byte_comparison_v2.yaml`.
- Query evidence: `forecasting/`, model-specific hooks under
  `adapters/huggingface/`, and the frozen execution configs.
- Graph, utility, and coverage: `graph/`, `selection/`, and frozen
  `backends/hf_runtime.py` (1/256 head routing, vectorized coverage path).
- State/storage: `cache_state.py`, packing code, `evaluation/shared_byte_contract.py`.
- Native PrefixKV semantics: `baselines/prefixkv_reimpl.py`, the ragged eager
  transport, pinned upstream `597f1ab032704951550f93bcc8a23f1454b80aa4`, and
  four-example parity artifacts bound by the large campaign plan.
- Baseline interpretation: `mosaickv/docs/baselines/{prefixkv,lookm,vl_cache}_spec.md`.
- Inactive extensions: `prototypes/`, `residual/`, and `repair/`.

Source-module paths above are relative to `mosaickv/src/mosaickv/` unless
qualified. Contemporary dirty-tree experimental changes are not described as
features of the frozen measured policy. The appendix distinguishes exploratory
attention-output/draft/logit strategies from that policy.

## Claim boundaries

- Fixed-byte gains: supported for named reimplementations, tasks, and caps;
  preserve the close Prompt Top-K outcomes and negative points.
- Lower-memory equivalence: E6's registered joint test failed; descriptive
  COCO cross-budget point estimates do not replace a paired test.
- PrefixKV native: nominal ratio and ongoing eviction differ from append-only
  common-byte experiments; smaller initial storage reverses at final/peak.
- Timing: HF eager, batch one, request means; no FullKV latency competition,
  production-throughput inference, or upstream fused-kernel speed claim.
- Three-tier architecture: exact path evaluated; prototype/repair specifications
  do not support measured claims.
