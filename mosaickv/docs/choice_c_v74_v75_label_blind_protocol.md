# V74/V75 label-blind OCR and chart protocol

## Purpose

V73 passed the active-memory target but failed its per-task quality gate on DocVQA and ChartQA. A label-blind paired audit found that the frozen candidate constructed no spatial, OCR/layout, or same-region edges on any of the 96 audited examples. It also routed 960 of 1,024 layer/KV-head identities through external Prompt Top-K selection and disabled repair. These are mechanism findings, not benchmark-label findings.

## Data separation

V73 answers are used only to determine whether MosaicKV, Prompt Top-K, and FullKV emitted the same string. The audit script projects the Parquet input to an explicit column allowlist and never loads `reference` or `task_score`.

V74 uses 16 new DocVQA and 16 new ChartQA examples. Its immutable ledgers exclude all matching repository ledgers, the v73 held-out ledgers, and other declared external campaign ledgers. Candidate selection uses only:

1. exact answer agreement with the model's FullKV output;
2. position-wise generated-token agreement with FullKV; and
3. measured active and residual KV bytes.

Ground-truth answers and official task scores are removed from v74 calibration telemetry and are not inputs to the freeze analyzer.

V75 is materialized only after v74 creates an immutable frozen-candidate file. Every v75 ledger records that file's SHA-256 and excludes the v74 calibration ledgers, the v73 held-out ledgers, and every matching declared repository ledger.

## Frozen candidate search

The v74 YAML preregisters four candidates: the v73 control and three evidence-preserving variants. The variants enable model-authoritative patch-grid spatial edges and vary prompt reservation, structural-evidence reservation, routed-head fraction, candidate fraction, and prototypes per routed head. They do not change the common active-byte cap, model, processor, prompt, media, precision, generation, or backend.

A candidate is eligible only if, on both calibration tasks, it uses at least 30% fewer active bytes and at least 30% fewer active-plus-residual bytes than Prompt Top-K. Among eligible candidates, the frozen choice maximizes, in order:

1. worst-task exact-answer agreement with FullKV;
2. macro token agreement with FullKV;
3. worst-task active-plus-residual reduction; and
4. lexicographically smallest candidate ID for deterministic ties.

If no candidate satisfies the memory requirements, the freeze job exits nonzero and Slurm dependencies prevent held-out materialization and evaluation.

## Held-out gate

V75 evaluates the frozen candidate, FullKV, and Prompt Top-K on 48 new examples each from DocVQA, MMStar, and ChartQA with two measured repetitions. The existing per-task quality-memory analyzer requires:

- at least 30% lower measured active KV bytes than Prompt Top-K;
- nonnegative mean task-score difference against both FullKV and Prompt Top-K;
- paired bootstrap lower bounds above the preregistered -0.05 margin;
- deterministic repeated answers;
- complete rows; and
- a transformation-free FullKV reference.

Latency remains informational and cannot reject the quality-memory result. A failed v75 gate is preserved as a negative result; it is never used to retune the frozen candidate.

## Execution

The workflow requires a clean committed source snapshot and the common environment:

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh
mosaickv/scripts/assert_clean_worktree.sh
bash mosaickv/scripts/submit_choice_c_v74_v75.sh
```

The dependency chain is calibration-ledger materialization, two A100 calibration workers, CPU freeze analysis on `parallel`, held-out-ledger materialization, three A100 held-out workers, and CPU held-out analysis on `parallel`. No stage polls logs or spends a GPU while waiting for a predecessor.
