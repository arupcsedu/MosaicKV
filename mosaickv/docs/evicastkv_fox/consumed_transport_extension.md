# Common-transport extension: consumed correctness inputs only

The two-input common-policy regate completed eight requests under source
`259796b816c922730bcd99ecc1ad5fe18de0d6cb`, with no failures and exact paired
initial/final/peak storage. It is representation evidence, not a task-quality
or latency experiment. The full CPU suite passed 1,726 tests. The failed
original native arithmetic and boundary-B controls remain negative.

Extend the same common valid-slot execution policy to the remaining thirty
inputs of the original, already consumed 32-input MileBench validation ledger.
Reuse the two passing inputs rather than infer them again. The parent contains
eight inputs from each of ActionPrediction, ActionSequence, CLEVR-Change and
Spot-the-Diff. Do not inspect annotations to select or replace any input.

The new array executes 120 requests: thirty inputs, two exact methods
(Prompt Top-K and `mosaickv_exact`), and 16/40 MiB initial physical caps. Four
workers process at most eight inputs each. Each worker independently checks
ordinary generation, extraction/reinjection, and MosaicKV retention-one parity
on its first input. The existing independent masked-source and compact-source
oracles check selected K/V and sixteen forced continuation steps.

Require unchanged numerical thresholds, original logical positions, no fallback,
and per-input initial/final/peak backing-storage matching. This does not establish
LOOK-M transformed-cache fidelity. Source hashes of transport, HF runtime,
adapter and worker must match the two-input controls before reusing them.

Allocate one A100, twelve CPUs, 64 GiB and 45 minutes per worker, using
`bii-gpu`, `bi_fox_dgx`, and `bii_dsc_community`. Limit this array to four workers
and all owned A100 jobs to six. The launcher now honors its registered wall
limit rather than inheriting a two-hour sbatch default. Chain one finite
`afterany` CPU analyzer; no polling, quality successor, fresh materialization,
or confirmation is authorized by this correctness registration.

Run from the committed clean worktree after sourcing `cache_env.sh`:

```bash
python mosaickv/scripts/evicastkv_consumed_regate.py \
  --root /scratch/djy8hg/runs/evicastkv/fox-review-20261028-mile-common-transport-32-v1 \
  --parent /scratch/djy8hg/runs/evicastkv/fox-review-20261028-v4 \
  --prior /scratch/djy8hg/runs/evicastkv/fox-review-20261028-mile-common-transport-v1
```

Global historical media/source identities remain unverified. Passing this
engineering check cannot certify a fresh confirmation cohort or superiority.

In parallel, run one bounded CPU ID review over both historical run namespaces,
recognizing all 29 official subtasks and explicitly named protocol-lock,
exclusion and analysis-plan manifests missed by the earlier root seeding.
Scheduler logs and raw prediction streams are not seeds. Each namespace has a
90-second traversal bound; missing links, unreadable roots and unresolved
aliases remain reported, not waived. This is not a global media-disjointness
certificate and does not materialize or select confirmation identities.
