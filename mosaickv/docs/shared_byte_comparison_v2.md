# Frozen comparison rules — version 2

This is a rules/implementation freeze, **not authorization for a new campaign**.
The [contract](../configs/experiments/shared_byte_comparison_v2.yaml) supersedes
the old byte-capped v1 placeholders for new comparisons. It does not rewrite
any historical protocol, raw result, running worker, or paper claim.

## Candidate and controls

The single candidate is `mosaickv_exact`: the existing prompt-window-16,
stable-head Prompt Top-K plus 1/256 routed-head graph policy, based on commit
`968b5fe4d7169b8c5c84f40f69c4933821432d4e`. There is no selection retuning in
this update. Prototype merging, residual storage and repair are disabled.
New instrumentation records live-cache backing storage. The contract binds
both the resolved algorithm-policy SHA-256 and source-content SHA-256 after
this update. A subsequent clean execution commit must preserve those contents.
This working-tree freeze is not itself a clean canonical execution SHA.

Within a task/lane, bind identical weights/revision, processor/revision, chat
template, prompts, media, preprocessing, tokenization, precision, attention,
generation, block and mandatory policies, sample identities, evaluator,
reference artifacts, warmups, repetitions and GPU controls. Use FP16 HF eager,
one-token blocks, deterministic fixed output limits, five warmups per method,
and seed 20271014. Quality has one recorded trial; latency claims require a
separate, identically controlled 20-repetition measurement. FullKV may supply
quality/reference outputs but is never a compressed-method latency competitor.

| Lane | Model | Actual route | Separate metrics | Registered caps (MiB) |
|---|---|---|---|---|
| PrefixKV | LLaVA-1.5-7B | `llava_description`: `PrefixKV/LLaVA-Description`, `official_1000`; `mmvet_prefixkv`: `PrefixKV/MM-Vet`, `official_218` | Corpus PPL; FullKV-reference ROUGE-L | 40, 80, 120, 160, 240 |
| VL-Cache | LLaVA-v1.6-Mistral-7B | `lmms-lab/COCO-Caption`, **`val`**; compatible `lmms-lab/DocVQA`, `validation` | CIDEr; ANLS | 10, 20, 40, 80, 120 |
| LOOK-M | LLaVA-1.5-7B | Materialized `MileBench/MileBench`, `official_evaluation`, representative subtasks | Accuracy; ROUGE-L; official hierarchy summaries | 10, 20, 40, 80 |

Every lane includes FullKV, Prompt Top-K, its named `*_reimpl`, and MosaicKV.
PrefixKV uses deterministic profile-shape rescaling with append-only decoding;
this is explicitly a **byte-matched reimplementation variant**, not official
PrefixKV. Its profile/hash and native fidelity evidence still need binding.
The ongoing C-only OCR validation is a separate variant, not a replacement
for PrefixKV's original PPL/ROUGE-L routes. Official reproduction stays in a
separate table/environment/budget convention. LOOK-M remains `lookm_reimpl`;
an upstream commit is attribution, not proof that official execution works.

PrefixKV generation is registered at 1,024 tokens, matching the existing
native-route output limit; PPL evaluates the **complete same annotation target**
for every method, separately from greedy generation. Do not truncate a longer
target silently. COCO, DocVQA and MileBench use 64 generated tokens. Reusing
DocVQA's historical 32-token rows in that 64-token lane is prohibited. ROUGE-L
uses a hash-bound FullKV-output artifact with identical controls, not answers
from a different precision or generation run.

## What was reconciled, and what remains unbound

Inspection of the original manifests and setup ledgers confirmed the two
PrefixKV dataset revisions, splits and full-ledger hashes recorded in the
contract. `mmvet_prefixkv` is the paper's 218-example route, **not** ordinary
judge-scored `mmvet`. COCO's old `validation` placeholder was incorrect; the
registry and actual manifests say `val`. No historical file was changed.

The PrefixKV parent ledgers are hash-bound reusable input artifacts, not proof
that their historical output rows meet the new controls. COCO requires a
new-plan binding to an explicit reviewed ID ledger. DocVQA reuse requires an
exact compatibility audit, including bytes and generation length. MileBench's
annotation and media sources must be bound independently in a composite
ledger. Do not assign one repository's revision to both sources. Until that
ledger and representative media are complete, its revision is explicitly
`null` and preparation fails closed. The intended strata are ActionSequence,
CLEVR-Change, Spot-the-Diff and TextNeedleInAHaystack; report only actually
validated subtasks and disclose any merged-image transport.

## Physical byte boundary

The cap applies to the **initial live compressed K/V backing storage after
construction, before reserving decode capacity**. Use the largest feasible
allocation below it. Bind an independent layout-capacity certificate recording
that maximum, the cap, and the unique-live-backing-storage boundary. Report
target shortfall explicitly; the cross-method realized-byte tolerance is zero.
If methods' maxima differ at a cap, that point is unsupported as a direct
same-byte comparison. Do not arbitrarily underfill one method to manufacture a
match. Record mandatory-policy infeasibility rather than dropping tokens.

Runtime traces now contain `cache_lifecycle`, and raw/Parquet result schemas add:

- `initial_active_kv_storage_bytes`;
- `final_active_kv_storage_bytes`;
- `peak_active_kv_storage_bytes`.

The legacy `active_kv_bytes` remains the post-compression **tensor payload**
(`numel × element size`) for backward compatibility; physical padded slots are
included. It is not necessarily backing storage: a short view can retain a
larger allocation. New storage accounting deduplicates allocations and counts
unused preallocated capacity. It samples after compression, after preallocation,
after each provisional decode and after each maintenance/repair transaction.
It does not infer a peak from the final value. Metadata reads do not copy cache
values or synchronize CUDA. Their cost is included in the existing reported
trace-instrumentation overhead, excluded from algorithm timing identically
for every method.

Peak here means the largest **live decode-cache storage at transaction
boundaries**, not the sum of transient old/new buffers within a concatenation,
nor the process peak. Model weights, inactive FullKV/source copies, planner
scratch, and allocator reservations are excluded. Preserve the separate
FullKV prefill bytes and peak allocated/reserved GPU-memory diagnostics; do not
claim this cache-only peak is total inference HBM usage. Append-only decoding
may increase final/peak storage above the initial cap. Match all three storage
quantities across compressed methods, using identical decode growth controls.

Old rows with no backing-storage/peak evidence remain readable with null fields;
they are not silently upgraded. The running clean-snapshot workers are unchanged.

## Validation, reuse and statistics

`evaluation.shared_byte_contract.validate_group` accepts a **derived** comparison
ledger, expected per-sample control fingerprints and independently bound
capacity certificates. It rejects missing/duplicate samples, failures, mixed
lanes/tasks/metrics, changed controls, fallback/eviction, underfilled allocations,
missing storage measurements, and unequal realized bytes. It returns only
compressed methods as latency competitors. Metric reduction remains in the
existing metric-aware analysis code, after this comparison gate.

Use sum-NLL/sum-token corpus PPL; rerun corpus CIDEr in each paired bootstrap;
recompute the official MileBench hierarchy in each replicate. Keep Accuracy,
ANLS, ROUGE-L, CIDEr and PPL labeled separately. Never average incompatible
scores. A shared-byte point tests quality and latency at equal memory; a
“30% smaller at equal quality” claim instead needs supported iso-quality points
from the curves and preregistered quality tolerances—not isolated unequal points
or synthetic interpolation passed off as measurements.

`reuse_decision` distinguishes reusable completed observations, missing/failed
observations, incompatible historical observations, and observations needing a
provenance audit. A retry does not erase the original failure. Reconcile source,
config, baseline/profile revisions, sample IDs, all control hashes and bytes
before reuse; reuse FullKV across caps only when its controls/reference artifact
are identical. Keep the original nominal-ratio campaign read-only as separate
historical curves. No whole-campaign rerun is authorized or required here.

## Resources and read-only verification

Maximum **four A100s across all MosaicKV arrays combined**, always. Each GPU
worker uses one A100, 12 CPUs and at most two hours. Register arrays with the
shared controller; `%4` on several independent arrays is not a global limit.
Preserve already-running jobs. Subdivide pending shards if a measured duration
would exceed two hours, without changing sample selection or method controls.

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/verify_shared_byte_contract.py --check-parent-ledgers
```

This command reads the freeze and existing parent ledgers only. It does not
materialize datasets, change raw results, install packages, or submit jobs.
Preparation now binds this contract and refuses candidate drift/unresolved
routes. CPU verification is not GPU parity, baseline fidelity, or paper-ready
empirical evidence. The large lanes remain blocked until their concrete plans,
inputs, fidelity evidence and physical-storage checks are complete.
