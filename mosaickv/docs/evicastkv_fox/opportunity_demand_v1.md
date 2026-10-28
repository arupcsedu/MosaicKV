# Opportunity routing and demand-weighted coverage: experimental V1

This opt-in revision is **not** the frozen Fox-review method, and is not a
paper-ready superiority result. The reviewed candidate remains unchanged when
`evidence_policy` is absent; its canonical config hash excludes the new null
field. The original six-mechanism architecture is not silently redefined:
these are prompt-window, exact-selection-only variants, with draft, prototypes,
residual storage and repair disabled.

## Factorial controls

| Arm | Routing | Coverage objective | Per-query final floor |
|---|---|---|---|
| `complete` | frozen entropy | frozen uniform target demand | frozen behavior |
| `evidence-routing-v1` | bounded feasible opportunity | frozen objective | used for opportunity scoring only |
| `evidence-coverage-v1` | frozen entropy | prompt-demand weighted | 0.95 of each query's Top-K anchor mass |
| `evidence-both-v1` | bounded feasible opportunity | prompt-demand weighted | same 0.95 floor |

Both 20 and 40 MiB are retained. Prompt Top-K and `vl_cache_reimpl` run at each
cap; FullKV runs once as a quality reference, not as a compressed latency rival.
The routed fraction stays **1/256** (one of 256 routes for the pinned COCO
checkpoint). Stable routes still execute the common scalar Top-K selector.
Local value/novelty coefficients, graph degree/shortlist, prompt window, model,
precision, tokenization, input processing, mandatory policy and append schedule
are unchanged. No layer/head capacity redistribution is introduced.

## Opportunity score

For a route let `A[q,i]` denote actual prompt attention, normalized over **all**
source positions for each prompt query. GQA query heads are averaged into their
KV owner by the existing capture helper; the last 16 prompt positions stay
separate. Let `T` be the feasible scalar Top-K set at the route's fixed capacity,
including mandatory entries. Its query masses are `m[q] = sum(i in T) A[q,i]`.
Use the common target `tau = max(mean(m), 1e-12)` and define:

\[
 H(S)=\frac1Q\sum_q\min\left(1,\frac{\sum_{i\in S}A_{qi}}{\tau}\right).
\]

The common target prevents averaging away low-mass queries. A *separate* target
equal to each query's anchor mass would saturate every query at the anchor and
leave no positive opportunity; that is deliberately not used.

Construct at most 8 removable and 8 omitted candidates using the derivative of
this capped objective at `T`. Enumerate at most 64 one-token equal-byte swaps,
never remove mandatory entries, and reject any proposal whose query mass falls
below 0.95 times that query's anchor mass. The score is the greatest positive
gain divided by `max(1-H(T), 1e-12)`, clipped to [0,1]. Pair ties use source IDs;
route ties use layer/head IDs. No positive gain means score zero; the fixed route
count is still honored with deterministic ties. No benchmark labels, draft
tokens, model forwards, gradients or global score-to-byte allocation enter it.

This score estimates **improvability**, not entropy or unavoidable error. The
witness exchange is diagnostic evidence of a feasible opportunity; it is not
silently installed in the routing-only arm. That arm continues to use the
original routed selector and can still produce a worse cache.

## Demand coverage and explicit query protection

For the routed graph let `d[j]` be prompt-query mean attention, divided by the
number of routes. Normalize only roundoff over the complete routed domain;
reject an incomplete domain rather than renormalizing a shortlist. The new term
is `C(S) = sum(j) d[j] max(i in S) s[i,j]`, including self coverage. Parallel
edge types retain the original maximum-per-pair rule. Nonnegative demand keeps
facility diminishing returns. The local utility and modality terms are unchanged.

The demand term alone is **pooled**, not individual-query protection. The latter
is a separate final constraint: for every routed prompt query, require retained
mass at least 0.95 of its own same-capacity scalar Top-K anchor. A violation
rolls back that route to its Top-K anchor before cache construction. This is
not FullKV fallback and does not change cardinality or measured physical bytes.
Report pre-guard ratios and rollback counts, including all-rejected cases.

The 0.95 floor is an explicit new surrogate design choice, not a registered
benchmark-quality margin and not a mathematical quality guarantee. It is fixed
in the versioned registration before GPU testing. It must not be loosened after
seeing rejections. Setting it to 1.0 is supported but may permit only tied sets,
because Top-K already maximizes pooled attention mass at fixed capacity.

## Execution and interpretation

`evicastkv_opportunity_study.py prepare` binds the unchanged parent validation
ledger, copies its first two **already consumed** inputs, and registers 13 arms
per input (26 requests). No freshness assertion or new cohort is manufactured.
Every model-loaded worker checks ordinary/reinjected/packed retention one;
all three opt-in variants get their own 16-token retention-one check. Every
compressed arm also compares kept K/V, positions, physical allocation and
16-step forced continuation against the independent representation oracle.

The canary checks numerical/storage correctness and activity only. Its clone
captures contaminate resource measurements; its timings are **not headline
performance**. Captions/references are retained, but two consumed inputs are
not statistical quality evidence. Acceptance cannot depend on favorable CIDEr.
No fresh development, confirmation or paper campaign is auto-submitted.

### Exact baseline diagnostic correction

The first canary's four VL-Cache requests hit the exact-only oracle's literal
`tier == exact` check before numerical validation. The pinned local VL-Cache
payload builder labels unchanged source entries `vl_cache_reimpl_exact`.
The corrective oracle accepts that explicit label only; it still independently
compares every retained K and V bitwise to the source and rejects transformed,
unknown, prototype and residual tiers. No runtime, policy, precision or numerical
tolerance changes are permitted when reusing the other 22 passed requests.

VL-Cache's unchanged adaptive layer allocation can legitimately introduce
padding in the shared dense packer. Its padding must be reported and charged
to the same actual initial/final/peak physical byte boundary; it must not be
forced to use uniform allocation or certified by logical bytes alone. Zero
padding remains required for Top-K and all four MosaicKV arms. Independent
packed/original-layout numerical tests remain mandatory for VL-Cache as well;
recognizing a tier name is not a numerical pass.

The selective retry preserves the original 26 attempts, reuses only 22 successes,
and runs only the four failures. Merged analysis reports all 30 attempts and
rejects duplicate successful keys or any policy/runtime source change.

Commands, after a clean immutable commit:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_opportunity_study.py prepare --root /scratch/djy8hg/runs/evicastkv/OPPORTUNITY_NEW_ROOT
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_opportunity_study.py submit --root /scratch/djy8hg/runs/evicastkv/OPPORTUNITY_NEW_ROOT
```

The finite array uses two one-A100 workers, 12 CPUs each, 20-minute limits,
`bii-gpu`/`bi_fox_dgx`; a CPU `afterany` callback always analyzes completeness.
Aggregate queued/running GPU admission remains capped at six. Existing jobs
are not altered. CPU caches, model caches and temporary files stay on scratch.

After a passing, active canary, a **separate freshness-audited development
registration** is needed for the four-arm quality comparison. That cohort is
not confirmation. Stop if no useful quality benefit appears. Any independent
COCO confirmation needs frozen parameters, prespecified N/tests/stopping rules,
complete byte-matched controls, and honest negative-outcome reporting.
