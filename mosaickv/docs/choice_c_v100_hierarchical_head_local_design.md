# Choice C V100: hierarchical head-local logit-safe eviction

## Status

V100 now has a strict configuration schema, CPU-only proposal planner, an
audited backend cache-state isolation constructor, and a bounded Hugging Face
probe-loop implementation. It is still not a measured result, frozen
candidate, or paper claim. The user has authorized only a retention-one parity
job followed, through an `afterok` dependency, by one compressed synthetic
sample on an A100. Calibration and held-out jobs remain forbidden. Retention-one
parity and the one-sample GPU debug remain unverified, so ordinary runtime
construction remains fail closed unless the debug-only environment gate is set.

The design uses V99 only as an immutable exclusion boundary and uses only the
aggregate mechanism facts permitted by the V99 negative seal. It does not use
V99 answers, references, task scores, generated token sequences, per-sample
telemetry, failure identities, or knowledge of which samples were difficult.
A future calibration ledger must exclude every prior Choice C ledger.

## Why V99 was insufficient

V99 started from FullKV and repeatedly proposed one cumulative eviction. A
failed proposal was split until the search reached individual blocks. This was
safe and easy to audit, but it asked almost the same expensive question many
times: “Is this slightly smaller whole cache still safe?” Aggregate V99
telemetry records 8,073 candidate-cache probes for 64 samples, while only one
sample reached the fixed memory target. Local safe moves did not reliably form
a safe path to the global target.

V100 changes the search structure, not the scientific thresholds. It asks
which layer/head regions can safely lose capacity, builds a small collection of
nested choices, and verifies their combination. The KL-divergence, absolute
logit-difference, greedy-token, memory, fidelity, fold, and layout-utilization
gates remain unchanged.

## Core idea in simple terms

Imagine that every layer and KV head owns a shelf of evidence blocks. V99
removed a few books, checked the model, removed a few more, and repeated this
up to 128 times. V100 first groups shelves, tests several large changes, and
then opens only the groups that need a closer look. This is the same idea as
finding a bad cable by testing half a bundle rather than checking every wire.

Each shelf receives a nested capacity ladder:

```text
Full capacity  ->  coarse midpoint  ->  target-adjacent capacity
     S0        contains     S1      contains        S2
```

The lower-capacity set must be a subset of every higher-capacity set. Mandatory
tokens and protected coherent layout groups remain exact at every level.
Blocks never move between layer/head routes.

## Detailed algorithm

### 1. Reuse one FullKV teacher

V100 performs one ordinary prefill and one two-token FullKV teacher-forced
rollout. The resulting reference logits and greedy tokens are reused by all
candidate probes. A second prefill is forbidden. Every candidate cache is an
isolated state and is discarded before measured decoding.

### 2. Build the protected core

The exact protected core contains mandatory tokens, coherent multi-block
document/layout groups, and the preregistered prompt-mass floor. Protection is
label-blind. It may use prompt attention, forecast attention, attention times
value deviation, graph coverage, modality, and layout metadata; it may not use
the benchmark answer or score.

This protection is applied before constructing capacity ladders. Therefore a
lower ladder level cannot accidentally split a protected group.

### 3. Build route-local nested ladders

A route is exactly one layer and one KV head. For each route, blocks are ranked
once by the frozen label-blind risk rule. V100 creates three block-aligned
sets: FullKV, a coarse midpoint, and a target-adjacent set. The target-adjacent
set is contained in the midpoint, and the midpoint is contained in FullKV.

Nested sets matter because a failed aggressive reduction can be rolled back to
a known larger set without choosing an unrelated collection of blocks. This
makes the search deterministic and auditable.

### 4. Screen byte-balanced route partitions

Routes are assigned deterministically to eight partitions. Partitioning
balances proposed removed bytes and uses layer/head identity only for stable
ties. The first screening round tests each partition’s coarse capacity change
against FullKV while all other partitions remain exact.

Safe partitions keep their largest verified change. Unsafe partitions are
split into two deterministic child partitions. Binary refinement has a maximum
depth of two. An unresolved route remains exact; the algorithm does not guess.

### 5. Assemble a global cache

Every verified ladder step becomes an item with exact byte savings and
label-blind evidence utility. A deterministic multiple-choice knapsack chooses
exactly one capacity level for each route. Among solutions at or below the
hard cap it retains the largest actual byte count, avoiding unnecessary
over-compression, then maximizes evidence utility. Atomic layout groups remain
indivisible.

The knapsack is not itself a correctness proof. Effects from two individually
safe route groups can interact, so the assembled cache receives a joint logit
probe.

### 6. Resolve interactions and verify jointly

If the assembled cache fails, V100 rolls back the partition contributing the
largest preregistered output-impact risk and substitutes a verified safe donor
step with the same or greater savings. At most two conflict-resolution rounds
are allowed. The final cache must satisfy greedy-token equality, KL at most
0.02, and maximum absolute logit difference at most 1.0 across both probe
steps.

The entire search is limited to 24 candidate-cache evaluations and six
sequential decision rounds. The first correctness implementation evaluates
candidate caches sequentially; it makes no batched-probe performance claim.

### 7. Fail closed without a fake target result

If no jointly verified cache reaches the cap, V100 executes the smallest
jointly verified exact cache and records its actual byte count. It marks
`target_reached=false`, fails the freeze gate, and does not substitute an
unverified target cache. This prevents a memory number from being paired with
quality produced by a different safety policy.

## Why this could help

The hierarchy reduces repeated full decode probes by asking coarse questions
first. Route-local ladders preserve deterministic cache ownership and make
rollback cheap. The final joint probe catches nonlinear interactions that
independent head tests cannot see. Starting from FullKV retains V99’s safe
reference, while the strict probe cap prevents another 128-probe search.

This is a hypothesis, not a result. It may still fail if no cache at the 30%
memory advantage is inside the two-token stability envelope, if head effects
interact too strongly, or if dense physical padding erases route-local byte
savings.

## CPU implementation completed

The CPU implementation now includes:

1. Strict opt-in configuration fields for hierarchy, probe, and rollback
   limits. V100 is mutually exclusive with the V98 shortlist and V99
   progressive modes; unused V100 controls must remain zero.
2. Deterministic, nested, block-aligned route-capacity ladders with mandatory
   and atomic protected cores.
3. Deterministic byte-balanced partitioning, bounded binary refinement, and
   unresolved-route exact fallback.
4. Sparse multiple-choice knapsack with actual-byte accounting, one capacity
   level per route, deterministic tie-breaking, target underfill, and target
   shortfall records.
5. A fail-before-execution probe-budget tracker for both candidate count and
   sequential rounds, plus backend-neutral token/KL/logit gate evaluation.
6. Bounded risk-directed rollback with only previously verified donor levels.
7. CPU tests for configuration rejection, nesting, randomized tensor-free
   route shapes, atomic protection, partitions, probe limits, knapsack,
   unavoidable capacity shortfall, and interaction rollback.
8. Backend-neutral isolated candidate construction with recursive metadata
   cloning, K/V and auxiliary-storage alias detection, immutable FullKV
   reference checkpoints, successor-state adoption checks, and disposal-time
   mutation checks.
9. Mocked CPU tests proving that cache, attention-mask, nested model metadata,
   packed validity masks, and separate candidate states cannot leak into one
   another or into the reference. Malicious aliasing adapters and reference
   mutation fail explicitly.
10. A bounded HF controller that reserves a joint-probe slot, truncates
    refinement before either hard budget is exceeded, admits only explicitly
    passing route levels, assembles them with the deterministic knapsack, and
    disposes every candidate through an isolation handle. Joint failure uses
    only verified rollback/donor levels; exhaustion returns FullKV and records
    `target_reached=false`.

The initial correctness path evaluates candidate caches sequentially. It does
not claim batched-candidate performance or measurements.

## Authorized GPU validation boundary

1. Create a clean immutable commit containing the integrated loop and debug
   scripts.
2. Run checkpoint-level retention-one FullKV parity at the documented FP16
   eager tolerance.
3. Only if parity exits successfully, run one compressed synthetic sample with
   `MOSAICKV_ENABLE_V100_GPU_DEBUG=1` and verify probe budgets, state disposal,
   byte accounting, and trace completeness.

The YAML calibration and held-out submission flags remain false. The runtime
rejects the V100 flag unless the exact debug environment value is present, so
an unverified V100 configuration cannot silently execute an older method or
enter an ordinary experiment. Successful debug execution would establish only
implementation correctness evidence; it would not authorize calibration or a
paper result.

## Planned calibration and unchanged gates

After implementation, one fresh disjoint calibration may compare FullKV,
Prompt Top-K, and the single V100 candidate on 32 DocVQA and 32 TextVQA
examples. The candidate must use at least 30% less active and total cache memory
than Prompt Top-K. FullKV answer and token fidelity may be at most five points
below Prompt Top-K on every task and every one of four deterministic folds.
Every sample must reach the target, pass the final joint logit gate, retain zero
residual bytes and zero prototypes, and meet 75% atomic-group reservation
utilization.

No held-out ledger may be materialized unless all gates pass and a frozen
candidate is created.
