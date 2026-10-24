# V4: direct-error, single-slot exact-cache exchanges

## Why V3 is not promoted

Preserve `/scratch/djy8hg/runs/mosaickv/bounded-history-memory-v3-20261023-v1`
unchanged as a negative development result. Source `b1ffddb4fad7001e1fa54152fa1448aab9aa0f4e`
completed 208/208 rows. None of its 48 generation proposals or 48 teacher-forced
proposals passed the every-head reconstruction guard. All final selections
therefore matched the original MosaicKV exact anchor. Proposed average error
could fall while individual heads regressed; no proposed donor layer was safe.
The V3 identities are consumed and explicitly excluded below. We do not use
their answers, correctness scores, or failure identities to tune V4.

V4 targets this proposal-feasibility failure. It is not a proven answer-quality
improvement, a baseline reproduction, a prototype method, or a repair method.

## Algorithm

Start from the unchanged MosaicKV exact compressed anchor. For each layer/head,
reuse existing prompt-forecast attention `a_i` and the original cached values
`v_i`. Let `m` be retained mass, `n` retained weighted-value sum, and `o*` the
full prompt-derived attention output. Compute candidate effects directly:

```text
after removing i: (n - a_i v_i) / (m - a_i)
after adding   j: (n + a_j v_j) / (m + a_j)
error: mean_channels (candidate_output - o*)^2
```

These formulas include softmax renormalization after eviction. This is more
specific than ranking by attention mass or value norm, but still a surrogate
for future autoregressive answer quality. No future labels or benchmark
references enter the planner.

Build source attention, FP32 weighted values, mandatory masks, selected masks,
and FP64 reductions once per layer. Find the lowest-error admissible removal
or addition independently per head, with deterministic physical-index ties.
Each complete donor layer loses one optional token per KV head; another layer
gains one token per KV head. Different heads may select different source
positions, but their **lengths** remain equal within a layer. Equal audited
layer-slot byte costs make each transaction exactly byte-neutral, with no
cross-layer padding. K/V, post-RoPE keys and logical positions remain unchanged.

Every affected head must satisfy both the current-state and original-anchor
bound, with the unchanged tolerances `atol=1e-7`, `rtol=1e-6`. The summed
head-MSE change of the pair must be strictly positive. Checking the original
anchor after every transaction prevents accumulating numerical tolerance drift.
After all transactions, recompute the full guard from original source values.
If that final guard fails, return the same-byte compressed anchor, not FullKV.

Choose the largest summed verified error reduction, with donor/receiver layer
IDs breaking ties. Refresh only the two changed layers' options. Bounds remain
0.5–1.5 times uniform prompt capacity, subject to mandatory/source feasibility.
The versioned policy permits at most 16 accepted exchanges, 64 pair
verifications, and 64 candidate positions per temporary chunk. Source scans
are bounded by the initial layers plus two layers per accepted exchange; no
new full model inference or quadratic evidence graph is introduced.

No safe donor, no safe receiver, or no positive verified gain means **stop and
keep the current admissible cache**. Do not weaken the guard to create moves.
The initial implementation supports one-token blocks only, using the existing
audited batch-one HF eager LLaVA transport. Other combinations fail explicitly.

## Validation and stopping rules

CPU-only fixtures must demonstrate an executable positive exchange, correct
safe-no-move behavior, exact bytes, mandatory preservation, source isolation,
determinism, analytical/brute-force agreement, complete replayable traces, and
bounded search. A separate fixed synthetic 32-layer/32-head case checks that
head-count scaling does not make all proposals automatically inadmissible.
Synthetic feasibility is never reported as real-checkpoint quality evidence.
The submission tool recomputes this CPU gate from the clean source SHA,
hash-binds its artifact, and refuses GPU submission if it fails.

The small V4 development protocol registers eight next-in-order examples per
task, excluding all 14 identities per task used by explicit V1/V2/V3 candidate
development. Parent cohorts have been evaluated historically, so this is
**retrospective development, not globally untouched/held-out evidence**.

Every compressed arm (Prompt Top-K, the unchanged MosaicKV exact anchor, and
`mosaickv_direct_error_v4`) runs every shared 32/40/80 MiB budget. FullKV runs
once per task as a quality reference, not as a latency comparator. LLaVA model
revision, FP16 eager attention, prompts/media, one-token blocks, mandatory
policy, greedy 1,024-token trajectories, five warmups, and the common V2
generated-history controller remain unchanged. PPL and FullKV-reference
ROUGE-L remain separate. ROUGE-L is not official MM-Vet answer accuracy.

Short teacher-forced targets retain the existing common rule: reserve only
the available decode updates. Validate both phases independently; require
identical selections when their prompt budgets match. All compressed methods
must match realized initial/final/peak KV bytes within each comparison.

There are 20 one-A100 workers and 160 sample-arm rows, with 12 CPUs, 96 GiB
RAM and two-hour worker limits. An aggregate controller allows at most six
A100s; no old GPU workers are rerun. After an authorized scoped commit:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/submit_bounded_history_memory_v3.py --version 4 \
  --root /scratch/djy8hg/runs/mosaickv/bounded-history-memory-v4-NEW-IMMUTABLE-ID
```

## Reporting

Traces preserve every verified exchange, donor/receiver positions, per-head
before/after MSE, byte cost, capacities, final original-anchor guard, source
hashes, no-move reason, and timing. The analyzer distinguishes **guard passed**
from **cache actually changed**: the former alone cannot imply an improvement.
Raw failures and all planned denominators remain explicit. Preserve negative
outcomes, including a safe-but-useless planner or worse task fidelity.

Do not launch larger runs because execution or reconstruction passes. If V4
cannot produce safe transactions on real inputs or cannot show a consistent
quality benefit at the lower budgets, stop. More samples do not fix a known
algorithmic loss. Equal-budget comparisons measure quality/latency, not
smaller-memory superiority. A quality target and smallest-quality-eligible
competitor rule must be frozen independently before any separately authorized
powered confirmation. No such confirmation or paper campaign is enabled here.
