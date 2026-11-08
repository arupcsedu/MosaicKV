# Compact escapes and lower-overhead six-mechanism hybrid

This is a consumed-input engineering study, not a new quality campaign. The
validated storage-only defaults and retained hybrid are unchanged. Every arm
uses the same LLaVA-1.5 checkpoint, BF16 eager arithmetic, media, prompt, 128
prediction limit, protected-token policy and original logical positions.

## Changes and separate controls

1. **Common source codec:** original modulo-eight predictive coding, with a
   two-bit rare-symbol alphabet only when its actual payload plus two palette
   bytes is smaller. Original sources shared by all K/V consumers are retired;
   neither quantization nor model arithmetic changes. All packet offsets,
   selector bytes and replay buffers are charged. Raw fallback is lossless.
   This optimization is applied to storage-only controls too; it is not a
   benefit credited to forecasting or prototypes.
2. **Planner variant:** retain all source nodes, forecast, sparse graph,
   value-aware utility, budgeted selection, real prototype/residual tiers, and
   repair. Learned descriptor pooling/graph neighbors and optional selection
   use a deterministic top-128 forecast shortlist plus mandatory nodes.
   Vectorized pooling reduces Python/device-transfer overhead. This is an
   explicit graph/selection approximation, not an equivalent graph hash.
3. **Online policy variant:** evaluate actual prototype attention mass times
   dispersion before PV. On a trigger, promote the real pinned residual,
   update the current K lease and replay scores once; the ordinary V fill
   uses restored V. No second vocabulary readout or private decode probe is
   performed. This does not guarantee parent answers. A separate forced
   control validates the transaction, not naturally recovered benchmark quality.

The five arms are old-source storage, compact-source storage, compact-source
retained hybrid, compact-source shortlisted hybrid, and compact-source
shortlisted score-risk hybrid. All six mechanism implementations remain intact.
The overlay is still additive; single-head prototypes do not retire the shared
all-head source. No claim of additional prototype-induced compaction is made.

## Preregistered finite admission

Two saved original COCO code archives and the same two consumed model inputs
are hash-bound. CPU round trips must be exact and complete source plus replay
storage must fall by at least 1%. Native packets must match CPU payload and
offset bytes on every layer; in-place replay must preserve isolated sources.
A decoder median-ratio screen of 1.05 only prevents an obviously regressing
operator from launching a larger engineering benchmark. It is **not** the
request-latency acceptance criterion.

Five model canaries precede timing. Codec arms require bitwise agreement with
all 128 frozen storage-parent vocabulary vectors. The unchanged retained
hybrid requires bitwise agreement with its own frozen hybrid logits plus all
128 storage-parent greedy tokens; guarded candidate logits are not falsely
assumed equal to parent logits. The shortlisted and guard-free arms require
the unchanged first-32-token agreement, maximum KL <= 0.02 and
maximum logit difference <= 1.0. A divergent greedy prefix invalidates later
same-prefix KL interpretation. The guarded shortlist must also preserve all
128 parent tokens. Invalid arms remain recorded and untimed.
Accepted arms receive five warmups and twenty randomized paired repetitions.
Strict latency requires the Bonferroni-adjusted one-sided upper ratio <= 1.00
over ten registered input/comparison pairs. Four-percent superiority requires
<= 0.96. Slower results are trade-offs or failures, not passes.

## Memory evidence

Separate audited requests observe prefill tensor results, source conversion
overlap, asynchronous construction, and decode-owned tensor backing. A whole
request allocator envelope is also measured, suppressing inner peak resets.
Auxiliary GPU tiers and pinned CPU residuals are reported separately. These
phase observers do not yet provide a time-aligned cross-thread KV ownership
ledger or classify opaque allocator workspaces. Therefore exact whole-request
KV-only peak remains explicitly uncertified; it is never replaced by a sum or
maximum of partial counters. This study improves coverage, not a false seal.

## Execution

From a clean source, source `mosaickv/scripts/cache_env.sh`, set `PYTHONPATH` to
this worktree's `mosaickv/src:mosaickv/scripts`, then run
`python mosaickv/scripts/compact_hybrid_study.py prepare --root NEW_ROOT`.
CPU workers run on `bii`; positive gates alone release two one-A100 workers
under `bi_fox_dgx`, twelve CPUs each, at most one hour. Arrays use the shared
six-GPU aggregate controller. `afterany` analyses run on failures too and
conditionally launch only the registered next engineering stage. No fresh
cohort or paper-scale experiment is authorized.
