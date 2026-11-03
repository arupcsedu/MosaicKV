# Bounded source-hotpath latency engineering

Only the same two consumed LLaVA-Description/MM-Vet inputs; independent quality
N=0. EviCastKV=MosaicKV. The frozen source-precision/storage variant disables
the six principal algorithm mechanisms; do not call it the complete method.
All baselines are local representation adaptations, never official execution.
Their differing measured storage points are NOT byte-matched comparisons.

The completed operation profile identifies source projection dispatch and
index/view preparation as near-dominant named host work. Protected scatter
and sparse-patch dispatch are also substantial. CUDA event intervals include
host gaps, not isolated GPU-active time. Some outer-wrapper/event observer
overhead remains unattributed; no speedup is inferred from summing intervals.

Two sequential interventions, maximum two one-A100/12-CPU workers at a time,
60-minute worker limits and six-A100 aggregate controller registration:

1. A request-private fixed source-route launch plan. Original validation runs
   EVERY call. Same Triton 3.4.0 compiled binary, constexprs, grid, arithmetic
   and flags; current runtime output pointers/strides are supplied every time.
   Never cache destination views or GPU tensors. Header/weight/phase metadata
   identity and code pointers must remain stable; ABI changes fail closed.
   Opaque global scratch is forbidden. Plan size is bounded to 128 routes;
   all runner handles are cleared at private request exit.
2. Lazy construction of genuinely unused prefix views, plus one bitwise sparse
   restore launch replacing separate protected-row scatter and correction
   launches. The original constructor already checks that corrections cannot
   touch protected rows. An explicit metadata certificate is set ONLY after
   that check; removing a subset of protected text rows preserves disjointness.
   Uncertified packets fail explicitly. Disjoint writes may run concurrently;
   integer decode, escape/cumsum logic and original bit values are unchanged.
   No new GPU metadata, buffers, source copies, offload or KV are introduced.

The original reader is the passed carry_literals path. Descriptor reuse and
the earlier generic compiled dispatcher remain unpromoted. Stage one compares
original versus prepared source launch. If it passes, stage two compares that
prepared anchor with prepared+restore; otherwise it compares original with
restore-only. Unsafe/incomplete results stop the whole chain. This is not an
unbounded search and no additional policy can be silently substituted.

Before model timing, synthetic GPU controls compare original/prepared source
projections at different lease strides and check runner cleanup. Restore tests
cover raw and packed bodies, partial final blocks, all 65,536 BF16 bit patterns,
disjoint sparse patches, and untouched history sentinels. Then every one of
the five methods in BOTH registered modes must reproduce original packet bits,
all 128 vocabulary logits, and independently audited initial/final/declared
decode-owned peak bytes. Timed fast counters bind to those separate audited
companions and are never relabeled physical peak measurements.

Each paired stage randomizes modes/methods, uses five warmups and twenty timed
repetitions, and profiles early/middle/late AFTER timing. Promotion requires
the unchanged familywise-adjusted EviCastKV end-to-end ratio upper bound <=0.99
on BOTH consumed inputs. If neither proposal passes, stop negative. If a policy
passes, apply it identically where applicable to all five adapters in the
existing complete-request benchmark. Report <=1.05 similarity and <=1.0 strict
no-slower gates separately, including negative outcomes and construction cost.

Finite afterany callbacks run primitive checks -> paired analysis -> at most
one restore follow-up -> conditional all-method timing -> terminal closeout.
No continuous queue/log polling and no automatic source editing while workers
enforce immutable source. All outputs write once with hashes, clean source SHA,
current scheduler IDs and scratch-only caches. Failed jobs still receive
completeness analysis. Larger/fresh quality campaigns remain blocked even if
an engineering timing gate passes; whole-request GPU peak is uncertified.
