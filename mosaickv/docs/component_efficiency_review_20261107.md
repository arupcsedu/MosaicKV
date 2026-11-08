# Component efficiency review: redesign before removing mechanisms

This review preserves the six-component implementation and validated storage
reader. EviCastKV is the paper name for MosaicKV. Any reduced-component path
is a separately labeled variant, not a silent new definition of the full method.
The user's November 7 request permits retiring unhelpful mechanisms only after
testing whether they can be used more efficiently. No fresh identities or large
campaigns follow from this engineering review.

## What the completed evidence says

The retained single-head precision overlay adds 6,713 GPU bytes and 1,024 pinned
CPU bytes on each timing input. It does not retire the all-head source owner.
Its live-parent token lock preserves the parent's answer, rather than improving
benchmark quality. The first-update policy never repaired naturally in the
small completed panel. Forced restoration/veto controls validate mechanics,
not answer recovery. Keep those failures and the historical unguarded results.
The 24-pair request benchmark found the retained hybrid 3.6--4.0% slower than
the same-codec storage parent. All observations are consumed development data.

## A better role to test for each component

| Component | Observed limitation | More useful bounded role | Progression condition |
| --- | --- | --- | --- |
| Forecast | Averaging can hide rare queries; projected attention is not answer correctness | Reuse individual observed queries for sensitivity/precision allocation; no duplicate q_proj when captured data suffice | Measurable downstream effect, not only better reconstruction |
| Evidence graph | 143 ms profiled for 603 nodes; no historical CIDEr gain | Use small source-tile candidate neighborhoods; build a graph only when net replacement is feasible | Complete source owners can be retired and graph improves the allocation versus a graph-free control |
| Value utility | Norm/error magnitude can choose harmful restoration | Signed output impact and complete net byte costs, rather than additive exact protection | Independent improvement over an equal-byte simple selector |
| Budgeted selector | A positive head-local utility is not a positive whole-source saving | Choose complete ownership transactions, including history/workspace and metadata | Measured initial/final/peak backing fits; no padding or shared alias is omitted |
| Prototypes/residuals | A single-head KV prototype cannot remove a shared source row | Lossless all-consumer source dictionaries first; lossy source prototypes only after actual cost/causal feasibility | Bitwise source reconstruction and net physical packet savings before GPU work |
| Uncertainty repair | No natural benefit; first-step K snapshot is 5.67 MiB | Do not attach a per-token parent guard to a lossless representation; reject infeasible lossy plans before allocating repair state | A genuine failed prediction is recovered under a bounded, fully charged transaction |

No row above is an established quality gain. Removing a component from the
default storage variant does not remove it from the research implementation.
Graph-free/source-only variants must report the disabled mechanisms explicitly.

## First new test: source dictionaries, not another additive KV overlay

The source codes are the parent's original three-bit quantizer symbols.
Four uniformly selected original source rows form a per-layer dictionary.
Each source row chooses the dictionary row offering the smallest encoded
palette/escape cost. Every 512-symbol group can still use direct coding, the
original previous-layer predictor, or raw storage. The common all-head palette
codec and the legacy byte codec compete per layer, exactly as in the actual
runtime. The new dictionary must beat that minimum, not just one codec.

The proposed packet charges every raw anchor, uint8 row-to-anchor index,
uint32 offset, codec selector and payload byte. It is selected only when its complete packet
is smaller. All original source bits, feature order and positions survive.
The decoded row feeds every original K/V projection and every head. Dictionary
prediction therefore avoids post-RoPE averaging and requires no parent-token
guard to establish representational equality. This is a common lossless codec
variant, not proof that future-query forecasting or uncertainty repair helped.
Apply it to the storage-only comparator as well if it progresses.

Two CPU workers use only saved code captures from the consumed Description and
MM-Vet diagnostics (not the separate COCO timing images),
all 32 layers, no model weights or labels. Require exact roundtrip and at least
2% reduction in **serialized source-code** bytes on both before native-operator
work. This is not a claim about total KV or whole-request GPU peak. A packet
failure must retain the current codec; do not start another quality sweep.

If this test is negative, the justified simplification is to keep the
validated source-only path for performance claims and the six-stage overlay
as experimental. Do not retain overhead merely to make all six boxes active.

## Bounded second use of the graph/selector idea

Uniform anchors failed: all 64 source streams roundtripped, but no dictionary
beat the actual common codec. Do not implement its native reader. A separate
CPU-only screen uses a sparse bipartite source-to-anchor graph with at most
16 candidate rows, and greedy facility coverage of actual residual-byte gain.
An anchor must pay for its raw row and the first row-ID map before admission.
This tests a physically relevant use for graph/selection rather than semantic
coverage with no source retirement. The final complete packet still must beat
the best byte/symbol codec. It may reject every anchor before construction.

The fixed diagnostic layers are 0, 8, 16, 24 and 31 on the same two consumed
Description/MM-Vet captures. The original adjacent layer, not the preceding
tested layer, remains the predictor. A positive subset screen would require
all 32 CPU layers before any GPU operator. There is no quality fitting, no
new sample, and no automatic large experiment. The source-only hook-free
variant is tested separately; it does not pretend these graph experiments
establish an all-six contribution.
