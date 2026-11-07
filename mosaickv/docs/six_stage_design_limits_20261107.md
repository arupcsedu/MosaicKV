# Six-stage design: what the current measurements can and cannot establish

EviCastKV and MosaicKV name the same system. The current experimental hybrid
adds a final-layer, single-head precision overlay to the validated storage
reader. It is not an implementation-level validation of original all-layer
KV-slot compaction. Keep all six explicit mechanism/ablation labels.

## Why the current architecture cannot produce an adaptive storage win

Let S be the common serialized prompt source, H the unchanged history, W the
reader backing/workspace, and E/P/R the overlay's extra exact, prototype and
GPU repair state. With the same common codec and no shared-source retirement,

    hybrid decode-owned KV = S + H + W + E + P + R

The parent already owns S, H and W. Adding E/P/R cannot reduce its backing.
The measured overlay is +6,713 GPU bytes and +1,024 pinned CPU residual bytes
on each of the two timing inputs. This is not a claim about every input.
Source-code repacking saved approximately 4.1 MiB on those inputs, but that
lossless optimization applies to storage-only too. It must not be credited to
the adaptive overlay or used as an unequal-control superiority comparison.

The live parent guard requires the same greedy token each update. When the
parent counterfactual is exact and isolation holds, this preserves the
parent's entire caption by induction. It cannot improve benchmark quality
over that parent. Historical unguarded component results are not current
guarded-variant quality evidence. No natural repair in a small panel is not
proof of useful repair recovery. Forced veto/restoration proves mechanics only.

## Correctness before another timing claim

The broad consumed counterfactual check found rare numerical differences in
the single-head shortcut even though synchronous/device records agreed. Its
two numerical-failure inputs passed full-head counterfactual parity at every
prefix. Use the full-head path for the next audit. Do not loosen the registered
FP64 KL/logit/token conditions or silently reuse the faster single-head times.
Keep negative diagnostics and all consumed identities. Whole-request KV peak
remains uncertified; decode-owned backing is a narrower measurement.

## A source-owner transaction, not another additive head overlay

Any future adaptive memory-reducing design needs an explicit ownership group:
all K/V heads and both projections consuming a serialized source tile. Graph
nodes may remain layer/head-local KV blocks, but retirement must commit across
the complete source-consumer group. A transaction would:

1. Forecast query demand without benchmark labels and pool demand across the
   source's actual consumer routes, preserving rare per-query demand.
2. Use graph relationships to propose coherent source groups, not merely a
   fixed visual quota or isolated head-local replacements.
3. Compare output-impact estimates and actual short-horizon model controls;
   reconstruction scores alone have already proved unreliable here.
4. Allocate one combined byte budget including prototypes, mappings, numerical
   corrections, exact protection, history and decode workspace.
5. Retire a source owner only after every consumer has a valid replacement.
   Keep per-position RoPE: never average incompatible post-RoPE keys.
6. Charge CPU residuals and GPU repair staging; a promotion/eviction transaction
   must remain within the same lifetime budget and preserve isolation.

This is a design constraint and proposed research direction, not a measured
improvement or authorization for another large campaign. A bounded CPU storage
feasibility screen and same-consumed-input numerical intervention must succeed
before independently validating quality. It may fail: aggregate consumers can
erase apparent head-local savings and residuals may cost more than they save.

## Immediate execution-only work

The currently justified changes are exact and default-off: shared native
readout, device-buffered diagnostic records with mandatory prepublication
validation, and hash-identical metadata serialization. They reduce overhead
without promising new quality or adaptive KV savings. Stop promotion when
numerical, memory or strict adjusted latency controls fail. Do not launch
larger cohorts merely because code executes successfully.
