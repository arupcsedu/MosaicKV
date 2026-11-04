# Private prompt-reader replay: strict four-percent engineering target

The November 4 all-five-method predicate-span run is a complete negative:
MM-Vet remains 0.23% slower than the fastest PrefixKV storage control.
Preserve all previous results. These are two consumed inputs and local
precision/storage variants, not new benchmark or paper evidence. All six main
MosaicKV mechanisms remain explicitly disabled in this execution diagnostic.

## Intervention and hypothesis

Construction-only changes cannot plausibly save another 4% of an approximately
11.4-second request. Repeated prompt reconstruction still traverses Python and
launches its unchanged kernels thousands of times. Capture **only** those
immutable prompt-reader operations into request-private CUDA graphs. Native
eager attention, ordered adjacent-layer code replay, growing history and
logical/decode positions remain outside capture. The existing whole-model
graph rejection remains in force. No attention arithmetic, K/V payload,
quantization, retained membership or numerical tolerance changes.

One fixed-address BF16 lease is shared serially by all layers and components.
Charge its complete allocated capacity, including unused positions, once in
the shared backing inventory. Allocate lazily after the initial boundary and
retire it after decoding but before the final boundary. Peak accounting must
include it throughout its lifetime. Register `predictions - 1` maximum updates
and reject overrun. Reject overlapping leases, changed bindings and any
nonzero private tensor allocation during capture. Driver graph metadata is
not certified whole-request KV ownership; record CUDA allocated/reserved peaks
and do not claim a certified whole-request memory win.

## Fair controls and finite execution

All five local methods receive the same replay option. Preserve their original
fast readers too, including the PrefixKV whole-reader control. A faster reader
cannot be hidden to manufacture a win. Capture, warm kernels, instantiate,
synchronize and retire graphs **inside** every measured full request. No graph
or replay scratch survives a request. Original and replay arms use identical
inputs, weights, BF16 precision, eager attention, 128 predictions, history,
GPU controls and immutable parent packets.

Run full CPU regression, real CUDA synthetic replay/isolation/accounting
controls, then both consumed model controls for every arm. Require bitwise
packet signatures, all 128 parent vocabulary-logit words and identical initial
and final persistent backing. Report the newly measured decode-owned peak,
including the fixed workspace; do not assume view bytes or old peaks.

Only then run five warmups plus twenty randomized paired full requests per
arm. The primary gate is the multiplicity-adjusted upper paired ratio <=0.96
against **every** registered baseline reader, not just its original reader.
Use 10,000 bootstrap replicates and a fixed 64-contrast family across two
inputs and two timing stages. A passing two-method gate releases only the
bounded all-five-method extension. Every producer gets `afterany` analysis;
failures cannot silently disappear. At most two one-A100/12-CPU/48-GiB workers
run concurrently, reservation `bi_fox_dgx`; each shard is under two hours.

No fresh materialization, independent quality, calibration or paper campaign
is authorized by this chain. Independent benchmark-quality N=0. Passing this
diagnostic does not establish superiority of the complete EviCastKV algorithm.
