# Same-buffer next-layer prompt-K lookahead

This default-off **execution** variant retains the frozen precision/storage
parent, all K/V words, original positions, history, mandatory selections and
numerical checks. It does not validate the full six-mechanism algorithm.

## Motivation and transaction

The optimized consumed profiles show substantial steady-state host dispatch
and prompt reconstruction. Earlier prefetch allocated another full prompt K
buffer and copied it into the ordinary lease. Instead, the current graph reader
has ONE request-wide fixed-address buffer. After current V/PV is enqueued,
output projection and MLP no longer read that buffer. An auxiliary stream waits
on a recorded main-stream event and replays the **already captured next-layer
K graph directly into that same buffer**. The next layer waits on a producer
event before reading or writing it. No extra prompt copy or tensor buffer exists.

Only layers 1–31 within a decode update are eligible. Never prefetch from the
last layer across a token boundary, because history sealing/verification may
borrow the buffer. The first update follows the original capture path. Prepared
lossless code replay advances exactly once at the next layer's expected history
epoch; a prefetched K is consumed rather than replayed twice. V always follows
the original path. Pending/foreign/misaligned transactions fail closed.

GPU controls use A-B-A requests and exact K/V/output words with ragged prompt
lengths before loading checkpoint weights. Model controls require all 128
parent vocabulary-logit vectors bitwise, original packet hash, no fallback and
identical measured initial/final/decode-owned-peak tensor storage. Reusable
stream/events are driver resources, not certified whole-request GPU peak.

## Finite diagnostic

Complete CPU suite -> two one-A100 controls/screens -> CPU terminal analysis.
One GPU, 12 CPUs, 48 GiB, twenty-minute workers, within six aggregate A100s.
Same two consumed inputs only, no benchmark-label access, independent-quality
N=0. Five warmups/three randomized timed requests screen the original and
lookahead readers, including native-direct EviCastKV and BOTH PrefixKV readers.
Every arm uses event-sum accounting. No unoptimized baseline cherry-picking.

This screen cannot establish the 4% statistical latency claim: that still
requires twenty repetitions, multiplicity-adjusted upper paired ratio <=0.96
against every original/optimized baseline reader on both diagnostic inputs.
No larger quality campaign is automatically released. If overlap fails, retain
the negative result and original default; do not hide source construction costs.
