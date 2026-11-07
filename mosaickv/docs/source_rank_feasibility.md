# Source-level replacement feasibility, CPU only

Single-head overlays cannot retire a projection input needed by all heads.
The current actual zero-code replacement saves only tens of KiB, and repair
of a low-attention prototype did not recover the changed token. Preserve those
results. Before another GPU design, audit whether a source-level rank-limited
prototype representation can plausibly replace enough original source payload.

Low-rank KV reconstruction is established prior work; cite
[Palu (2024)](https://arxiv.org/abs/2407.21118). This study does not implement
Palu, decompose or modify its projection weights, or claim the principle is new.
It examines request-local normalized projection INPUTS from two already consumed,
hash-bound pretrained captures, without labels, training, model weights or oracles.

For every original layer, screen fixed ranks 96 and 128. The tail energy of
the FP64 source Gram spectrum gives the best unquantized rank-r source MSE.
If that optimistic bound is already worse than the existing three-bit source,
record rejection and DO NOT invent a quantized-factor result. Otherwise test
actual BF16 coefficients and a three-bit basis. Retain original text/protected
coordinates. A 1e-12 numerical slack is registered ONLY for this CPU source-MSE
question; no model KL/logit/token threshold changes. Require no worse source
MSE and smaller actual CPU payload before considering an eligible layer.

Also count the existing exact byte-stream codec, offsets and replay scratch
for both original and a conservative adaptive representation: use the smallest
eligible rank per layer, otherwise keep that layer's original source. Do not
quote raw payload savings while discarding the common codec. These are CPU
payload measurements, NOT GPU active KV, initial/final/peak, latency, answer
quality or an implemented six-mechanism backend. Exact/protected sidecars and
all reconstruction/history costs remain unmeasured here and unchanged in the
production variant. No cache format or model execution default is modified.

Run two CPU workers, twelve CPUs and 16 GiB each, thirty-minute bounds, then
afterany CPU analysis. Stop on a negative screen; do not increase ranks or
submit another GPU sweep. Even a positive screen requires a separate native
constructor/reader and pretrained numerical gates. No paper/fresh campaign
follows automatically.
