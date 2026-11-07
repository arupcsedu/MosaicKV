# Device-resident per-token guard records

The readout guard's per-token CPU transfer creates a synchronization boundary
that storage-only does not have. This variant keeps the same native finite,
FP64 KL, maximum-logit and greedy-token tests on the GPU at every step. The
same GPU predicate selects the actual output. It copies each five-scalar
record into a bounded request-local 127-by-five FP64 buffer (5,080 bytes of
reported diagnostic activation backing, not K/V).

Before the final prediction is returned, all used records are transferred and
validated. Any nonfinite flag fails the entire request; no result is published.
Closing an incompletely validated guard also validates it, so forgetting the
explicit export cannot turn an invalid request into a pass. Unlike synchronous
validation, an invalid request may execute additional GPU steps before it is
rejected. This change in failure timing is explicit, not a relaxed threshold.

Do **not** substitute `torch._assert_async`. The installed PyTorch 2.8.0 docstring
warns that a failed CUDA assertion poisons the context and is unsuitable for
ordinary error checking. The negative operator case here must leave the context
usable for the following positive/lifetime control. No device reset or unsafe
exception suppression is used.

An auxiliary graph must retire before the original prompt pool's strict
RNG-owner audit, on successful and failed requests. The scoped wrapper always
runs that original teardown and then propagates validation failure; it does
not exempt a new allocation or bypass an existing ownership invariant.

Only the existing consumed-input, four-arm benchmark is permitted after the
structural-template numerical/storage run completes. Both hybrid arms use the
same graph templates and native readout handoff; only record transport differs.
All 128 frozen logit vectors, exact initial/final/decode-owned peak K/V bytes,
complete per-step guard records and the unchanged adjusted latency gates remain
mandatory. Eight warmups and 24 balanced paired repetitions are retained. This
is not fresh task-quality evidence or whole-request peak certification.
