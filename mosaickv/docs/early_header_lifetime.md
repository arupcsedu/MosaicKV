# Early exact header lifetime: frozen consumed-input diagnostic

The late-packing study at `exact-scale-replay-20261108-v1` passed all 128-logit
bitwise checks on both consumed inputs, reducing initial/final persistent GPU
KV by 1,436,812 and 1,482,896 bytes. CPU residual storage was zero. It failed the
complete storage gate: requested PyTorch KV/workspace peak increased from
162.376203 to 166.159690 MiB and from 161.405039 to 161.422626 MiB. Its original
artifacts and negative promotion decision remain unchanged. Instrumented request
time is not a clean latency comparison.

The saved peak witnesses locate the maximum during prefill, before late header
serialization. Attention, native fitting, model/shared workspace, and accumulated
immutable packets overlap there. A later persistent saving cannot remove that
earlier peak. This diagnostic changes **when** an exact representation is held;
it changes no selection, model parameter, quantizer, original FP32 bit or guard.

Only completed async native/text publications may be archived. Replace both
prepared and native packet descriptors together, preserving shared source
identity. Never modify an in-flight source or use an empty scale in a model
reader. Before final publication completes, reconstruct the exact FP32 headers,
restore both descriptors and the cache, and invalidate its ownership catalog.
The existing downstream constructor and persistent lossless replay then see
their unchanged ABI. Charge all palettes, archives, reconstructed owners,
temporary overlap, CPU copies, and online encoding/transfer time. Any additional
owner retained elsewhere is still counted by the allocator ledger.

The scope is common to compatible affine-source transports, including the
storage-only comparator; exact raw baselines get no imaginary scale savings.
Default behavior stays unchanged. CPU tests cover shared ownership, original
header retirement, bitwise restoration, catalog invalidation, duplicate calls,
and request/packet identity rejection. Run the existing real-header CUDA operator
gate, then the same two consumed pretrained inputs, with complete-predicate-only
control versus early-plus-late exact headers. Keep all 128 logits bitwise equal,
the unchanged 32-token numerical gate, and the same requested-byte ownership
certificate. Require smaller measured initial, final **and** request-wide
KV/workspace peak before progressing. No micro-prototype, history, new cohort or
paper campaign is authorized by an execution-only success.

The finite CPU → operator → model → `afterany` analysis chain uses at most two
one-A100/12-CPU workers and the common aggregate six-A100 admission controller.
If this placement still fails peak reduction, do not promote it or pretend that
a smaller persistent packet certifies a whole-request improvement.
