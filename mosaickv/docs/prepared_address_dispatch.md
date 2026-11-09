# Bounded same-binary CUDA argument transport

Installed Triton 3.4.0's NVIDIA launcher explicitly accepts integer uint64
addresses; its Tensor branch calls data_ptr and cuPointerGetAttribute for each
pointer. Pin the exact driver source SHA before enabling this default-off path.
The first real compilation/launch still receives all original tensors. Replays
still recover all live private tensors, verify CUDA device/gradient contracts,
read their current actual data_ptr values and retain existing request-owner,
phase, geometry and stride validation. Only the redundant driver's pointer-
attribute query is avoided. No new kernel or GPU pointer table is introduced.

Apply the same cached sparse-patch dispatch to compatible baseline/storage-only
paths. The mixed-source ABI additionally benefits when that format is present.
No weights, values, source precision, numerical limits or positions change.

CPU ABI/lifetime/isolation tests -> actual source/patch CUDA words and sentinels
-> a preregistered bounded 20-repetition host-enqueue improvement gate -> only
then the same two consumed pretrained inputs -> conditional direct randomized
timing. The unchanged all-128-logit bitwise guard, initial/final byte equality,
complete ownership and whole-request scope remain mandatory. Epoch-predicate
batching was timing-neutral, so this candidate retains synchronous checks.
Fresh cohorts and paper-scale experiments remain blocked.
