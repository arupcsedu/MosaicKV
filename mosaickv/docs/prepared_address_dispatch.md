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

The Python-validation implementation was correct but its measured enqueue cost
was higher, so its operator gate blocked model execution. Preserve that negative
root. A separately versioned successor retains **all** per-call device,
gradient, live-owner and NULL-buffer checks in one CPU C++ binding; it does not
remove checks to obtain a speedup. Build/source/binary and Torch ABI hashes are
recorded, builds stay under canonical scratch, and the native source ships in
the package. Run the same operator stopping rule before considering model work.
The Slurm wrappers load the installed GCC 11.4.0 module; the system GCC is too
old for the locked PyTorch headers. Ninja is taken from the common Python
environment during compilation, without installing or altering packages.

After the operator and consumed-model gates, verify the identical cached-patch
transport on Prompt Top-K and each compatible local baseline before direct
paired timing. Both sides receive the same private-prefill optimization.
Failures stop the finite dependency chain; they do not authorize a fresh cohort.
