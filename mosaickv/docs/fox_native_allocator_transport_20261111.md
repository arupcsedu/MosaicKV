# Bounded allocator transport control

This is an engineering diagnostic, not a paper benchmark or a replacement for
the original six-mechanism EviCastKV implementation. It preserves all prior
negative results and numerical thresholds. The positive lossless BF16 source
control and the original failing lossy source are different variants.

## Why another observer is necessary

The installed PyTorch 2.8 allocator's C++ `TraceEntry` carries `mempool_`; its
Python event serializer omits that field. Deferred Python owner observations
can also be ambiguous across address reuse. The existing typed analysis leaves
those generations unknown rather than guessing roles or reporting an exact peak.

The new recorder tests whether native numeric events can preserve allocation
domain identity and ordered requested-byte lifetimes. It does not assign KV
roles. A default pool is not proof of non-KV ownership, and a source-constructor
scope is not proof that every temporary is persistent KV.

## Fixed control and safety contract

- One initialized native caching allocator, one attached capture per process.
  PyTorch has no detach API. The recorder state therefore lives for the process.
- Fixed-capacity numeric storage. The callback runs under allocator locks and
  must not acquire the GIL, invoke Python/CUDA/PyTorch, allocate memory, or take
  another lock. Admission closes before the out-of-lock stop operation waits
  for writers and serializes. Overflow invalidates the capture.
- Two flat, non-nested synthetic pools. Nested pool precedence is not assumed.
- Actual tensor backing witnesses, aliases, address reuse, requested/completed
  free events, and complete final requested-storage reconciliation.
- Segment reservation sizes are not counted as live requested storage.
  A free request does not retire storage until the completed-free event.
- CPU regressions and the complete common-environment suite run before one
  one-A100/12-CPU/ten-minute capability worker. A finite `afterany` callback
  independently replays the saved events. No automatic model promotion follows.

The same source also registers a separate, two-worker CPU-only proof recovery
for 678 already discovered graph artifacts. It uses their exact pinned producer,
not filenames as proof, retains failures, imports prior proofs without rerunning
them, and does not restart the historical scan or materialize any cohort.

## Not established by this diagnostic

Even a passing transport control does not establish an exact source-KV peak,
vendor allocations outside PyTorch, complete historical media coverage,
constructed byte-matched cache fidelity, independent benchmark quality, latency
superiority, or useful six-component contributions. Those require their own
validated evidence. Original HF K/V allocation happens before `cache.update`;
whole attention and joint Q/K RoPE scopes cannot be labeled KV-only.

Installed API references:

- `torch/include/c10/cuda/CUDACachingAllocator.h`: `TraceEntry`,
  `CUDAAllocator::attachAllocatorTraceTracker`, and its lock/GIL warning.
- `torch/cuda/memory.py`: `MemPool`, `use_mem_pool`, `memory_snapshot`.
- Pinned PyTorch Git revision: `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`.
  `c10/cuda/CUDACachingAllocator.cpp` supplies requested-byte ALLOC/FREE events,
  calls attached trackers even without Python history, and attaches trackers to
  the existing initialized device allocators.

Canonical launch (after a clean scoped source commit):

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_allocator_transport_probe.py register \
  --root /scratch/djy8hg/workdir/fox-review-v18/run-20261111-1 --execute
```

All caches and compiler products remain in scratch. No packages or model
weights are installed; no benchmark identity or label is accessed by this probe.
