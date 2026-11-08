# All-head source prototypes and time-aligned ownership

This is an isolated engineering diagnostic, not a promotion of the validated
storage-only baseline and not an assertion that all six research mechanisms
now improve quality. Historical source and single-head overlays are unchanged.

## Architectural transaction

One original 3-bit affine source supplies all 32 K and 32 V projections. An
additive single-head prototype cannot retire that source. Instead, substitute
one **source-domain** prototype group across every channel, before projection
and before original-position RoPE. A complete original-to-physical map retains
every logical position. Original FP32 quantizer headers and exact K/V protection
are preserved. Only positions sharing one affine quantizer group may merge;
original text/mandatory positions cannot merge. Equal FP32 code barycenters are
rounded to representable codes, an explicitly lossy implementation choice.

The returned object has no original code tensor or storage view. Original
source-group codes reside in separate pinned CPU storage. A controlled repair
reconstructs the original complete code packet and retires the prototype packet;
every head is repaired together. Captured replay mutation is explicitly rejected:
online repair needs an uncaptured transaction and private reader/codec rebind.
The controlled repair is **not** evidence of natural model quality recovery.

The bounded pretrained diagnostic uses one eight-position visual group per
layer, chosen by actual last-16 prompt demand across heads times affine-source
distortion. It is NOT labeled as the six-component submodular planner. This
tests the missing ownership primitive before integrating a more costly planner.
Every layer has the same physical row count, preserving the existing adjacent
layer codec geometry. Compare complete encoded storage INCLUDING maps, headers,
codec payload/offsets, replay, CPU residuals and decode workspace, not raw row
savings. Retain storage-only when complete savings or model fidelity fail.

The separately registered `first_layer_shared` diagnostic freezes the first
layer's label-blind group for all subsequent layers and shares one actual GPU
map owner. This preserves inter-layer source-row alignment; independently
selected maps can enlarge the modulo-8 residual codec despite fewer raw rows.
This is a new storage-policy diagnostic, not a reinterpretation of the first
negative storage result. No answer/annotation influences this change.

## Registered controls

Use only the same two consumed COCO identities and pinned LLaVA-1.5 checkpoint
from the preceding fused-fill diagnostic. No new benchmark labels are accessed.
Preserve BF16/eager, processor, original 603 logical positions and 128 greedy
predictions. First 32 predictions must retain greedy-token agreement, KL <=0.02
and maximum logit difference <=1.0. Control logits are bitwise against their
frozen reference. Do not relax thresholds on failure. No fresh/cohort/paper-scale
follow-up is authorized by this plan. Failed approximations remain negative.

## Whole-request ownership

`measurements/kv_timeline.py` reduces the actual single-device CUDA allocator
event order across threads/streams. Allocation address plus generation is its
identity. `free_requested` remains live until `free_completed`; aliases count
once. Semantic classifications distinguish model, KV persistent, KV workspace,
shared KV workspace, non-KV and unknown. The timeline observer retains no tensor
objects. Review exact source-function stack rules, not broad module/path rules.

The installed PyTorch internal APIs are `_record_memory_history` and `_snapshot`
in the common environment's `torch/cuda/memory.py` lines 864--1038. They cover
PyTorch allocator ownership, not arbitrary vendor allocations or process HBM.
Capture overflow, missing allocations, conflicting roles, incomplete requests,
unknown generations and opaque unclassified workspaces block certification.
The whole-request audit starts before media loading/processor and ends after
detokenization/retirement/synchronization. It is excluded from clean timings.

The audit-only retry records tensor-owner addresses with host timestamp
intervals, then resolves them to allocator generations using `time_us`. Snapshot
markers must verify the clock domain; absent/ambiguous timestamps fail closed.
No tensor is retained and no address inherits a prior generation's role. This
avoids repeated whole-history snapshots inside each asynchronous layer.
Release cached cuBLAS workspaces before this excluded audit; release them again
at completion and attribute the actually retired owners to shared projection
workspace. This is a measured lifetime transaction, not baseline subtraction.
Clean timing requests are unchanged. Generic Linear outputs are classified by
actual module identity, distinguishing K/V projections from other activations.
RMSNorm line-specific rules require the installed source file hash in the plan.

The next audit revision uses explicit, ephemeral CUDA allocation beacons for
clock calibration instead of assuming snapshot markers refer to the current
snapshot. It intersects their measured host/allocator offset intervals and
requires each ownership observation to fit uniquely for the entire interval.
Allocation-generation lookups are indexed by address and time, not quadratic
scans over repeatedly reused buffers. Physical segment release/unmap events
also retire fully covered private-pool allocations; partial overlap fails.
Original source receipts bind all 32 heads and both K/V components to an actual
allocation generation and require physical retirement before decode starts.
Constructor-owned attention probabilities are classified as shared model/planner
workspace, rather than silently excluding their lifetime from the KV audit.

The consolidated audit replaces only legacy weak-reference result scanning;
numerical/stride/protection checks still run. Its old phase counters are marked
inadmissible, not silently reused. A fully classified interval certificate is
recomputed from serialized allocation generations before propagation to the
request coverage record. It cannot be accepted from Boolean flags alone.

CUDA trace `alloc.size` is **requested tensor-storage bytes**, not allocator
rounding. Baseline/final snapshots therefore use `requested_size` too, matching
the active-cache `UntypedStorage.nbytes()` convention. Allocator padding,
reserved segments, non-PyTorch vendor workspace, and process HBM are separate;
the conservative allocator envelope is retained. Earlier mixed-unit operator
peak certificates are superseded, NOT upgraded; their projection/repair parity
observations remain valid. A one-byte clock beacon is matched as one byte.
Collect unreachable request owners and flush completed allocator events at
audit boundaries so snapshot conversion cannot hide delayed retirement.

The installed `c10/cuda/CUDACachingAllocator.h` defines `SnapshotInfo`, `SegmentInfo`
and `BlockInfo`. The read-only C++ helper returns the identical active block
addresses, requested sizes and rounded sizes without symbolizing unrelated
device traces. Verify equality to `torch.cuda.memory_snapshot()` before use;
retain the unchanged strict two-scalar graph-metadata guard. Build this helper
on a CPU job first; compilation failure blocks GPU submission. Build products
are content-addressed under the scratch cache, never installed into the common
environment. No model weights, attention operators, or generation path change.

A fully classified native operator transaction can certify its **bounded** peak;
that must never be relabeled whole-request. The model capture records unresolved
allocations and keeps its exact request certificate false until every required
classification is supported. No sums of phase maxima or arbitrary allocator
baseline subtraction can substitute for this proof.

Run CPU tests, then two one-A100/12-CPU workers (one-hour bounds), followed by
an `afterany` analyzer. The GPU admission controller enforces the aggregate cap.
No continuous log tailing. Main worktree, paper sources and old evidence remain
untouched. This diagnostic cannot establish statistically independent quality
preservation or latency superiority.

## Source-bound CPU ownership review

`review_shared_proto_ownership.py` derives new write-once certificates from saved
allocator generations; it never edits predictions, original reports, lifetimes,
or timestamps, and never submits GPU work. Supplemental rules distinguish the
image projector/embedding/FFN/logit activations from KV representations and
reconstruction. Language attention scores/probabilities, causal masks, RoPE and
custom attention workspaces are conservatively charged as shared KV workspace
for BOTH arms. Unknown sources and missing/overwritten histories still fail.

An ambiguous deferred timestamp observation assigns NO role and cannot establish
an owner identity. If every generation has independent reviewed source-stack
attribution, the rejected observation is unused and may be superseded; its
original blocker and error details remain in the derived record. No ambiguity
about a source retirement receipt is tolerated. The certificate recomputes the
maximum concurrent bytes over all allocator intervals, not phase maxima.

Bind the producer plan, inputs, manifests, raw rows, logits, native-control
receipts and exact source blobs/hashes. Additional upstream source is checked
against the producer's package version and installed wheel RECORD; disclose
that historical captures did not separately hash every supplemental file.
Future preparation pins those files before capture. The numerical guard,
source prototype policy and generation settings are unchanged.

The audit-only analyzer now counts reused canaries before checking eight-row
completeness and memory pairs; the original incomplete derived report is
preserved. These fixes do not require rerunning numerical inference. A complete
certificate does not mean a smaller peak: construction overlaps and retained
workspaces can make the prototype variant's whole-request peak higher, even
when its initial/final GPU packet is smaller. Pinned CPU residuals must be
reported separately, and GPU-plus-CPU savings must not be inferred from GPU
packet savings. Audit wall time cannot be used as clean latency evidence.
