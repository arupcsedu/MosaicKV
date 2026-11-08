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
