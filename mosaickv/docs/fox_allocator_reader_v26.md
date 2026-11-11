# V26: actual allocator blocks, not just tensor backing

## Corrective snapshot-schema evidence

No-weight A100 diagnostic22820834 at1cd25eb captured an inactive free block
with size3,473,408bytes and historical requested_size8,388,608bytes. It
intersected **none** of the required owners. Its full snapshot and required
owner ranges are immutable under
`/scratch/djy8hg/workdir/fox-review-v26-snapshot/run-20261111-1/capture/`.
PyTorch2.8's [allocator source](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/c10/cuda/CUDACachingAllocator.cpp)
shrinks free remainders without resetting their historical request; snapshot
exports that field even for inactive blocks. The parser therefore ignores only
inactive blocks' stale requested-size upper bound. Physical address/size
geometry remains strict, required inactive owners remain rejected, and active
requested-size/range checks remain unchanged. Five regressions preserve those
distinctions. Never clamp a live block or infer its allocated size.

## Passed foundation

V25 at26a33a6 passed128/128 and bitwise logits for both consumed inputs and
all three registered controls, and constructed source/raw initial/final/decode
peaks matched exactly in tensor-backing bytes. Preserve that successful result.
It does not establish whole-request or allocator-block equality.

The follow-up audit identified a concrete allocator concern: separate K and V
allocations at odd head lengths can round independently. V26 packs each raw
head into ONE exact-size `[2,N,128]` BF16 owner with disjoint contiguous K/V
views. Every byte is used. Its payload is512*N bytes; selections and values do
not change. Strict validation rejects overlaps, cross-head aliases and unused
tails. This common storage improvement applies to compatible raw controls; it
is not an algorithm-specific baseline penalty or dummy alignment payload.

## Registered allocator controls

Pin BOTH allocator environment aliases before Python, identically for every
method: `backend:native,expandable_segments:True`.
Require actual native backend and measured expandable segments; a string alone
is not evidence. Unsupported environments fail closed. Reuse the SAME8.25MiB
component workspace and logical-position buffer across the three isolated arm
runs, rather than reallocating equivalent pools with different fragmentation.

At **every** explicit cache-ownership observation, resolve each strong tensor
owner's pointer and complete requested range against an actual CUDA allocator
snapshot. Charge the unique active block size, not a rounding formula, reserved
segment size, or view payload. Include shared phases, source/raw payloads,
patches, workspace, position metadata, incoming K/V, histories and old/new
history overlap. Reject missing, inactive, cross-device or out-of-range owners.
Record token-update/layer/growth phases; retain replayable ownership deltas and
independent initial/final/peak reductions. Tensor-backing and allocator-block
bytes remain separately named.

The no-weight operator canary runs BEFORE admitting the full CPU suite; a
failed launch/configuration therefore cannot waste a full suite or admit model
workers. PyTorch2.8 rejects an explicit roundup-power2 setting of zero; the
corrected configuration leaves the native default in effect. The canary measures100 allocator observations under a
conservative synthetic metadata load, then extrapolates to4322 observations
times3 arms. Its registered instrumentation budget is360seconds. Only passing
native arithmetic, block mapping, overhead and full CPU gates may release two
one-A100/12-CPU/15-minute workers and afterany analysis. Use the SAME two
consumed inputs and unchanged128-token/logit/KL gates; no fresh identities,
calibration or paper campaign. Instrumented timings are NOT latency evidence.

## Scope limits remain explicit

This maps **declared post-construction decode-owned KV** at every recorded
ownership boundary. It is stronger than allocation witnesses or an allocator
envelope, but not a certificate over prefill or source-construction lifetimes.
Original verification prefill/captures, construction overlap, weights, ordinary
attention activations, opaque vendor workspace and reserved memory remain
outside the declared cache boundary. No whole-request, superiority, six-stage
or old49MiB-lossless claim follows. The new lossless-source packet budget is
explicitly different from the earlier lossy49MiB family.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_bounded_reader_v26.py launch \
  --root /scratch/djy8hg/workdir/fox-review-v26/run-20261111-1
```
