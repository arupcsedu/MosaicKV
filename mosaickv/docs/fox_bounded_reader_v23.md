# V23 bounded native reader — consumed-input engineering gate

EviCastKV and MosaicKV name the same project. This explicitly labeled storage
variant does not activate the six research mechanisms. Historical V20–V22
source3 failures, raw transport controls, and original artifacts stay immutable.

## Why this reader differs

The former packed attention used FP64 reductions rather than native BF16 BMM
rounding. V23 retains the complete token axis in native QK BMM, FP32 softmax
cast to BF16, and PV BMM. Only independent heads are partitioned. Two fixed
2048-by-128 BF16 component buffers occupy 1,048,576 bytes, reused across heads,
layers, and requests. B=1 dispatch is excluded. Unsupported geometry fails.
No full-layer K/V reconstruction lease is permitted.

Original normalized BF16 projection inputs are stored once per layer. Native
head-width projection and the original post-RoPE phases reconstruct K/V into
the head workspace. Every BF16 discrepancy against original prefill K/V has
an explicit sparse bit patch, including indices and values in storage costs.
Functional raw-head caching bypasses projection for those heads and retires
their patches. It is not unused alignment or baseline padding. At least one
source head remains per layer so normalized-source ownership remains useful.

This representation cannot claim losslessness at the old 49 MiB source3 cap.
The old cap is reported unsupported, not silently redefined. A new diagnostic
finds an exact intersection between **actual** source/cached-head bytes and
the raw dense uniform-capacity lattice. It must construct and remeasure both
representations; an arithmetic witness alone is not an acceptance artifact.
Search failure is explicitly unsupported, not a proof of mathematical
infeasibility. No model outputs enter this budget search.

## Registered bounded verification

Only the same two consumed Description/MM-Vet inputs are authorized. There is
no dataset rematerialization, fresh cohort, calibration, or paper campaign.
Pinned LLaVA-1.5-7B, original revision, BF16/eager, seed13, deterministic CUDA,
and `CUBLAS_WORKSPACE_CONFIG=:4096:8` remain unchanged.

1. Complete CPU suite and CUDA operator parity at N98/375/600/604/727/731.
2. Two one-A100/12-CPU workers, each at most15minutes, only after both pass.
3. Original-source reader:128 predictions against immutable original FullKV.
4. Raw selected reader:128 predictions against native **same-selected** K/V.
5. Retention1 raw reader:128 predictions against original FullKV.
6. Require all128 greedy predictions, maximum logit difference<=0.001, and
   report bitwise equality. Original32-token KL/logit guards remain unchanged.
7. Require exactly equal initial, final, and observed decode-owned peak bytes
   for the constructed raw/source controls. Count packet payloads, phases,
   patches, cached raw heads, logical-position buffer, fixed KV workspace,
   raw history, old/new history-growth overlap, and incoming update owners.

The all-valid request-private transport uses no redundant physical-length
mask buffer; original logical positions are retained. Other requests keep the
ordinary HF path, and exception cleanup restores all hooks and dispatches.

## Explicit accounting limits

This is a **post-construction decode-owned tensor-backing** gate, not an exact
whole-request KV certificate. Original verification prefill/captures, packet
construction overlap, model weights, attention scores/probabilities/output,
opaque vendor workspace, and allocator reservations are excluded and must not
be called saved KV. CPU position metadata is separately reported. Necessary
source reconstruction scratch is included in the fixed KV workspace, not
reclassified away. Whole-request certification and independent quality remain
open, regardless of this diagnostic outcome. No latency/superiority claim is
authorized. Slurm uses finite `afterany` callbacks, not continuous log polling.

Launch from a clean initialized worktree after sourcing scratch cache paths:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_bounded_reader_v23.py launch \
  --root /scratch/djy8hg/workdir/fox-review-v23/run-20261111-1
```
