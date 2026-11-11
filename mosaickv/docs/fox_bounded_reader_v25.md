# V25 full-width native projection with one bounded component pool

V24 is a negative **construction** result, not128-prediction evidence. Its
head-width GEMM changed projection arithmetic so extensively that original-KV
patches pushed source storage to487,288,428 and488,803,158bytes, above original
raw FullKV. Both workers correctly stopped before decoding; no fake byte-match
or larger-cap waiver was accepted. Preserve those outputs unchanged.

## Corrective representation and reader

Store original normalized BF16 input per layer and original shared RoPE phases.
Use the native **complete output-width** GEMM, not head-width projections. Its
output writes into one fixed1024-token-by4096-channel BF16 component pool.
Apply K RoPE with a single1024-by128 BF16 head scratch. The constructor compares
every K/V word with original prefill and charges any sparse correction patches.
No claim assumes those patches are zero: they are measured.

Consume K into native scores, then overwrite the SAME component pool with V.
Native batch32 zero-stride QK/PV and the full32-head native softmax retain the
successful V24 CUDA dispatch. No full K+V pair is materialized simultaneously.
The two necessary fixed buffers occupy **8,650,752bytes**, charged to both
representations. This is larger than V24's unsuccessful512KiB workspace; it is
not hidden or described as a lower-workspace improvement.

## Actual constructed byte matching, without dummy fill

The source phase cost does not lie on the coarse uniform-all-heads raw lattice.
V25's raw control instead owns each selected K/V head separately, with balanced
per-head capacities differing by at most one token. One real BF16 head-token
costs512bytes. Store **no raw prompt padding**. The largest real balanced cache
at the source's measured packet target is built using prompt-only Top-K scores,
the unchanged mandatory policy, and original K/V words. If actual source costs
do not lie on this finer lattice, reject the match rather than round or add
unused data. The byte search never reads outputs or labels.

Temporary head gaps exist only in the common component workspace, are charged
there, and receive explicit invalid-position masks. Generated history follows
the shared prompt maximum and stays valid. The independent native HF control
uses the same selections and explicit per-head validity masks in a padded
**verification-only** DynamicCache; its padding is not presented as baseline
storage. A typed request-private internal-mask contract prevents an all-valid
claim for ragged data. Other adapters/requests retain their ordinary dispatch.

## Bounded verification and limitations

Same two consumed inputs, pinned BF16/eager LLaVA, deterministic CUDA,128
predictions, original logical positions, maximum logit difference<=0.001,
all128 greedy tokens, and original32-token KL/logit guards remain unchanged.
Full CPU suite and CUDA component-operator parity must precede two one-A100,
12-CPU,15-minute workers. Afterany analysis independently verifies saved logits,
immutable hashes, and actual initial/final/observed decode-owned peak storage.
Every failure remains in JSONL and completion records. No continuous log poll.

The source/raw controls use the same bounded transport and physical cache
boundary: packet backing, phases/patches, logical-position buffer, histories,
old/new growth overlap, incoming K/V, and both workspace buffers. CPU position
maps are separate. Verification prefill/captures, constructor overlap, weights,
attention activations, opaque vendor workspace and allocator reservations are
excluded. **Whole-request KV peak remains uncertified.** This is not a paper
claim, fresh quality validation, latency win or six-component validation. The
old49MiB lossy-source result remains negative/unsupported; the new lossless
diagnostic budget is explicitly different.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_bounded_reader_v25.py launch \
  --root /scratch/djy8hg/workdir/fox-review-v25/run-20261111-1
```
