# Shared lossless-storage feasibility, not selector superiority

## Why a separate direction

The saved delimiter and donor experiments are negative consumed-input
diagnostics. Donor masking worsens Top-K mean KL, while EviCastKV effects are
mixed across prefixes. Joint-only failures remain after donor loss is isolated.
Thus neither a larger delimiter reservation nor a weakened guard is justified.
Preserve those negative records and their identities.

This separately labeled experiment changes storage rather than evidence:
encode **the unchanged saved BF16 K/V values** without losing any bit. It adds
no selected position, eviction, prototype, fit, quantization, draft query or
model update. Apply it to BOTH Top-K and EviCastKV. It cannot establish an
algorithmic advantage over an equally optimized baseline.

## Codec and cache lifetime

For each group of 1,024 BF16 values, store sign and mantissa in one byte per
value. Encode the exponent in three bits: seven consecutive exponent values
are represented relative to a one-byte base; the eighth code refers to a
literal exponent byte. Choose the base by the densest seven-exponent interval
with lowest-base tie breaking. Store per-group literal offsets explicitly.
The decoder uses only integer operations and reconstructs all original bits,
including signed zeros, infinities and NaN payloads. If encoding would expand a
tensor, store it raw. No model-output or label information chooses the format.

Build one selected source layer at a time. During decoding, unpack only the
current layer into temporary full BF16 K/V buffers. Run the original eager
attention with the original values/order and original RoPE/logical positions.
Release that layer's materialization immediately after its attention call.
Keep generated history raw, append-only, and in the same 16-token growth chunks.
The feasibility transport fails explicitly beyond 32 decode updates.

The CPU constructor and GPU integer decoder are experimental and are not
enabled by the normal MosaicKV CLI. CPU packing is expected to add substantial
request overhead; no latency improvement is promised.

## Controls and stopping rules

Use only the same consumed Description `000000532947` and MM-Vet `v1_197` and
the two frozen original per-head selections. Bind all inputs and the negative
donor audit by hash. Run:

1. exhaustive GPU bit-pattern and irregular-shape decoder checks;
2. ordinary FullKV generation/reinjection parity (16 tokens);
3. a raw version of the same layer-at-a-time transport;
4. lossless storage using the identical selection and 32 saved FullKV prefixes.

Both raw and encoded transports must reproduce every saved anchor logit
bitwise. The existing 32-prefix KL/logit/token guard and absolute limits stay
unchanged. Lossless storage retains any existing anchor errors; report absolute
FullKV fidelity separately rather than converting anchor parity into a quality
pass. Verify source immutability, original logical positions, all 31 history
updates, no mandatory eviction, no residual/prototype transformation, and no
outstanding temporary layer buffers after a forward.

## Storage and timing boundaries

Count unique backing allocations of all encoded component tensors, raw
fallbacks, raw histories, history-growth overlap and temporary full current-layer
K/V. Compare encoded initial/final/decode-transaction peak against BOTH the raw
transport control and the saved original transport. Do not manufacture a win
by comparing only to a newly inflated control.

This scope certifies only post-construction live decode-cache backing. It
excludes the inactive original prefill source, constructor workspaces, model
weights, attention activations and allocator reserve. **Construction and
whole-request/HBM peaks remain uncertified.** Record CUDA allocated/reserved
peaks separately; these include model/source/activation memory and are not the
KV accounting metric. Record CPU packing, synchronized CUDA-event decode time
and wall time as diagnostics, not warmed repeated end-to-end performance.

A successful result would justify engineering the shared storage path further,
not claiming selection-policy superiority or launching paper-scale evaluation.
Preserve failures. Fresh cohorts, calibration and large campaigns remain
disabled. Use two one-A100 workers, 12 CPUs and 16 GiB per worker, 20-minute
limits, the common environment, scratch-only caches, and `bi_fox_dgx`; chain a
finite CPU analysis with `afterany`.

Before GPU submission, 31 focused CPU tests passed in the common environment,
covering every BF16 bit pattern, partial blocks, raw fallback, preservation of
the generated suffix, tiny-model 32-prefix decode parity, source immutability,
hook/temporary-buffer cleanup, and claim separation in the analyzer. Initial
tests caught a CPU integer-promotion mismatch, an unsupported cache constructor
keyword, and an incorrectly ordered incompressibility fixture; each was fixed
before submission. Ruff, shell syntax and targeted MyPy for the codec passed.
The real-model GPU result remains separate from these synthetic controls.

## Selective compiler-cache recovery

Attempt `22178401` completed Description but MM-Vet failed opening a shared
Triton intermediate file; both passed the GPU bit-pattern and raw-transport
controls first. Preserve the completed Description output and the MM-Vet
failure. The corrective decoder makes count/prompt/live-length ordinary runtime
scalars rather than creating a compiled variant for every decode length. Each
Slurm worker also receives a separate scratch-only Triton cache directory.
The encoded format, source selection and exact reconstructed bits are unchanged.

Retry preparation verifies the identical codec and jobs and the hashes of the
unchanged constructor/accounting modules. It binds successful old worker
outputs and retains their actual producer SHA/plan; only failed indices are
executable. The analyzer displays producer SHAs, preserves retry history and
does not pool cold compiler timings across producers. This is reused
correctness/storage evidence, not a mixed-source performance campaign.
Twenty-four focused CPU tests passed after this correction, including explicit
subset submission, invalid-index rejection, source-aware reuse in analysis and
a regression excluding sequence lengths from Triton constexpr specialization.

## V2 bounded engineering check

After complete forced-prefix correctness/storage evidence, keep the same two
consumed identities and saved selections. Replace the CPU constructor with an
integer-only GPU encoder producing exactly the same five component buffers.
Use integer histograms, lowest-base tie breaking, stable literal offsets and
the same raw fallback. No learned values, labels or output-driven format choice
is introduced. Check GPU/CPU encoded-byte equality on every BF16 pattern,
irregular shapes and a deliberately incompressible raw-fallback fixture before
loading the model.

Replace the repeated scan over all tensor storages at every layer with cached
per-layer resident/lease byte totals; history-growth overlap still counts both
old and new backing. An independent owner enumeration validates final counts,
and a CPU regression compares every transaction boundary with full enumeration.
Apply this accounting optimization equally to raw and encoded controls.

This V2 test additionally uses actual autoregressive greedy decoding, not only
the FullKV forced prefix. Compare raw and encoded transports to the ordinary
exact DynamicCache over 32 tokens, and test full-retention lossless parity over
32 tokens. Keep the original forced-prefix guard as a separately hash-bound
artifact; do not compare different trajectories or let the new parity control
waive a prior absolute-fidelity failure. These are correctness controls, not
official task scores. No fresh cohort or statistical superiority claim follows.

Constructor timing synchronizes CUDA before and after construction; decode has
its own synchronized CUDA events. Both remain shared-prefill diagnostics that
exclude selection, prefill, reference construction and request warmups/repeats.
Construction/whole-request memory peaks remain uncertified. Twenty-eight
focused CPU tests passed before V2 GPU submission; GPU encoder/greedy controls
are unmeasured until the immutable workers complete.
