# Consumed-input exact-storage study

This is a four-stage engineering diagnostic, not a new benchmark cohort or a
claim that EviCastKV fixes the previously observed answer-quality deficit.
The fixed inputs are LLaVA-Description `000000532947` and paper-style MM-Vet
`v1_197`. The selectors, original K/V bits, logical positions, BF16 checkpoint,
eager attention and greedy decoding stay unchanged. Both Prompt Top-K and
EviCastKV receive every common storage optimization.

## 1. Per-token feasibility

For an MHA layer with H heads and dimension d, a token retained by m heads
requires 2md BF16 values as explicit K/V. Its normalized source row requires
Hd BF16 values before metadata. Propose source storage only when 2m > H;
ties use explicit K/V. Reconstruct source rows with the **original full-shape
projection**, original weights and original RoPE, then overlay original K/V
for low-degree rows. The zero canvas preserves the original GEMM dimensions;
zeros at unused rows are never selected.

This threshold is only a raw-payload proposal. Choose among whole-layer K/V,
whole-layer source and token-hybrid representations using actual encoded
buffer bytes, including position maps, direct-slot maps and rotary tensors.
Shared rotary backing is deduplicated for total accounting. The conservative
per-layer decision charges its rotary backing in full. GQA is not supported.

## 2. Numerical-patch density

Compact source projections previously failed BF16 bit equality because changing
GEMM shape changes rounding. Measure the exact differing elements and store
their original BF16 bit patterns plus int32 flat indices. This is a fixed
lossless exception stream, **not residual repair**. Test fixed patches across
three reconstructions and compare the complete source-plus-patch cost against
encoded explicit K/V. Fall back to that K/V representation if patches cost more.
No tolerance is relaxed. This is not advertised as an efficient codec when
exceptions are dense.

## 3. Direct-buffer construction

Decode encoded K/V directly into the current attention lease. For source rows,
retain original-shape K/V GEMMs, fuse staged-BF16 RoPE with gather into the K
prefix, release full K, and gather V directly into its prefix. This eliminates
extra selected-K/V temporaries, not the original-shape GEMM cost. Both the
codec and RoPE kernel must pass strided-window sentinel tests: writes must not
touch preceding/following history or another head.

## 4. Generated-history compression

Use one reusable raw 16-token tail. Before overwriting a full tail, seal an
immutable lossless encoded chunk. Raw codec fallback must clone, never alias
the mutable tail. Reconstruct sealed chunks and the used tail into the lease
in original order. No history token is evicted. Test both 32- and 128-token
outputs. Count encoded chunks, raw tail, temporary encoding storage and lease.

## Gates, accounting and automation

Each stage requires retention-1 identity, synthetic CUDA controls, original
selected K/V bit identity, complete own-greedy logit identity against the
ordinary selected cache, no source mutation, correct logical progression,
complete history, zero leaked leases and complete hash-bound artifacts.
Failure stops the chain. Success authorizes only the next of these four fixed
consumed-input stages. It never authorizes calibration or a paper campaign.

Each stage is two one-A100 workers, 12 CPUs and a 20-minute limit, followed by
an `afterany` CPU analysis. The analyzer submits the next fixed stage only on
success. Maximum concurrent GPUs for this chain is two (aggregate policy six).
There is no daemon or continuous log polling.

Report actual initial/final/peak **post-construction decode-owned tensor
backing**: payload, indices, patches, shared rotary, history, attention lease,
history-growth overlap and observed reconstruction/encoding tensor results.
Inactive diagnostic full-prefill/captures, model weights, attention activations,
internal GEMM workspaces, allocator reservation and construction temporaries
are outside this boundary. Whole-request HBM peak remains uncertified.

Constructor wall time is separate. CUDA-event and synchronized wall decode
timings use instrumentation and a single run, and are **not** a fair repeated
latency comparison or an end-to-end speedup claim. A subsequent timing study
would require matched controls, warmups, randomized order and repetitions.

## Attribution

[Palu](https://arxiv.org/abs/2407.21118) reconstructs K/V from compressed
intermediate states using low-rank projections. This diagnostic instead reuses
unchanged original projections and demands bit equality; it is not Palu or a
claim that source reconstruction is novel.
[AdaCode](https://aclanthology.org/2026.emnlp-main.269/) exploits concentrated
BF16 exponents with lossless coding. Our existing experimental fixed-three-bit
exponent codec is not the official AdaCode implementation and does not reproduce
its adaptive code/range optimization. Common lossless coding must also be
offered to baselines; savings relative to raw baselines alone are not an
algorithmic EviCastKV superiority result.

Entry point: `mosaickv/scripts/evicastkv_exact_storage.py`. Source must be clean
and immutable before prepare/submit. All output and caches remain in scratch.
