# Fox-review source controls: bounded consumed-input diagnostic

EviCastKV is the paper name of MosaicKV. This diagnostic does not change or
rename the frozen exact-selection candidate. It uses only the two historical
inputs in `source-transport-corrected-consumed-20261101-v1`, not fresh samples.
It is deliberately separate from the FP16 LLaVA-v1.6-Mistral COCO experiment.

## What can execute safely

The existing source codec is restricted to batch-one, BF16, full-MHA
`llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`.
All seven registered arms use that checkpoint, processor, identical input,
eager attention, 128 predictions/127 appends, deterministic settings, unchanged
original logical positions, and the existing lossless generated-source history.
There are two one-A100 workers with 12 CPUs and two-hour limits; the aggregate
owned-A100 ceiling remains six. The outer controller owns submission/admission.

1. FullKV reference: all original BF16 prompt K/V transferred to the isolated
   reader. No selection or lossy codec. Ordinary-generation and retention-one
   parity are also checked for 16 tokens before the experiment.
2. Prompt Top-K: half the source positions, or all mandatory positions if those
   exceed half, with deterministic score/position ordering. Exact selected K/V.
3. Shared-source 3-bit with mandatory protection only.
4. Shared-source 3-bit with random optional visual protection (seed 13).
5. Shared-source 3-bit with attention-based optional visual protection.
6. **Local direct projected-K/V 3-bit control, not KIVI**: existing grouped
   asymmetric recipe, keys along positions, values along features, group 128,
   FP32 scales/minima, lossless original protected post-RoPE K/V overlays.
7. Same representation as arm 5 with actual CUDA reconstruction replay enabled.

Mandatory protection includes all original text and first four/last eight
original positions. Random and attention optional protection each add exactly
`floor(0.05*N)` available visual rows per head, clipped to available rows.
Attention uses the original last-16 prompt attention window. This explicit
diagnostic allocation is **not** mislabeled as the frozen historical allocation.
All required numerical corrections and construction predicates remain enabled.
The existing source-native projection envelope remains `rtol=.01, atol=2^-6`;
the separate 32-token guard retains identical greedy tokens, KL <= .02 and
maximum logit difference <= 1.0. Lossy-source failures are recorded, not hidden.

`SharedSourceCache` supplies OFF/ON replay with shared reconstruction workspace
in both arms. Only immutable prompt/sealed-history reconstruction is captured,
not full transformer decoding or attention. Each arm is a complete independent
request. Packet hashes must match across OFF/ON, no graph can execute in OFF,
and ON must report positive capture/replay counts. Graph-private allocations,
workspace and all persistent source tensors are charged and retired explicitly.
Replay output parity is unchanged greedy tokens and maximum logit error <=1e-3.

## Storage and timing scope

This is a **common-model rate/distortion diagnostic, not an equal-byte table**.
Codes, headers, maps, exact overlays, numerical patches, lossless generated
history, materialization/workspace and graph-private owners are included by the
existing storage runtime. GPU and CPU residual storage remain separate.
Original inactive FullKV and capture owners must be retired before compressed
decoding. FullKV legitimately transfers its original tensors to its reference
representation instead of claiming their retirement.

Per-arm media/processor, captured prefill, construction, retirement and decoding
are recorded. A single trial, integrity hashing, diagnostic capture, and no
warmup make these **ineligible for headline latency**. Allocator peaks are
reported separately from the observed decode-owned cache storage boundary;
whole-request KV-only peak certification is not claimed.

## Recover before rerunning

The script hash-binds the historical corrected-source, native-metric,
prepared-vs-legacy replay, and source-vs-raw-history decisions. Those studies
remain valid in their recorded scopes, but neither prepared-vs-legacy replay
nor source-vs-raw-history equals the requested replay OFF/ON intervention.
No historical result is rewritten or retrospectively promoted.

The baseline inventory binds current source documents and immutable upstream
revisions. PrefixKV's historical parity/reproduction uses its own checkpoint,
environment and rounding conventions, not this new byte-adapted source policy.
LOOK-M's pinned upstream imports an absent class; only `lookm_reimpl` is
executable here. VL-Cache remains `vl_cache_reimpl`, not official author code.
An executable same-byte local port is not itself official-path fidelity proof.

## Explicit remaining blockers

- A label-blind feasibility-aware common-byte source controller is not supplied
  by this diagnostic. Different achieved bytes must never be called matched.
- KIVI, KVQuant and ZipCache have no verified, pinned executable implementation
  in this source study; they remain unsupported. The local quantizer is not a
  substitute for their names or fidelity evidence.
- A bridge to the current FP16/GQA Mistral exact study is unsupported by the
  full-MHA BF16 source constructor. Do not pool the two configurations.
- Consumed examples, two inputs and single trials cannot establish superiority,
  independent quality preservation or a whole-request memory advantage.

The fail-closed API is `prepare(root)`, `worker(root, index)` and `analyze(root)`
in `scripts/evicastkv_review_source_controls.py`. The outer immutable controller
adds afterany analysis and keeps fresh confirmation/larger campaigns blocked.
