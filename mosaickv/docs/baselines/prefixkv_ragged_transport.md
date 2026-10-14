# PrefixKV independent-layer HF repair (2026-10-13)

## Decode-policy audit and correction (2026-10-14)

The official 1,000-example Description reproduction at 50% retention passed
its registered PPL tolerance (3.40117 versus 3.41 ± 0.25) and measured ROUGE-L
0.82835. It used the legacy checkpoint, BF16, Transformers 4.31.0, 16 output
tokens and approximately 157 MiB. Its manifest explicitly did **not** certify
the local common-HF implementation. It cannot certify 40 MiB OCR inference
with HF-converted weights, FP16, Transformers 4.55.2 and 64 output tokens.
Evidence: `prefixkv/official-profile-r050-2c7a266/official_profile_reproduction_gate.json`
under `/scratch/djy8hg/runs/mosaickv`.

Transport parity previously compared against a manual oracle that reused the
local eviction formula. It therefore missed a real algorithmic difference:
upstream `third_party/PrefixKV/prefixkv.py` rounds forgotten positions at
prefill but keeps the **original floating forget ratios** during decode.
The local implementation instead used rounded retained positions divided by
prompt length. Those schedules can evict on different steps. Fixing that
discrepancy does not guarantee a quality improvement or explain every OCR failure.

Explicit, backward-compatible policies now distinguish:

- `allocation_policy: upstream_round`, `decode_policy: profile_ratio`: use
  original independent numpy rounding and original profile floats. Infeasible
  mandatory bounds or hard caps fail, rather than altering the native profile.
- `global_apportion` + `effective_profile_ratio`: fill the shared physical-byte
  budget, derive a new per-layer forget profile from achieved counts and use
  the upstream floating decode arithmetic with that **adapted** profile.
  This is a byte-rescaled `prefixkv_reimpl`, not an official operating point.
- `global_apportion` + `append_only`: preserve the identical prefill selection
  and append generated KV without further eviction. Raw results and traces
  carry **`prefixkv_selection_only_reimpl`** as the effective method. This is
  an ablation, not paper-native PrefixKV. Its storage growth matches other
  append-only methods at the same initial bytes, unlike native eviction.
- `realized_ratio` remains the default solely for configuration/artifact
  compatibility. No old results are overwritten or relabeled as repaired.

`evaluation/prefixkv_differential.py` invokes the pinned upstream class directly
on identical HF prefill caches. Its only adapter redirects the upstream
profile read to an in-memory, hash-recorded profile; upstream source, selection
and eviction code remain unmodified. Marker tensors independently check exact
selected and evicted logical positions, alongside per-layer K/V, 64-step
greedy logits/tokens, exact storage and source isolation. This is **algorithm
parity on common HF**, not reproduction of the legacy model checkpoint.

The GPU gate runs retention-1 parity, native-profile 50% differential parity,
then the 40 MiB adapted-profile differential. Quality workers proceed only
after these pass. The development cohort has eight IDs per task, selected
by parent-ledger order `[24:32]`, excluding earlier calibration/debug `[0:24]`.
This is retrospective development, not untouched paper evidence. All matched
arms share first/last mandatory tokens, block size one, FP16/eager, processor,
media, prompts, output length 64, seed and 40 MiB prefill cap. The native 50%
diagnostic remains separate. FullKV is a quality reference only. Zero warmups
and one trial make timing informational; no latency claim follows.

The frozen retry does not tune profiles or relax quality gates. A larger
preregistered paired quality evaluation and proper repeated latency trials
are still required; code/transport parity alone never releases the full sweep.

## What failed

The original `21292854` PrefixKV workers deliberately rejected compressed
plans. The common packer pads every layer to a common maximum length; that
does not represent PrefixKV's physical memory, and independently scheduled
decode evictions immediately invalidate a uniform-length assumption. The old
guard remains the default (`prefixkv.hf_cache_transport: common_guarded`).
Old failed artifacts and the completed Top-K/MosaicKV campaign are preserved.

## Opt-in replacement

`ragged_llama_eager_v1` is a local `prefixkv_reimpl` transport. Its supported
boundary is batch-one LLaVA-1.5, eager attention, Transformers 4.55.2 and an
ordinary `DynamicCache`. Other families, attention kernels and library
versions fail closed. It does not enable official PrefixKV reproduction.

Source contracts inspected:

- Installed `transformers/models/llama/modeling_llama.py`:
  `LlamaModel.forward` creates position embeddings from original
  `cache_position`; `LlamaAttention.forward` applies RoPE before
  `Cache.update`, then consumes a layer-local attention mask.
- Installed `transformers/cache_utils.py`:
  `DynamicCache.from_legacy_cache` inserts each layer independently.
- `third_party/PrefixKV/prefixkv.py`, pinned at
  `597f1ab032704951550f93bcc8a23f1454b80aa4` (MIT): layer-wise attention
  selection and at-most-one fixed-offset eviction after each decode.
- Local `adapters/huggingface/prefixkv_cache.py`: exact gathers, mask extent
  validation, logical-position provenance and independent-layer eviction.

K/V tensors retain `[1, KV_heads, layer_length, head_dim]` layout. There is
**no inter-layer padding**. Retained post-RoPE keys are copied, not recomputed
or re-rotated. Single-token decoding uses a zero additive mask of each
layer's actual `layer_length + 1`; every retained position precedes the query.
The next position advances on the original logical timeline. A request-scoped
hook changes only masks; it is disabled outside that forward call.

Marked ragged snapshots explicitly permit unequal lengths. Their scalar
`active_sequence_length` is the **maximum** layer length, never a shared
storage length or the logical sequence length. Exact per-layer lengths and
positions remain authoritative; byte measurements sum actual K/V storage.
Unmarked caches retain the existing uniform-length validation.

## Budget and scientific labels

The unified byte-cap variant rescales the frozen profile across the **total**
layer-position budget. It does not clamp every layer to the dense per-layer
average, which would destroy the profile and underfill the cap. Deterministic
integer apportionment fills the largest representable target. Equal per-layer
byte widths are required. Gathering verifies physical storage equals planned
bytes. No zero padding is used to manufacture byte matching.

The historical opt-in unified worker used the common last-prompt-token mandatory rule
(`start_size=0`, `protect_size=1`); this differs explicitly from upstream's
BOS-plus-tail prefill rule. Original profile shape, source SHA and transport
choice are recorded. This is a **byte-matched local variant**, not an official
paper operating point. The existing short-cache eviction fallback for fewer
than three positions is also a local choice; upstream can assert there.

The shared measured cap refers to the post-compression **prefill** cache.
PrefixKV's subsequent fixed-offset eviction schedule is part of its method;
other methods may append without eviction. Final bytes and eviction events
must therefore be reported separately. Equal prefill bytes do not imply equal
bytes at every decode step or prove paper-native fidelity. FullKV is a
correctness/quality reference, not a latency competitor.

## Quality correction: protected-boundary byte variant

The negative eight-example-per-task development run at source `9add825`
is preserved at `/scratch/djy8hg/runs/mosaickv/prefixkv-quality-9add825-20261013-v1`.
It did not pass the Prompt Top-K quality gate. A label-blind allocation audit
found that the recalibrated profile creates one- and two-position layers,
omits BOS, and can evict BOS through the legacy clamped offset. This is a
concrete defect in that variant; it does **not** explain all failures of the
older upstream-profile variant, which already retained BOS in that audit.

New retries explicitly set `cache.physical_budget_mandatory_policy:
first_and_last` for **all compressed methods**, and
`prefixkv.decode_boundary_policy: protected`. The latter reserves at least
`max(3, start_size + protect_size + 1)` positions per layer (or the entire
source for shorter original prompts). Deterministic global apportionment
redistributes the unchanged byte budget; infeasible caps are rejected. A
fixed eviction offset must lie between the protected boundaries. The decode
path rejects boundary eviction before deleting any key/value position and
records `protected_start_present` for each layer/step. Retention 1 remains
identity. `legacy_clamped` exists only for backward-compatible historical
diagnostics and is not the corrected quality candidate.

The three-position lower bound follows the pinned implementation's midpoint
assertion, while enforcing it during global byte allocation is an explicit
local byte-matched adaptation. This is still `prefixkv_reimpl`, not official
PrefixKV. CPU regressions and transport parity establish safety, **not answer
quality**. Both tasks must still pass the unchanged measured quality gate;
no full campaign is authorized by this code correction. Historical
last-token-only Top-K/MosaicKV rows cannot be reused in the new paired cohort.

## Validation and rerun policy

CPU tests use randomly initialized tiny LLaVA, no downloads. They compare 16
tokens and logits against explicit per-layer execution, with GQA and MHA,
unequal layer lengths, independent evictions, exact storage accounting,
retention-1 identity, cloned-state isolation, and subsequent ordinary
`generate` parity. Synthetic tests are not benchmark-quality measurements.

`scripts/verify_prefixkv_ragged_transport.py` supplies a separate clean-source
GPU gate: pinned real checkpoint, ordinary-generation and reinjection parity,
compressed per-layer oracle parity, finite logits and exact 40 MiB prefill
storage. Token agreement must be 1.0 and max absolute logit difference at most
1e-4, fixed before execution. Failure writes an artifact and blocks dependent
workers. Passing this gate is not proof of answer quality or upstream parity.

Then rerun a small derived TextVQA/DocVQA ledger using the same model,
generation, cap and scoring controls. Keep the old 128 PrefixKV tasks held;
do not run the obsolete transport again. Use a new output root and source SHA.
Do not expand the campaign until the transport and per-sample byte checks
pass. All retries share the global four-A100 admission controller.
