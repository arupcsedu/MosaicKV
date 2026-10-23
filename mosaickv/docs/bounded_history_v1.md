# Bounded-history MosaicKV exact: a PrefixKV-focused development candidate

Status: implemented and CPU-tested; pretrained quality, GPU latency, and paper
acceptance are **unmeasured**. No frozen campaign or historical prediction is
modified. This is an explicitly labeled extension of the exact-only path, not
evidence for prototype merging or residual repair.

## The problem this change actually fixes

The original MosaicKV exact path selects prompt blocks, then appends generated
K/V without eviction. Its preallocated cache reserves space for the entire
requested output. A small prompt cache consequently does not guarantee small
final or peak backing storage. Faithful PrefixKV, in contrast, also evicts
during decoding. Comparing their initial sizes alone concealed this systems
difference. The previous PrefixKV experiments remain unchanged, including their
negative peak-memory and latency findings.

V1 adds a decode-time policy rather than reducing a nominal retention number:

1. Derive the real bytes of one position across **every layer and KV head**.
2. Reserve bounded generated-history capacity from the hard physical cap.
3. Run MosaicKV's prompt forecast/graph/utility/exact selection with the
   remaining prompt capacity, retaining mandatory tokens.
4. Protect every selected prompt K/V entry for the whole generation.
5. Retain the first generated sink positions and the newest generated positions
   in a fixed circular suffix. Evict older non-sink generated entries.

MosaicKV's original K/V entries are never averaged, quantized, offloaded, or re-rotated.
For baselines, the ring preserves whatever their prompt algorithm constructed:
LOOK-M retains its own paper-derived prompt merging, not MosaicKV merging.
There are no residuals. Window/sink eviction itself is not claimed as novel;
this variant tests its combination with MosaicKV's prompt evidence selection.
The independently pinned [StreamingLLM source](../../third_party/StreamingLLM/README.md)
and local `streamingllm_reimpl` remain separate baselines, not MosaicKV code
provenance or validation.

## Byte contract

Let `c` be the all-layer/all-head bytes per dense sequence position, `B` the
lifetime live-KV storage cap, `G` the generated capacity, and `T` the actual
number of decode cache updates. The first output token comes from prefill, so
`T = output_tokens - 1`, or `target_tokens - 1` during teacher forcing. Reserve

\[
g = \min(G,T),\qquad
p = \min\left(N,\left\lfloor B/c\right\rfloor-g\right).
\]

`p` is the maximum prompt capacity. Mandatory positions must fit inside it;
otherwise fail explicitly, without discarding mandatory entries or using a
FullKV fallback. A nonidentity bounded run uses bytes as the selection bound,
not an additional nominal-retention sub-cap. `retention_ratio: 1.0` always
bypasses the entire variant and uses the untouched FullKV identity path.

The live allocation is `(p + g) * c`, including unused generated slots from
the moment it is installed. Initial, final, and peak **storage** therefore
include that reserve. Logical payload grows only until saturation. The unused
cap remainder is recorded; it is not filled with fake evidence. Underfill
because the source prompt has fewer than `p` positions must be disclosed.
For this explicitly labeled variant, legacy `active_kv_bytes` also reports
initial **backing storage including the reserve**, not just prompt payload.
The separate `packed_prefill_active_kv_bytes` trace field records prompt
payload. It must never be used to manufacture a memory advantage by ignoring
the generated reserve.

For the pinned LLaVA-1.5 FP16 layout, its config implies 32 layers, 32 KV heads,
128 channels, and two 2-byte tensors per position: **0.5 MiB per position**.
Thus the example 120 MiB lifetime cap with 32 generated positions leaves
104 MiB for prompt K/V. These are layout calculations, **not measured quality
or compression results**. A 40 MiB cap with the same history leaves only
24 MiB for prompt K/V; this tradeoff may hurt answers and must be measured.

Accounting uses `CacheLifecycle` over unique live tensor allocations. It does
not claim the whole process fits inside `B`: model weights, inactive prefill
copies, planner temporaries and CUDA allocator reservations are excluded from
KV size and separately reflected by peak GPU-memory telemetry.

## Positions and adapter boundary

Supported initially: Transformers **4.55.2**, LLaVA-1.5's exact `LlamaAttention`
class, HF eager, batch one, token blocks, and single-token decode only.
Qwen, Mistral/OneVision, SDPA, FlashAttention, vLLM and padded caches fail
explicitly for this opt-in transport. PrefixKV's different per-layer prompt
lengths are supported **without** cross-layer padding. Each layer keeps its own
prompt length, generated ring, exact mask extent, and original logical positions.
Ordinary paper-native paths are unchanged.

The circular suffix is not chronologically sorted. K and V share exactly the
same physical slot, and each stored key already contains its original RoPE
phase. Each next query receives its original logical position, never the ring
slot or current active length. A layer-specific all-valid attention mask has
the exact occupied physical extent. A single causal decode query may attend
all occupied positions; this does **not** admit multi-token prefill over the
ring, where a more elaborate causal mask would be needed.

The adapter rejects generic cache snapshot/reinjection and crop/rollback
after installation rather than silently losing ring ownership. Multi-token
or batch decoding is not an admitted interface.
Construct isolated caches from separate prefill snapshots when isolation is
needed. Prototype/residual/repair and oracle fidelity paths are disabled for
this initial transport. Retention-one parity remains independently testable.

Installed source boundaries inspected:

- `transformers/models/llama/modeling_llama.py:LlamaAttention.forward` applies
  RoPE **before** `Cache.update`; `LlamaModel.forward` supplies logical
  `cache_position`/`position_ids` separately.
- `transformers/cache_utils.py:DynamicCache.update` delegates to each layer.
- `mosaickv/adapters/huggingface/bounded_history_cache.py` owns reservation,
  fixed storage, ring positions, masks and support checks.
- `mosaickv/adapters/huggingface/base.py:decode_one_token` preserves logical
  growth while physical cache growth saturates.
- `mosaickv/backends/hf_runtime.py` reserves before selection and installs
  before live-storage accounting, including setup in compression time.

## Labels and reproducibility

Enable with `bounded_decode: {generated_capacity: 32, sink_tokens: 4}`.
MosaicKV and simple baselines get the suffix `__bounded_generated_history_v1`;
published-baseline adaptations instead end with
`__bounded_generated_history_v1_reimpl`. None may be pooled with historical
unmodified method rows. Supported selectors are PrefixKV, LOOK-M, VL-Cache,
Prompt Top-K, random/uniform/value Top-K, H2O, StreamingLLM, PyramidKV, SnapKV,
and RoCo. Elastic Cache's distinct decode transaction is explicitly unsupported.
All bounded configuration
values enter the canonical config hash. Omission of `bounded_decode` leaves
historical canonical config hashes and ordinary inference unchanged.

Set `cache.retention_ratio: 0.9999999999999999` for the byte-driven variant;
`1.0` is reserved for the unchanged identity control. A second nominal-ratio
sub-cap is rejected, not silently allowed to underfill the requested bytes.

PrefixKV must use the audited ragged transport, globally apportioned profile
rescaling, and `decode_policy: append_only` for the initial selection-only
stage. The ring **replaces**, rather than compounds, native decode eviction.
The prompt profile shape and selected positions are preserved. Such a run is
not paper-faithful native PrefixKV and does not replace its official parity
or native PPL/ROUGE-L evidence. LOOK-M's common mandatory tokens are reserved
before optional ranking; an impossible mandatory+recent reservation fails.

## Small cross-baseline development pilot

`configs/experiments/bounded_history_baselines_v1.yaml` registers two previously
used examples per task from LLaVA-Description and paper-style MM-Vet, using
metadata-order selection without answer-based filtering. It is explicitly
retrospective development, not a fresh held-out result. There are 22 workers:
two FullKV references and five compressed methods at both 40 and 80 MiB for
each task. The methods are MosaicKV exact, Prompt Top-K, `prefixkv_reimpl`,
`lookm_reimpl`, and `vl_cache_reimpl`, all with the same ring policy.

Each worker uses one A100, 12 CPUs and at most two hours; the shared admission
controller caps aggregate occupancy at four A100s. Reference workers must
finish before compressed workers run. Five warmups and one quality trial per
example are identical across methods. All generation runs request 1,024 tokens;
teacher forcing uses complete identical human targets. PPL and ROUGE-L are
separate. FullKV is excluded from compressed latency comparisons. Every
compressed row must have the same measured initial/final/peak storage at its
cap, including reserved slots. No 40-vs-80 direct comparison is made.

VL-Cache on this single LLaVA-1.5 checkpoint is a generalized reimplementation,
not its native Mistral reproduction. LOOK-M on PrefixKV datasets is a
controlled adaptation, not MileBench evidence. These runs test the common
controller, not reproduction of all three papers' original protocols.

`scripts/bounded_history_pilot.py` prepares hash-bound derived ledgers and
per-worker configs, executes import-safe metric routes, and writes separate
per-task summary Parquet and a diagnostic report. It verifies the clean source,
model and dataset revisions, profile pin, configs and all bound files. Failed
attempts retain error records; successful workers resume only after artifact
hash verification. Reference-control hashes explicitly describe the common
ring with a FullKV quality-reference exemption, rather than falsely declaring
all runs append-only. No installation or model download is performed.

The example is
[`llava_bounded_history_v1.yaml`](../configs/development/llava_bounded_history_v1.yaml).
It is a development preset, not permission to launch a paper campaign. Adapt
dataset tuples only through the reviewed native-metric materializer.

Traces record the hard cap, prompt capacity, sink/recent capacities, reserved
bytes, setup time, final retained generated logical positions, evicted count,
and initial/final/peak live allocation sizes. Failures preserve the planned
reservation and standard structured failure record. No answer/reference field
is passed to the ring or reservation controller.

## Validation and comparison requirements

CPU tests exercise real random-weight HF LLaVA architecture code without model
downloads: ring logits/tokens versus explicit chronological exact eviction,
repeated wraps, GQA/MHA layouts, source/request isolation, mandatory prompt
preservation, complete teacher-forced targets, cap/rounding checks, strict
configuration and retention-one bypass. These are correctness evidence, **not
pretrained PPL or ROUGE-L results**.

Before a large run, use the faithful PrefixKV checkpoint and native datasets:
LLaVA-Description and paper-style MM-Vet. Keep the old official reproduction
and the original nominal-retention lane intact. FullKV supplies the fixed
ROUGE-L reference and teacher-forcing controls, not a compressed latency row.

First test approximately eight examples per route with ordinary MosaicKV,
bounded MosaicKV, faithful PrefixKV, and Prompt Top-K with and without the
same bounded controller. Compute corpus PPL from total NLL/token counts and
FullKV-reference ROUGE-L separately. Preserve every failed row.

Direct quality/latency comparisons must use the **same measured physical
budget**: 40 versus 40 MiB, 80 versus 80 MiB, etc. Do not present 35 versus
40 MiB as an apples-to-apples result. Compare lifetime storage including
reserves, not prompt-only payload against another method's whole cache.
If PrefixKV's native variable allocation cannot realize the same storage
boundary, label it a separate native operating point; do not hide the mismatch.

A smaller-cache claim is a **separate iso-quality question**, not an equal-byte
quality win: each method needs a measured budget curve and the same registered
quality requirement. Do not promise a 30% advantage, infer it from this
allocation formula, or select an inferior PrefixKV adaptation as the nearest
competitor. The variant must first preserve pretrained PPL/ROUGE-L; if its
short generated history degrades quality, it has not solved the research goal.

### Development checks

Use the common environment and scratch cache policy:

```bash
source mosaickv/scripts/cache_env.sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /scratch/djy8hg/env/mosaickv/bin/python -m pytest \
  mosaickv/tests/unit/test_bounded_history_cache.py \
  mosaickv/tests/integration/test_bounded_history_runtime.py -q
```

The original MosaicKV-only development check additionally included `test_config`,
`test_preallocated_cache`, `test_hf_adapters`, `test_hf_runtime`,
`test_cache_isolation`, `test_hf_tiny_models`, `test_prefixkv_ragged`, and
`test_prefixkv_native_cli`: 248 tests passed on 2026-10-23 before baseline extension.
The baseline-extension check covers 196 tests including all common-ring
selectors, layer-ragged PrefixKV, complete PPL/ROUGE-L execution, the pinned
LOOK-M evaluator fixture, and existing native-lane controls. Scoped Ruff checks
and strict MyPy checking of the new cache module also passed. This does not
certify the entire repository, a clean canonical source SHA, pretrained
quality, or GPU timing. No model download, environment installation, Slurm
submission, or git commit was needed for these development checks.
