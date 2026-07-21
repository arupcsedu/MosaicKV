# Baseline fairness audit

Audit date: 2026-07-20. Scope: the unified Hugging Face eager runtime and the
nine requested method labels. This is a source, configuration, trace, and
structural-accounting audit. It is not a benchmark result.

## Verdict

**The repository is not yet ready for a scientifically fair nine-method
comparison.** The methods share the same model adapter, message builder, decode
loop, and eager backend, but the current experiment configurations do not share
an active-KV byte budget, mandatory-token policy, block granularity, warmup
protocol, or fully verifiable input-content identity. No current result set may
be presented as a complete baseline comparison.

The main failure is the distinction between selected payload and physical HF
cache storage. The planners enforce a logical block/slot/byte budget, while
[`pack_runtime_payloads`](../src/mosaickv/backends/hf_runtime.py#L751) pads every
layer and KV head to the largest selected head length. The padded tensor is the
active device cache and therefore controls the primary comparison budget. A
nominal token or block retention ratio is not an acceptable substitute.

The audit made four narrow correctness fixes without changing any scoring,
selection, merge, or repair algorithm:

1. the HF entry point now enables PyTorch deterministic algorithms and seeds
   CPU and CUDA from the recorded seed;
2. the runtime rejects an output-length policy it does not implement instead
   of silently executing fixed-length decoding;
3. evaluation rows now use the sample's effective method label, so an exact-only
   safety fallback is not reported as `mosaickv_proto` or `mosaickv_full`; and
4. new traces distinguish source prefill bytes, logical active payload bytes,
   padded physical prefill bytes, padding bytes, and final post-decode bytes.

Existing immutable runs are not rewritten. Their result rows retain their old
requested-method labels and their traces remain the authoritative evidence of
any fallback.

## Controlling budget definition

For quality comparisons at a compressed operating point, define one integer
`B` in bytes before launching any method. A row is byte-matched only when:

```text
packed_prefill_active_kv_bytes == B
```

`packed_prefill_active_kv_bytes` is the sum of `numel * element_size` for every
physical K and V tensor immediately after compression/packing and before the
first decode step. It includes padding required by the actual backend layout.
It excludes CPU residual storage, which must be reported separately. The final
`active_kv_bytes` includes decode-time cache growth and is comparable only when
generated-token counts are also identical.

The following fields have deliberately different meanings:

| Field | Meaning | Use |
|---|---|---|
| `source_prefill_kv_bytes` | Full prompt K/V payload | FullKV reference and denominator |
| `logical_prefill_active_kv_bytes` | Selected exact/prototype/merged payload before backend padding | Algorithm diagnostic only |
| `packed_prefill_active_kv_bytes` | Physical device K/V tensor after packing | **Primary comparison budget** |
| `packed_padding_kv_bytes` | Physical minus logical prefill payload | Backend-layout overhead |
| `active_kv_bytes` | Physical K/V after the fixed decode | End-state memory diagnostic |
| `residual_kv_bytes` | Separate CPU residual payload | Report separately and include in total-storage analyses |

FullKV is a quality reference at its natural full-cache budget. It cannot be a
budget-matched compressed systems point at `B < source_prefill_kv_bytes`.

## Shared-control audit

| Control | Current mechanism | Verdict |
|---|---|---|
| Model weights and precision | `RunConfig.model` requires an immutable revision and precision; `_model_kwargs` maps that precision for the one shared adapter load. | Enforceable within one cohort, but no cohort validator currently rejects mismatched manifests. |
| Processor and chat template | Every method calls the same adapter's `prepare_inputs` and processor. | Enforceable for a shared runtime instance. Not proven across separate runs because processor/chat-template content is not hashed. |
| Media preprocessing | The same processor receives the same standardized messages. InternVL is an exception because its public checkpoint-specific preprocessing is supplied externally. | Partly enforceable; media bytes and realized processor outputs are not preserved in the current manifest. |
| Prompt text | The harness deterministically selects sample IDs and constructs messages once per request. | Not auditable from current manifests: the fields named `prompt_set_sha` and `media_set_sha` are derived from sample ID/reference selection, not prompt/media content ([`_input_provenance`](../src/mosaickv/hf_cli.py#L252)). |
| Output length | The shared loop executes exactly `max_new_tokens`; non-fixed output policies now fail closed. | Pass for new unified-runtime runs using the same generation config SHA. |
| Deterministic decoding | Runtime rejects sampling and nonzero temperature; CLI now calls `torch.use_deterministic_algorithms`, `manual_seed`, and `cuda.manual_seed_all`. | Pass for new runs, subject to deterministic-kernel availability. Old unified runs recorded the flag without enforcing it globally. |
| Active KV byte budget | Logical planners enforce their configured unit; new traces record physical packed bytes. | **Fail.** Existing configs mostly use blocks, and head/layer padding can exceed the logical budget substantially. |
| Mandatory tokens | Simple baselines, VL-Cache, and MosaicKV protect the terminal prompt token; LOOK-M protects its recent window; PrefixKV protects start and tail positions. | **Not identical by design.** Preserve these paper-derived rules and report their byte cost; do not call them identical. |
| Block granularity | LOOK-M, PrefixKV, and VL-Cache require `block_size=1`; simple baselines and MosaicKV are configurable and existing MosaicKV runs use 16. | **Fail for existing runs.** A cross-method token-level stratum must set all applicable methods to block size 1. A block-16 MosaicKV stratum must exclude token methods or label the granularity difference. |
| Attention backend | `HuggingFaceMosaicKVModel` rejects anything except eager attention. | Pass inside the unified HF runtime. vLLM, SGLang, SDPA, and FlashAttention-2 are unsupported for this comparison. |
| Warmup count | The evaluation harness performs no warmups; the dedicated FullKV benchmark defaults to one warmup and five trials ([`FullKVBenchmarkConfig`](../src/mosaickv/fullkv.py#L31)). | **Fail for systems comparisons.** Warmup count is not in `RunConfig` or the unified manifest. |
| Measurement protocol | Unified phases use synchronized CUDA events when the exemplar is CUDA ([`_timed`](../src/mosaickv/backends/hf_runtime.py#L137)); the total is a host clock. Dedicated FullKV uses a different repeated-trial CUDA-event protocol. | **Fail for paper latency comparisons.** Use one repeated-trial runner for every method before comparing latency. |

Required attention/query capture is method work, not an unfair perturbation to
hide: FullKV needs neither, prompt-attention methods need attention, and
MosaicKV additionally needs query capture/draft rollout. End-to-end latency
must include these required costs, while phase tables should show them
separately.

## Method-level audit

| Requested method | Selection/representation | Mandatory policy | Granularity and configured budget | Fairness disposition |
|---|---|---|---|---|
| `full_kv` | Identity; no pruning, merge, quantization, offload, or repair | All source positions remain | Full cache only | Quality reference; only byte-matched at 1.0 |
| `random_kv` | Seeded exact-block selection | Terminal prompt token | Configurable blocks; block/slot/byte logical budget | Deterministic, but physical padding must be matched from the trace |
| `prompt_attention_topk` | Exact blocks ranked by prompt-window attention | Terminal prompt token | Configurable blocks; block/slot/byte logical budget | Same caveat as random; attention capture is required method overhead |
| `lookm_reimpl` | Paper text-prior selection and KV merge | Whole LOOK-M recent window | Token blocks; block budget only | LLaVA-1.5 reimplementation stratum only; byte-match by realized packed storage |
| `prefixkv_reimpl` | Layer-wise positions from a profile or fixed-global allocation | Configured start and protected tail | Token blocks; logical block or byte budget | Layer-adaptive profiles can incur large HF padding; calibration provenance is mandatory |
| `vl_cache_reimpl` | Layer-adaptive exact positions from post-vision attention sparsity | Terminal token plus configured recent fraction | Token blocks; block budget only | Paper equations retained, but physical layer padding can violate a nominal ratio |
| `mosaickv_exact` | Forecast, graph, utility, exact selection | Terminal prompt token | Configurable blocks; block/slot/byte logical budget | Eligible only after packed bytes are exactly matched |
| `mosaickv_proto` | Exact plus prototypes when adapter declares safe merging | Terminal prompt token | Configurable blocks; logical active budget | Unsupported as a prototype method on every current post-RoPE adapter; executes an explicitly labeled exact fallback |
| `mosaickv_full` | Exact, prototypes, CPU residuals, repair when supported | Terminal prompt token; promoted blocks persist | Configurable blocks; logical active budget | Unsupported as the full method on every current adapter; executes an explicitly labeled exact fallback with no repair |

The method-specific mandatory sets and paper-required token granularity are
algorithm definitions. Changing LOOK-M or PrefixKV to use MosaicKV's mandatory
set would no longer be a paper-faithful reimplementation. Those differences
must be stratified and disclosed rather than normalized silently.

## Byte-budget validation artifact

[`baseline_budget_validation.parquet`](../results/baseline_budget_validation.parquet)
is generated by
[`build_baseline_budget_validation.py`](../scripts/build_baseline_budget_validation.py).
It is tagged in both columns and Parquet metadata as
`validation_smoke_not_measured_result`. It contains no model-quality or timing
measurement and must never be copied into a measured-results table.

The deterministic fixture is a two-layer, two-KV-head, 12-position, FP32
post-RoPE cache with a 1,536-byte source and token-sized blocks. At its
768-byte structural target, the artifact records:

| Requested method | Logical bytes | Packed physical bytes | Exact byte match | Requested algorithm exercised |
|---|---:|---:|---:|---:|
| `full_kv` | 1,536 | 1,536 | No | No; FullKV rejects 0.5 |
| `random_kv` | 768 | 1,152 | No | Yes |
| `prompt_attention_topk` | 768 | 1,280 | No | Yes |
| `lookm_reimpl` | 768 | 768 | Yes | Yes |
| `prefixkv_reimpl` | 768 | 768 | Yes | Yes |
| `vl_cache_reimpl` | 768 | 1,408 | No | Yes |
| `mosaickv_exact` | 768 | 768 | Yes | Yes |
| `mosaickv_proto` | 768 | 768 | Yes | **No; exact safety fallback** |
| `mosaickv_full` | 768 | 768 | Yes | **No; exact safety fallback** |

These values demonstrate a representational failure mode, not expected model
performance. All nine retention-1 structural rows reconstruct the full
1,536-byte cache. The artifact records both requested and effective method, so
a byte match alone cannot make a fallback eligible.

Rebuild and inspect it only in an environment whose verification has already
passed and which contains PyArrow:

```bash
PYTHONPATH=mosaickv/src /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/build_baseline_budget_validation.py

/scratch/djy8hg/env/mosaickv/bin/python - <<'PY'
import pyarrow.parquet as pq
table = pq.read_table("mosaickv/results/baseline_budget_validation.parquet")
print(table.schema.metadata)
print(table.to_pandas())
PY
```

## Existing run audit

Nineteen manifests were present under `/scratch/djy8hg/runs/mosaickv` at the
audit time. They do not form a nine-method cohort: there is no unified FullKV,
random, prompt-attention, VL-Cache, or requested MosaicKV-prototype/full result
set. The comparable-looking LLaVA synthetic runs also use different nominal
ratios and different physical budgets.

The following are real trace observations used only to diagnose the audit;
they are not a comparison table:

| Run ID and measurement type | LLaVA-1.5 BF16 eager method | Nominal ratio | Logical prefill payload | Final physical KV | Audit interpretation |
|---|---|---:|---:|---:|---|
| `9a4e9b64e84a4ae984d1d446442adeb0`, `method_measured` | requested `mosaickv_full` | 0.5 | 20,918,272 bytes | 319,815,680 bytes | Effective exact-only fallback; global padding capacity was 595 positions |
| `lookm-reimpl-hf7b-17118553`, `baseline_reimpl_measured` | `lookm_reimpl` | 0.2 | 62,914,560 bytes | 70,778,880 bytes | Different ratio and budget; not comparable to the other rows |
| `prefixkv-reimpl-hf7b-17125638`, `baseline_reimpl_measured` | fixed-global `prefixkv_reimpl` | 0.5 | 157,286,400 bytes | 161,480,704 bytes | Uniform layer sizes limit padding |
| `prefixkv-official-profile-validation`, `baseline_reimpl_measured` | offline-profile `prefixkv_reimpl` | 0.5 | 157,286,400 bytes | 285,212,672 bytes | Same logical bytes, much larger physical storage due to the maximum layer length |

The 0.5 PrefixKV rows are a direct warning against ratio-only comparisons: the
same logical byte total can produce very different physical storage. The final
column also includes decode growth, so future cohorts must use the new packed
prefill field as `B` and retain final bytes as a separate diagnostic.
For these older traces, logical prefill payload is derived exactly from the
recorded valid packed-slot count and the checkpoint's K/V dtype and head
dimension; new traces record it directly.
Each row above is keyed to its immutable manifest under the stated run root;
the manifest supplies git/config SHA, exact model/dataset revisions, software,
GPU, backend, seed, and measurement type. The shared base git SHA was
`f7f75cd313e0d76d3864ed63e0ab98ca61fc4c9a`; all four manifests record dirty
patch provenance, so they are ineligible for a canonical paper table.

## Unsupported method/model combinations

“Unsupported” here means not eligible for a paper comparison today. A source
class or a fail-safe fallback is not experimental support.

| Model(s) | Method(s) | Status and exact reason |
|---|---|---|
| All five audited HF models | `mosaickv_proto`, `mosaickv_full` below retention 1.0 | Unsupported. Every adapter declares post-RoPE keys, `supports_prototype_merge=False`, and `supports_residual_repair=False`; the runtime uses an exact-only safety fallback. See the capability declarations for [LLaVA-1.5](../src/mosaickv/adapters/huggingface/llava.py#L24), [Qwen2.5-VL](../src/mosaickv/adapters/huggingface/qwen2_5_vl.py#L26), [OneVision](../src/mosaickv/adapters/huggingface/llava_onevision.py#L24), and [InternVL](../src/mosaickv/adapters/huggingface/internvl.py#L49). |
| Qwen2.5-VL 3B/7B, LLaVA-OneVision 0.5B, InternVL2.5 4B | `lookm_reimpl` | Unsupported. The current reimplementation is restricted to the LLaVA-1.5 MHA path and rejects unequal query/KV head counts; no cross-family paper-faithful parity gate exists. |
| Qwen2.5-VL 3B/7B, LLaVA-OneVision 0.5B, InternVL2.5 4B | paper-faithful `prefixkv_reimpl` | Unsupported under that claim. The executable path is explicitly `generalized_prefixkv_reimpl`; it is not official or LLaVA-parity PrefixKV. |
| Qwen2.5-VL 7B, LLaVA-OneVision 0.5B, InternVL2.5 4B | all nine paper-result rows | Unsupported pending an exact-revision checkpoint/runtime acceptance gate. Static adapter registration is not load evidence. |
| LLaVA-1.5 7B and Qwen2.5-VL 3B | `vl_cache_reimpl` paper-result rows | Unsupported pending a checkpoint-specific VL-Cache run and retention-1 parity gate. The implementation and synthetic equation tests alone are insufficient. |
| Every model | every method on vLLM, SGLang, SDPA, or FlashAttention-2 | Unsupported in this audit. The unified compression runtime is HF eager-only. |

At retention 1.0, every compression method is required to take the exact
no-transform path. That boundary test does not establish support for prototype
merging or residual repair below 1.0.

## Remaining threats to fair comparison

1. **No byte-budget-aware physical planner.** Selection budgets do not account
   for the global padding capacity imposed by the HF cache packer. Until fixed,
   run a pilot, read `packed_prefill_active_kv_bytes`, and reject or retarget
   the row before evaluation. Do not round methods into coarse ratio buckets.
2. **Input hashes are not content hashes.** Current HF manifest prompt/media
   hashes encode the selected IDs and references, while preprocessing and
   tokenization hashes encode labels plus model identity. Preserve canonical
   prompt strings, media content hashes, rendered chat text, processor config,
   and final `input_ids`/media tensor digests in the next manifest schema.
3. **No cohort identity or preflight gate.** Separate manifests can differ in
   any controlled field without an automated rejection. A comparison tool must
   require equal model/config-input/generation/backend hashes and one explicit
   `comparison_cohort_id`.
4. **Mandatory policies differ.** Report common mandatory bytes and additional
   method-specific protected bytes. Include an ablation only if it is labeled
   as a non-paper variant.
5. **Granularity differs.** Existing MosaicKV runs use block size 16 while the
   three published reimplementations require token blocks. Do not combine them
   in one primary table without a block-size-1 MosaicKV rerun.
6. **FullKV timing is on a different runner.** The dedicated reference has
   warmups, repeats, CUDA-event phase timing, GPU telemetry, and per-trial rows;
   compressed methods currently have single evaluation invocations and a host
   total. No latency or throughput comparison is valid yet.
7. **Method-specific instrumentation changes prefill.** Attention/query capture
   differs by method. Include required capture in end-to-end TTFT and report a
   common untouched-prefill diagnostic separately.
8. **Final cache bytes depend on decode maintenance.** PrefixKV can evict while
   other methods grow, and repair can promote blocks. Equal prompt budgets do
   not imply equal final bytes. Match output length and report both prefill and
   per-step/final active bytes.
9. **Residual memory is outside the primary active budget.** `mosaickv_full`
   must separately report CPU residual bytes, transfer traffic, and total
   stored bytes; otherwise device-cache equality can hide a memory advantage.
10. **Published-code parity is incomplete.** LOOK-M uses a different official
    checkpoint/backend/chat template; PrefixKV official profiles can lack
    recoverable calibration IDs; VL-Cache has no assumed official code. Never
    merge official and `_reimpl` rows.
11. **Calibration provenance remains a leakage risk.** PrefixKV profiles and
    any tuned VL-Cache ambiguity settings need immutable, disjoint calibration
    sample IDs before evaluation starts.
12. **Hardware-state controls are incomplete in the unified runner.** GPU
    clocks, power state, background processes, and process concurrency are
    captured by the dedicated FullKV measurement stack, not by every unified
    method run.

## Minimum protocol for a valid baseline table

1. Freeze one ordered sample ledger containing prompt bytes, media hashes,
   rendered chat strings, processor outputs, and token IDs.
2. Use one exact model revision, processor revision, dtype, eager backend,
   seed, fixed output length, warmup count, trial count, and GPU allocation.
3. Use `block_size=1` for the primary nine-method stratum. Put block-16
   MosaicKV in a separately labeled systems/ablation stratum.
4. Choose a representable byte target `B`. Run packing preflight for every
   method and admit a row only when physical prefill bytes equal `B` exactly.
5. Keep FullKV as the uncompressed reference at its source byte count; do not
   pretend it is a compressed byte-matched point.
6. Require requested and effective method labels to agree, except for the
   explicit retention-1 no-transform suffix.
7. Run all methods through one repeated-trial measurement harness and retain
   every per-trial observation and failure.
8. Join or compare rows by actual byte fields, never by nominal retention
   ratio alone.
