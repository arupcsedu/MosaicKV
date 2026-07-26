# Experiment 2: systems microbenchmark validation

## Scope and eligibility

The bounded campaign contains 440 timed trial rows: 440 completed and 0 failed. Every point used 5 warmups and 20 timed repetitions.

The source tree contained the explicitly staged implementation patch. These are validation rows, not paper-eligible final measurements; every manifest records the Git SHA and patch SHA.

The immutable point manifests contain 2 worktree fingerprint(s): `13da87316352688460ad9ee4c8584dd5fa66d5f56a29ffc3a754d464328ea33f`, `ff9b62254b6daeec83e209006866ad29fef2cc88b2d447ab09a0dd1b7643e16f`. A documentation-only line-wrap change occurred before the final Qwen-3B point manifest was written; no executable source changed during the run.

All reported p50/p95 values are derived from per-trial measurements. Missing backend telemetry remains missing; it was not synthesized.

## Backend parity gates

| Backend | Compared trials | Matching trials | Verdict |
|---|---:|---:|---|
| hf_sdpa | 16 generated tokens | 16 | passed for Qwen2.5-VL-3B FullKV; exploratory |
| hf_flash_attention_2 | 16 generated tokens | 16 | passed for Qwen2.5-VL-3B FullKV; exploratory |
| sglang | 4 repeated trials plus isolation probes | 32/32 versus controlled HF SDPA | Stage A and cross-parity passed; exploratory |
| vllm | 2 uncached trials | 16/16 versus controlled HF | Stage A and cross-parity passed; exploratory |

These newer rows are superseding implementation diagnostics and are not mixed
into the original 440-row aggregate above. HF SDPA job `17333707` completed
five warmups and 20 trials after job `17346718` passed real-checkpoint parity.
FlashAttention environment job `17347802`, parity job `17348171`, and
performance job `17349594_0` also passed. SGLang Stage A job `17349428` passed
determinism, active-byte accounting, and request isolation; its same-input HF
comparison passed after aligning the recorded Qwen resize profile. vLLM job `17349373` isolated prefix-cache
cold/warm token divergence, so prefix caching is now opt-in and fail-closed for
correctness runs; uncached job `17351185` passed Stage A and HF parity. Native vLLM/SGLang MosaicKV
remains unsupported, and no simulated native row exists.

## Supported coverage

HF eager measures FullKV and exact-only MosaicKV for Qwen-3B/7B. The LLaVA
packing failure was fixed with vectorized per-layer gather: bounded job
`17346904_2` completed five warmups and 20 trials for all four requested
FullKV/exact points in 17m42s. Prototype merging and repair are absent because
all audited cached keys are post-RoPE and no safe phase-aware prototype
transform exists. Consequently residual-transfer time, repair time, and repair
frequency are measured zeros only where the method definition guarantees no
repair.

HF adapters are batch-size one. vLLM Stage A passes, but batch sizes 2 and 4
have not been run under the corrected common-lock path. No sequential HF proxy
is labeled as batching.

## Compression-overhead regimes

Compression overhead is compared with the paired FullKV p50 decode-time saving at the same model, backend, resolution, image count, output length, and batch size.

| Model | Method | Retention | Output | Compression p50 (s) | Decode saving p50 (s) | E2E delta (s) | Overhead outweighs saving |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-VL-3B-Instruct | mosaickv_exact | 0.1 | 256 | 0.936 | -2.407 | 3.943 | yes |
| Qwen2.5-VL-3B-Instruct | mosaickv_exact | 0.1 | 64 | 1.064 | -0.577 | 1.972 | yes |
| Qwen2.5-VL-3B-Instruct | mosaickv_exact | 0.2 | 256 | 1.419 | -2.391 | 4.277 | yes |
| Qwen2.5-VL-3B-Instruct | mosaickv_exact | 0.2 | 64 | 1.350 | -0.582 | 2.337 | yes |
| Qwen2.5-VL-3B-Instruct | mosaickv_exact | 0.4 | 256 | 2.106 | -2.416 | 5.029 | yes |
| Qwen2.5-VL-3B-Instruct | mosaickv_exact | 0.4 | 64 | 2.107 | -0.582 | 3.052 | yes |
| Qwen2.5-VL-7B-Instruct | mosaickv_exact | 0.1 | 256 | 1.467 | -1.744 | 3.779 | yes |
| Qwen2.5-VL-7B-Instruct | mosaickv_exact | 0.1 | 64 | 1.453 | -0.431 | 2.350 | yes |
| Qwen2.5-VL-7B-Instruct | mosaickv_exact | 0.2 | 64 | 1.992 | -0.430 | 2.910 | yes |
| Qwen2.5-VL-7B-Instruct | mosaickv_exact | 0.4 | 64 | 3.076 | -0.429 | 4.025 | yes |

## Updated decode-latency finding

The original 10 aggregate regimes predate the dense exact-cache fast path and
must not be used to characterize the current implementation. They remain in
the immutable old aggregate for provenance.

Current fixed-block selection aligns a shared physical extent, packs one
vectorized gather per layer, omits model-side masks for dense all-valid caches,
preallocates append-only decode storage, and preserves logical M-RoPE positions
separately. In matched Qwen2.5-VL-3B output-64 jobs, Prompt Top-K and MosaicKV
exact both used 1,769,472 active bytes; MosaicKV p50 decode was 3.338 s versus
3.396 s for Prompt Top-K. MosaicKV's p50 compression cost was still 0.778 s,
so it remained slower end-to-end. Forecast centroid and graph planning—not
per-token cache masking—are now the primary optimization target.

An explicit eager-prefill/SDPA-decode mode now reduces the current Qwen-3B
retention-0.1 exact point from 3.346 to 2.995 s p50 decode (10.5%) and from
4.762 to 4.400 s p50 end-to-end (7.6%) at identical 589,824 active bytes.
Retention-1 MosaicKV and FullKV produced the same 64-token sequence and exact
active bytes across 20 trials. These dirty-tree diagnostics are not inserted
into the original aggregate, and the hybrid backend must be applied identically
to all methods in any fairness table.

## Unsupported combinations

| Combination | Status | Reason |
|---|---|---|
| HF eager-prefill/SDPA-decode exact methods | conditionally supported | Implemented only for all-valid caches without repair; the backend pair is explicit in config/manifest. |
| HF FlashAttention-2 compressed methods | unsupported | Prompt-attention evidence remains eager-only; specialized SM80 FullKV is verified. |
| SGLang Qwen-3B FullKV Torch-native | supported for validation | Stage A, request isolation, byte accounting, and exact 32/32 HF-SDPA token parity pass. |
| vLLM Qwen-3B FullKV uncached | supported for validation | Installed vLLM 0.10.2 Stage A and exact 16/16 HF token parity pass; prefix-cache cold/warm divergence remains fail closed. |
| vLLM Qwen-7B / LLaVA-7B | unsupported in this bounded run | No matching current-lock parity pass was available before scheduling. |
| Native MosaicKV on vLLM/SGLang | unsupported | No safe request-scoped KV mutation API exists in the audited versions. |
| `mosaickv_proto` / `mosaickv_full` | unsupported | Safe post-RoPE prototype handling has not passed correctness. |
| HF batch 2/4 | unsupported | Explicit HF adapters enforce batch size one. |

## Interpretation limits

This is a controlled synthetic visual workload intended to validate systems instrumentation and expose trends within 2.5 hours. It is not a quality benchmark and does not establish paper claims. Serving Stage-A parity now passes, but those newer diagnostics are not retroactively inserted into the original 440-row HF-eager aggregate.
