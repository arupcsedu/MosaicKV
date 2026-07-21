# LOOK-M paper specification and local reimplementation

Status: implemented as `lookm_reimpl`; **not official LOOK-M code**. The
author repository is preserved as the `third_party/LOOK-M` submodule at
`ecf0f51a9c416c2d85e47faf2638502f01a6d748`. Its MIT license and attribution
remain in `third_party/LOOK-M/LICENSE`.

Primary sources:

- Wan et al., “LOOK-M: Look-Once Optimization in KV Cache for Efficient
  Multimodal Long-Context Inference,” Findings of EMNLP 2024, pp. 4065–4078,
  DOI [`10.18653/v1/2024.findings-emnlp.235`](https://doi.org/10.18653/v1/2024.findings-emnlp.235).
- Official code: [`SUSTechBruce/LOOK-M`](https://github.com/SUSTechBruce/LOOK-M/tree/ecf0f51a9c416c2d85e47faf2638502f01a6d748),
  pinned to the SHA above.

This specification separates the published equations from behavior observed
in the pinned source. Paper results and official-code results must use the
label `official_lookm`; unified-runtime results must use `lookm_reimpl` and
measurement type `baseline_reimpl_measured`.

## Algorithm implemented by `lookm_reimpl`

Let `L` be the prompt-cache length for one transformer layer and head, `A_p`
the eager prompt attention probabilities, `T` the text-token indices,
`M = floor(alpha_recent * L)`, and `N = floor(alpha_important * L)`.

### Text-prior scoring

Following paper equations 4–5, the cumulative score of key position `j` is

```text
A_s[j] = sum_i A_p[i, j]
T_p    = max_j A_s[j]
A_s[j] = A_s[j] + T_p, for j in T
```

Scores are computed separately in every layer and attention head. The local
implementation consumes eager attention shaped `[1, heads, queries, keys]`.
It does not reuse MosaicKV forecasts, evidence graphs, block utility, or
submodular selection.

The pinned source accumulates prompt attention at
`LLaVA-mix_merge_v1/llava/model/kv_token_merge/modify_llama.py:715-725`.
Its text-prior classes implement a different shortcut at lines 679–693: they
subtract `65516` from hard-coded 576-token image spans and the recent window,
select top-k positions, reset selected/recent scores to zero, and retain the
positions whose resulting score is nonnegative. That can retain every text
position and need not produce exactly `N+M` slots. `lookm_reimpl` implements
the published max-score prior and exact `N+M` selection, not that shortcut.

### Conserved tokens and mandatory rules

Following paper equations 6–8:

1. The last `M` positions form the recent window and are mandatory.
2. The top `N` text-prior scores are selected from positions `[0, L-M)`.
3. Conserved keys and values are gathered in increasing original position
   order.

The paper does not define BOS, system-prompt, image-boundary, or other special
tokens as independently mandatory. `lookm_reimpl` therefore adds no MosaicKV
mandatory-token policy beyond the recent window. Text tokens receive the
score prior; they are not categorically mandatory. Ties, which the paper does
not specify, are broken deterministically by the lower physical position.

At retention ratio `1.0`, MosaicKV's repository-wide reference contract takes
precedence: all source positions are retained in order and no merging is
performed. This is a correctness extension, not a reported LOOK-M paper
setting.

### Pivot selection

For every evicted key `k_e`, equation 9 assigns the conserved key with maximum
cosine similarity:

```text
pivot(e) = argmax_c cosine(k_e, k_c)
```

Matching is many-to-one, independent per layer and head. The same assignment
is used for values, as stated in the paper. A zero-norm key uses an epsilon
denominator, and `argmax` chooses the earliest conserved position on a tie.
No MosaicKV graph edge, modality compatibility constraint, or positional-span
constraint participates in LOOK-M pivoting.

### KV merge strategies

For a conserved pivot `x` with assigned evicted members `y_i` and
`n = number of members`, the implementation provides all three paper
strategies and applies the key-derived assignment/weight to both K and V:

- `averaged`, equation 10:
  `(x + sum_i y_i) / (n + 1)`.
- `pivotal`, equation 11:
  `(x + sum_i ((y_i + x) / 2)) / (n + 1)`.
- `weighted`, equation 12:
  `(x + sum_i (cosine(y_i, x) * y_i)) / (n + 1)`.

The default is `pivotal`, matching the official example script's
`text_prior_pivot_merge` mode. Merge computation stays in the source tensor
dtype because that is what the paper and official source do; it does not use
MosaicKV's FP32 prototype construction, prototype tier, residual tier, or
repair path.

## Layer, head, cache, and position behavior

- Selection and merging run independently for every decoder layer and head.
- The common cache interface uses token-sized blocks (`block_size: 1`) because
  LOOK-M selects individual KV positions.
- The initial supported path requires multi-head attention with equal query
  and KV head counts. The pinned LLaVA-1.5 code operates on `[batch, 32,
  sequence, 128]` head tensors. Grouped-query attention is unsupported; no
  undocumented head aggregation is substituted.
- The adapter supplies per-position original logical indices. The packed
  runtime preserves these indices separately from its shorter active cache
  length and uses the original prompt length as the next decode position.
- Hugging Face LLaVA and the pinned official path cache keys after RoPE. The
  official attention computes RoPE before concatenating and compressing the
  cache (`modify_llama.py:1376-1387,1417-1418`). LOOK-M then averages those
  rotated keys even when positions differ. `lookm_reimpl` reproduces that
  LOOK-M assumption; it does not claim that these merged positions are
  MosaicKV prototypes or that cross-phase averaging is generally safe.
- Every active byte count is computed from the realized merged K/V tensor
  storage, not estimated from the requested ratio.

## Supported model assumptions

The paper evaluates long-context multi-image settings with LLaVA-1.5,
MobileVLM, Yi-VL, and InternVL-family configurations. The pinned example uses
`liuhaotian/llava-v1.5-7b`, a 4096-token maximum context, 576 tokens per image,
greedy decoding, and up to 512 new tokens
(`configs/model_configs.yaml:1-11`). The official LLaVA worker uses its vendored
LLaVA stack and the `llava_llama_2` conversation template
(`workers/model_workers.py:8-43,67-109`).

The unified implementation currently supports only the registered eager
LLaVA-1.5 adapter and batch size one. It accepts the Transformers conversion
`llava-hf/llava-1.5-7b-hf` for local execution, but that conversion is not the
same model ID/backend executable as official LOOK-M. Qwen2.5-VL,
LLaVA-OneVision, InternVL, SDPA, FlashAttention-2, vLLM, and SGLang are marked
unsupported for `lookm_reimpl` until a paper/source justification and a
controlled parity gate exist.

## Unified runtime mapping

| Concern | Unified implementation |
|---|---|
| Strict configuration and ratios | `src/mosaickv/config.py::LookMConfig` |
| Method identity | `MosaicKVMethod.LOOKM_REIMPL` (`lookm_reimpl`) |
| Scoring, selection, pivoting, merging | `src/mosaickv/baselines/lookm.py` |
| Common cache packing and decode | `src/mosaickv/backends/hf_runtime.py` |
| CLI | `mosaickv evaluate --method lookm_reimpl` |
| Controlled artifact comparison | `src/mosaickv/baselines/lookm_parity.py` |
| Example config | `configs/hf_lookm_reimpl.yaml` |

Example:

```bash
mosaickv evaluate \
  --config mosaickv/configs/hf_lookm_reimpl.yaml \
  --task synthetic_smoke \
  --local-files-only
```

The trace contains `implementation: lookm_reimpl`, `official_code: false`, the
official source SHA, cumulative-score summaries and digest, selected positions,
pivot-assignment count/digest and cosine range, merged bytes, and common timing
fields. The compact digests keep development traces bounded without discarding
the ability to compare an exported tensor trace byte-for-byte. It never labels
a unified run as official LOOK-M.

On the audited cluster, the one-sample cached-checkpoint reimplementation smoke
is submitted separately from any official run:

```bash
mkdir -p /scratch/djy8hg/runs/mosaickv/lookm/slurm-logs
sbatch --output=/scratch/djy8hg/runs/mosaickv/lookm/slurm-logs/%x-%j.out \
  mosaickv/slurm/lookm_reimpl_smoke.sbatch
```

This command validates only `lookm_reimpl`. It must not populate an
`official_lookm` parity artifact.

## Known source and runtime deviations

| Topic | Paper | Pinned official source | `lookm_reimpl` |
|---|---|---|---|
| Text prior | Add `max(A_s)` to text scores | Hard-mask 576-position image spans; resulting cache can exceed `N+M` | Paper equations 4–8 |
| Image span | Modality-derived text indices | Assumes 576 tokens/image and mutates offsets by 575 per preceding image | Adapter modality map; no fixed image length |
| Tie handling | Unspecified | `torch.topk`/`max` implementation-dependent tie behavior | Lower physical position |
| Numerical guard | Unspecified | Direct norm division | Epsilon for zero-norm keys |
| Checkpoint/backend | Research LLaVA path | Original/vendored `liuhaotian/llava-v1.5-7b` | Transformers conversion unless an identical compatible checkpoint is supplied |
| Cache positions | Not explicit | Legacy custom attention | Original logical positions retained explicitly |
| Retention 1.0 | Not evaluated as a special case | Compression callback may bypass when `L <= N+M` | Guaranteed no-transform FullKV-equivalent path |

The checkpoint/backend row prevents an official-vs-reimplementation result
from being called directly comparable today. See
[`lookm_parity_report.md`](lookm_parity_report.md). No numerical parity value
may be filled until the comparison validator reports `status: comparable`.
