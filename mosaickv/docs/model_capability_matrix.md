# Model capability matrix

Audit date: 2026-07-19. Model revisions are immutable SHAs and must be used in
all experiments. This document separates three facts that are often conflated:

1. **architecture support**: the installed source contains a matching model
   class/registry entry;
2. **modality support**: the checkpoint and processor define that input type;
3. **load verified**: the exact weights were loaded in the audited environment.

No requested checkpoint had complete local weights and no GPU was visible on
the login node, so **load verified is unsupported for every row**. A source
support mark is not an experimental result.

## Pinned checkpoints

| Model ID | Audited revision | Text architecture | Nominal checkpoint dtype |
|---|---|---|---|
| `llava-hf/llava-1.5-7b-hf` | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` | Llama/Vicuna | float16 |
| `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` | Qwen2.5-VL/Qwen2 | bfloat16 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | Qwen2.5-VL/Qwen2 | bfloat16 |
| `OpenGVLab/InternVL2_5-4B` | `2cf4a8158bbc40d35015e7c63b527890de4d27b3` | InternVLChat + Qwen2.5-3B | bfloat16 |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | `74dd0bf867a4cda7950c17663794267c60cf4b40` | LLaVA-OneVision + Qwen2 | float16 config; card describes bfloat16 training |

Configuration and model-card evidence is pinned through the respective model
repositories: [LLaVA 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/b234b804b114d9e37bb655e11cbbb5f5e971b7a9),
[Qwen2.5-VL 3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3),
[Qwen2.5-VL 7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5),
[InternVL2.5 4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B/tree/2cf4a8158bbc40d35015e7c63b527890de4d27b3),
and [LLaVA-OneVision 0.5B](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/tree/74dd0bf867a4cda7950c17663794267c60cf4b40).

## Architecture/backend support

“Source-supported” below means only that the audited implementation has the
exact architecture entry. The current runtime column is the answer to whether
the environment can actually load it now.

| Model | HF Transformers 4.57.6 | vLLM 0.11.2 source | SGLang 0.5.10.post1 source | Current exact-revision load |
|---|---|---|---|---|
| LLaVA 1.5 7B | **Source-supported, native** `LlavaForConditionalGeneration` | **Source-supported** | **Source-supported** | **Unsupported**: weights absent; no GPU; serving env broken |
| Qwen2.5-VL 3B | **Source-supported, native** `Qwen2_5_VLForConditionalGeneration` | **Source-supported** | **Source-supported** | **Unsupported**: weights absent; no GPU; serving env broken |
| Qwen2.5-VL 7B | **Source-supported, native** (same class) | **Source-supported** | **Source-supported** | **Unsupported**: cached ref only, no snapshot/weights; no GPU; serving env broken |
| InternVL2.5 4B | **Source-supported only through checkpoint remote code** with `trust_remote_code=True`; not a native class for this `internvl_chat` config | **Source-supported** | **Source-supported** | **Unsupported**: remote code/weights absent; no GPU; serving env broken |
| LLaVA-OneVision 0.5B | **Source-supported, native** `LlavaOnevisionForConditionalGeneration` | **Source-supported** | **Unsupported**: no `LlavaOnevisionForConditionalGeneration` registry/model implementation | **Unsupported** |

Installed-source evidence:

- Transformers classes are in
  `transformers/models/llava/modeling_llava.py:308`,
  `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1358`, and
  `transformers/models/llava_onevision/modeling_llava_onevision.py:661` under
  `drc_rag_benchmarks_yml_20260421/lib/python3.11/site-packages`.
- The InternVL config's `auto_map` points to
  `modeling_internvl_chat.InternVLChatModel`. At the pinned revision that class
  defines `self.language_model` with `Qwen2ForCausalLM` and forwards
  `past_key_values`; this source is checkpoint remote code, not installed code:
  [pinned file](https://huggingface.co/OpenGVLab/InternVL2_5-4B/blob/2cf4a8158bbc40d35015e7c63b527890de4d27b3/modeling_internvl_chat.py).
- vLLM registers `InternVLChatModel` at
  `vllm/model_executor/models/registry.py:289`, LLaVA at `:317`,
  LLaVA-OneVision at `:326-328`, and Qwen2.5-VL at `:360-362`. Exact classes
  are `models/llava.py:508`, `qwen2_5_vl.py:1058`, `internvl.py:1076`, and
  `llava_onevision.py:481` under `drc_rag_bench_env`.
- SGLang has `models/llava.py:618`, `models/qwen2_5_vl.py:554`, and
  `models/internvl.py:493`; `srt/configs/model_config.py:1312`, `:1325`, and
  `:1332` list their architecture names. A repository-wide search found no
  OneVision class or registry entry; a comment mentioning its config is not
  implementation support.

## Input modalities

These marks describe checkpoint/processor capability, not successful execution
in the current environment.

| Model | Image | Multiple images in one request | Video | Exact evidence |
|---|---:|---:|---:|---|
| LLaVA 1.5 7B | Yes | Yes | **Unsupported** | The [model card](https://huggingface.co/llava-hf/llava-1.5-7b-hf) states multi-image support; installed vLLM `models/llava.py:175-176` returns unbounded image items and `:531` rejects non-image modalities. |
| Qwen2.5-VL 3B | Yes | Yes | Yes | The [model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) demonstrates multiple image items and video input. Installed vLLM `models/qwen2_vl.py:952-953` returns unbounded image/video limits; SGLang `multimodal/processors/qwen_vl.py:308-365` handles image and video modality items. |
| Qwen2.5-VL 7B | Yes | Yes | Yes | Same processor architecture as the 3B row; [7B model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). |
| InternVL2.5 4B | Yes | Yes | Yes | The [model card](https://huggingface.co/OpenGVLab/InternVL2_5-4B) explicitly defines single-image, multi-image, and frame-based video handling. vLLM `models/internvl.py:900-902` conditionally adds video; SGLang `multimodal/processors/internvl.py:271-347` and `:407-610` handle image/video lists and modality items. |
| LLaVA-OneVision 0.5B | Yes | Yes | Yes | The [model card](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) identifies single-image, multi-image, and video scenarios. vLLM `models/llava_onevision.py:136-137` returns unbounded image/video limits. SGLang remains unsupported for this architecture. |

Backend modality support can be narrower than the checkpoint. In particular,
SGLang's generic LLaVA processor contains multi-image/video-like branches, but
that does not turn the LLaVA 1.5 checkpoint into a video model or supply the
missing OneVision model implementation.

## Research access and cache layout

All numeric shapes below are derived from the pinned text configuration and the
installed Transformers cache contract. Transformers
`cache_utils.py:84-121` defines each dynamic K or V layer as
`[batch, num_kv_heads, sequence, head_dim]` and concatenates along sequence.

| Model | `past_key_values` accessible in HF | Query/`q_proj` accessible in HF | Per-layer HF K and V layout |
|---|---|---|---|
| LLaVA 1.5 7B | Yes, wrapper forward returns the language-model `Cache` | Yes: `model.language_model.layers[i].self_attn.q_proj`; query tensor requires a hook | 32 layers; each K and V `[B, 32, S, 128]` |
| Qwen2.5-VL 3B | Yes | Yes: `model.language_model.layers[i].self_attn.q_proj`; installed attention computes `query_states` at `modeling_qwen2_5_vl.py:645` | 36 layers; each K and V `[B, 2, S, 128]` |
| Qwen2.5-VL 7B | Yes | Yes, same module path | 28 layers; each K and V `[B, 4, S, 128]` |
| InternVL2.5 4B | Yes through remote wrapper `model.language_model` | Yes through its Qwen2 language model; installed `Qwen2Attention.q_proj` is at `modeling_qwen2.py:134` | 36 layers; each K and V `[B, 2, S, 128]` |
| LLaVA-OneVision 0.5B | Yes, wrapper forward returns the language-model `Cache` | Yes: `model.language_model.layers[i].self_attn.q_proj` | 24 layers; each K and V `[B, 2, S, 64]` |

For LLaVA 1.5, the pinned nested config names Vicuna/Llama but omits standard
Llama dimensions. Installed `configuration_llama.py:156-168` supplies hidden
size 4096, 32 layers, 32 attention heads, and defaults KV heads to attention
heads at `:195-198`, giving head dimension 128. The other values are explicit
in their pinned configs.

The accessible object is the **post-RoPE K cache and V cache**, while a forward
hook on `q_proj` observes pre-reshape/pre-RoPE projection output. MosaicKV must
document which query representation it uses. Qwen2.5-VL and OneVision use
multimodal rotary positions, so preserving only a compacted physical order is
not sufficient; original logical position tensors must remain attached.

Neither serving backend exposes these research objects through its public
generation API. vLLM instead uses an internal paged tensor shaped
`[2, num_blocks, block_size, num_kv_heads, head_size]`
(`vllm/v1/attention/backends/flash_attn.py:90-110`). SGLang's MHA pool uses one
per-layer K tensor `[slots + page_size, num_kv_heads, head_dim]` and analogous V
tensor (`sglang/srt/mem_cache/memory_pool.py:845-869`). See the
[backend matrix](backend_capability_matrix.md) before interpreting internal
tensor reachability as supported injection.

## Minimal recommended model/backend matrix

The detected general partition includes A100 80 GB GPUs, but the current
software cannot execute this matrix until clean environments and pinned model
snapshots exist. Once those prerequisites are satisfied, use the following
smallest defensible progression:

| Purpose | Model | Backend | Hardware target | Rationale |
|---|---|---|---|---|
| Smoke, cache shape, retention-1.0 correctness | LLaVA-OneVision 0.5B | HF eager | 1× A100 (40 or 80 GB) | Smallest checkpoint; native Transformers class; exercises image, multi-image, video, and Qwen2 positions. |
| Primary algorithm/quality bring-up | Qwen2.5-VL 3B | HF eager | 1× A100 80 GB | Full requested modalities, GQA cache, M-RoPE, and native source hooks. |
| Cross-family and published-baseline bridge | LLaVA 1.5 7B | HF eager; official baselines in isolated legacy envs | 1× A100 80 GB | Llama/MHA cache and closest checkpoint family to LOOK-M/PrefixKV. Do not equate HF conversion with their vendored model. |
| Scaling check | Qwen2.5-VL 7B | HF eager | 1× A100 80 GB | Same architecture at larger scale; add only after 3B correctness. |
| Systems prototype | Qwen2.5-VL 3B | separately pinned vLLM, then SGLang | 1× A100 80 GB | Both installed source trees have the architecture and image/video processors; backend hooks still require implementation. |

InternVL2.5 4B is valuable as a later remote-code architecture test, but it is
not minimal: it introduces custom checkpoint code and a distinct image tiling
pipeline before the reference path is stable. OneVision is not a SGLang target
for the audited version.
