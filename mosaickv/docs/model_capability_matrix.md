# Model capability matrix

Audit target: the installed common environment at
`/scratch/djy8hg/env/mosaickv` on 2026-07-21. “Source” means the installed
package contains a matching architecture. “Unsupported” means the exact
checkpoint has not passed a clean common-environment load and parity gate.

## Pinned checkpoints

| Model | Revision |
|---|---|
| `llava-hf/llava-1.5-7b-hf` | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |
| `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| `OpenGVLab/InternVL2_5-4B` | `2cf4a8158bbc40d35015e7c63b527890de4d27b3` |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | `74dd0bf867a4cda7950c17663794267c60cf4b40` |

## Current common-environment status

| Model | HF 4.49 source | vLLM 0.7.2 source | SGLang 0.4.3 source | Exact checkpoint load |
|---|---|---|---|---|
| LLaVA-1.5-7B | Native | Registered | No matching HF architecture entry | Unsupported |
| Qwen2.5-VL-3B | Native | Registered | Registered | Unsupported |
| Qwen2.5-VL-7B | Native | Registered | Registered | Unsupported |
| InternVL2.5-4B | Remote checkpoint code only | Registered | Unsupported | Unsupported |
| LLaVA-OneVision-0.5B | Native | Registered | Unsupported | Unsupported |

Installed Transformers classes are
`transformers/models/llava/modeling_llava.py:247`,
`transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1512`, and
`transformers/models/llava_onevision/modeling_llava_onevision.py:374`.
InternVL requires the checkpoint's `trust_remote_code` implementation, so it
is not marked native.

vLLM registrations are in `vllm/model_executor/models/registry.py`:
`InternVLChatModel` at line 159, LLaVA at line 161, OneVision at line 164, and
Qwen2.5-VL at line 175. SGLang defines Qwen2.5-VL at
`sglang/srt/models/qwen2_5_vl.py:474` and exports it at line 759. Its
`sglang/srt/models/llava.py:574` exports SGLang-specific LLaVA architecture
names, not the requested checkpoint's HF architecture name.

## Media capability

| Model | Image | Multiple images | Video |
|---|---|---|---|
| LLaVA-1.5-7B | Declared | Unsupported | Unsupported |
| Qwen2.5-VL-3B | Declared | Declared | Declared |
| Qwen2.5-VL-7B | Declared | Declared | Declared |
| InternVL2.5-4B | Declared by checkpoint code | Declared by checkpoint code | Declared by checkpoint code |
| LLaVA-OneVision-0.5B | Declared | Declared | Declared |

These are model/processor declarations, not successful execution claims. Every
media mode remains unsupported for measurements until its exact input path is
smoke-tested in the common environment.

## HF cache access

| Adapter | `past_key_values` | `q_proj` hook | Cache layout | Cached key phase |
|---|---|---|---|---|
| LLaVA-1.5 | Accessible | Accessible through Llama attention | `[batch, KV head, sequence, head dim]` | Post-RoPE |
| Qwen2.5-VL | Accessible | Accessible | `[batch, KV head, sequence, head dim]` | Post-RoPE |
| LLaVA-OneVision | Accessible | Accessible through Qwen2 attention | `[batch, KV head, sequence, head dim]` | Post-RoPE |
| InternVL2.5 | Unsupported until remote code loads | Unsupported until load | Unsupported | Unsupported |

For serving backends, public APIs do not expose model-owned query projections
or request KV tensors safely. Those capabilities are unsupported regardless of
source registration.

## Minimal executable matrix

The only current no-download matrix is synthetic CPU/HF plus the common CUDA
environment smoke. A model/backend matrix must not be advertised as executable
until the exact checkpoint and parity jobs pass under the common lock. Start
with Qwen2.5-VL-3B and LLaVA-1.5 through HF eager; add vLLM or SGLang only after
their wrappers are ported and validated.
