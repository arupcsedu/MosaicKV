# Model capability matrix

Audit target: the installed common environment at
`/scratch/djy8hg/env/mosaickv` on 2026-08-02. “Source” means the installed
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
| `llava-hf/llava-onevision-qwen2-7b-ov-hf` | `0d50680527681998e456c7b78950205bedd8a068` |

## Current common-environment status

| Model | HF 4.55.2 source | vLLM 0.10.2 source | SGLang 0.5.1 source | Current runtime evidence |
|---|---|---|---|---|
| LLaVA-1.5-7B | Native | Registered | Checkpoint compatibility unverified | HF eager controlled FullKV/exact-selection job `17346904_2` completed; fused parity remains model-specific |
| Qwen2.5-VL-3B | Native | Registered | Registered | HF eager/SDPA/FA2 FullKV parity; HF eager three-tier/repair path measured in bounded v17; serving Stage A parity passes |
| Qwen2.5-VL-7B | Native | Registered | Registered | HF loads; serving reruns pending |
| InternVL2.5-4B | Remote checkpoint code only | Registered | Unsupported | Optional HF remote-code path unverified in upgraded lock |
| LLaVA-OneVision-0.5B | Native | Registered | Unsupported | HF source/load gate pending |
| LLaVA-OneVision-7B | Native | Source-registered; backend checkpoint gate pending | Unsupported | HF BF16 job `17856497` passed 16-token parity, retention-one identity, real prototypes, and repair |

Installed Transformers classes are
`transformers/models/llava/modeling_llava.py:325`,
`transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1377`, and
`transformers/models/llava_onevision/modeling_llava_onevision.py:710`.
InternVL requires the checkpoint's `trust_remote_code` implementation, so it
is not marked native.

vLLM registrations are in `vllm/model_executor/models/registry.py`:
`InternVLChatModel` at line 227, LLaVA at line 238, OneVision at line 241, and
Qwen2.5-VL at line 259. SGLang defines Qwen2.5-VL at
`sglang/srt/models/qwen2_5_vl.py:442` and exports it at line 623. Its
SGLang-specific LLaVA implementation exports architecture
names, not the requested checkpoint's HF architecture name.

## Media capability

| Model | Image | Multiple images | Video |
|---|---|---|---|
| LLaVA-1.5-7B | Declared | Unsupported | Unsupported |
| Qwen2.5-VL-3B | Declared | Declared | Declared |
| Qwen2.5-VL-7B | Declared | Declared | Declared |
| InternVL2.5-4B | Declared by checkpoint code | Declared by checkpoint code | Declared by checkpoint code |
| LLaVA-OneVision-0.5B | Declared | Declared | Declared |
| LLaVA-OneVision-7B | Declared | Declared | Declared |

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

Qwen2.5-VL, LLaVA-1.5, and LLaVA-OneVision declare prototype/repair support.
Those declarations are restricted to Transformers 4.55.2 eager attention and
their registered exact M-RoPE or standard-RoPE phase transforms. Every
checkpoint retains an independent GPU gate; source-family compatibility does
not substitute for measured checkpoint evidence. InternVL remains exact-only.

For serving backends, public APIs do not expose model-owned query projections
or request KV tensors safely. Those capabilities are unsupported regardless of
source registration.

## Minimal executable matrix

The bounded validation matrix starts with Qwen2.5-VL-3B, LLaVA-1.5, and the
pinned OneVision-7B checkpoint through HF eager; OneVision-7B must use uniform
BF16. FullKV Qwen2.5-VL-3B is also gated through HF SDPA/FlashAttention-2 and
the verified vLLM/SGLang Stage-A paths. Qwen-7B remains checkpoint-specific;
no family-level result is inherited without its own gate.
