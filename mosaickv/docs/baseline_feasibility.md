# Published-baseline feasibility

Audit date: 2026-07-19; LOOK-M and PrefixKV pins updated locally on 2026-07-20.
Both official repositories are unmodified submodules under `third_party/` at
their audited commits. The common MosaicKV environment is the only supported
execution environment. External code must remain under `third_party/`, pinned
to an exact commit, and retain its license and attribution. Any paper-faithful
implementation must be named `*_reimpl` and must never be described as
official code.

## Summary

| Baseline | Official implementation | Exact audit pin | Directly runnable in current environment | Models supplied by authors | License | Feasibility |
|---|---|---|---|---|---|---|
| LOOK-M | Yes, but repository says code is still being organized | `SUSTechBruce/LOOK-M@ecf0f51a9c416c2d85e47faf2638502f01a6d748` | **Unsupported** | Original/vendored LLaVA 1.5 7B/13B; MobileVLM V2 3B/7B; Yi-VL 6B; older InternVL Chat ViT-6B/Vicuna-7B configs | MIT | Use `lookm_reimpl` in the common runtime; official execution is not enabled |
| PrefixKV | Yes | `THU-MIG/PrefixKV@597f1ab032704951550f93bcc8a23f1454b80aa4` | **Unsupported** | Vendored LLaVA 1.5, with 7B documented and 7B/13B prefix configs present | MIT | Use `prefixkv_reimpl` in the common runtime; official execution is not enabled |
| VL-Cache | No official code located for arXiv:2410.23317/ICLR 2025 | paper v1, 2024-10-29 | Local eager-HF `vl_cache_reimpl`; paper-model reproduction not yet run | Paper evaluates LLaVA-v1.6-Mistral-7B and LLaVA-v1.6-34B | Paper: CC BY-NC-SA 4.0; no software license exists | Formula/budget/leakage tests pass; no official parity is possible unless authors release code |

The similarly named 2025 paper/repository “VLCache: Computing 2% Vision Tokens
and Reusing 98%” is a different work and is not implementation evidence for
VL-Cache arXiv:2410.23317.

## LOOK-M

### Availability and pin

The [official repository](https://github.com/SUSTechBruce/LOOK-M) identifies
itself as the implementation of the EMNLP 2024 Findings paper. The audited
`main` head is
[`ecf0f51a9c416c2d85e47faf2638502f01a6d748`](https://github.com/SUSTechBruce/LOOK-M/tree/ecf0f51a9c416c2d85e47faf2638502f01a6d748),
with no release tags. Its README says the environment follows MileBench, lists
“reorganize the code” and “support more models” as TODOs, and says the code is
still being organized.

### Models and implementation coupling

Pinned configuration files identify:

- `liuhaotian/llava-v1.5-7b` and `liuhaotian/llava-v1.5-13b`;
- `mtgv/MobileVLM_V2-3B` and `mtgv/MobileVLM_V2-7B`;
- `01-ai/Yi-VL-6B`;
- `OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B`.

These are not the requested Qwen2.5-VL, InternVL2.5-4B, or OneVision
checkpoints. The LLaVA path is a vendored/original LLaVA stack, not the
`llava-hf/llava-1.5-7b-hf` Transformers conversion.

The core patch is
[`LLaVA-mix_merge_v1/llava/model/kv_token_merge/modify_llama.py`](https://github.com/SUSTechBruce/LOOK-M/blob/ecf0f51a9c416c2d85e47faf2638502f01a6d748/LLaVA-mix_merge_v1/llava/model/kv_token_merge/modify_llama.py).
It imports a vendored `v433_modeling_llama`, includes an author-machine absolute
`sys.path` at line 22, and the tree contains committed Python bytecode. This is
strong evidence against importing it into the primary MosaicKV environment.

### Pinned requirements

The exact [requirements file](https://github.com/SUSTechBruce/LOOK-M/blob/ecf0f51a9c416c2d85e47faf2638502f01a6d748/requirements.txt)
pins, among many packages:

| Package | Required version |
|---|---:|
| torch | `2.1.1+cu118` |
| torchvision / torchaudio | `0.16.1+cu118` / `2.1.1+cu118` |
| transformers | `4.37.0` |
| accelerate | `0.24.0` |
| flash-attn | `2.3.4` |
| datasets | `2.15.0` |
| xformers | `0.0.23+cu118` |
| triton | `2.1.0` |

Python is not pinned in the repository. These versions conflict materially
with the common stack (torch 2.5.1, transformers 4.49.0, accelerate 1.13.0,
datasets 4.1.1, and CUDA 12.4). Official LOOK-M execution is therefore
unsupported under the single-environment policy. Changing its dependencies
and calling the result official would invalidate paper fidelity; use the
separately labeled `lookm_reimpl` path.

### License and decision

The pinned [license](https://github.com/SUSTechBruce/LOOK-M/blob/ecf0f51a9c416c2d85e47faf2638502f01a6d748/LICENSE)
is MIT, copyright 2024 Bruce.wan. Feasibility is **conditional**: the official
code is preserved unchanged under `third_party/LOOK-M`; portability changes
live as isolated patches under `third_party/patches/LOOK-M`. Report results as
official LOOK-M only on the authors' supported checkpoint/configuration. The
paper-equation port to the HF conversion is the distinct `lookm_reimpl` method
documented in [`docs/baselines/lookm_spec.md`](baselines/lookm_spec.md), never
official code.

## PrefixKV

### Availability and pin

The [official repository](https://github.com/THU-MIG/PrefixKV) supplies PyTorch
code. The audited `master` head is
[`597f1ab032704951550f93bcc8a23f1454b80aa4`](https://github.com/THU-MIG/PrefixKV/tree/597f1ab032704951550f93bcc8a23f1454b80aa4),
with no release tags.

### Models and implementation coupling

The README instructs users to download LLaVA-1.5-7B and demonstrates only that
model. The tree also contains `prefixkv_llava-v1.5-7b_*` and
`prefixkv_llava-v1.5-13b_*` configuration files. It vendors the original LLaVA
package and modifies its attention behavior through
[`patch_attention_forward.py`](https://github.com/THU-MIG/PrefixKV/blob/597f1ab032704951550f93bcc8a23f1454b80aa4/patch_attention_forward.py)
and
[`prefixkv.py`](https://github.com/THU-MIG/PrefixKV/blob/597f1ab032704951550f93bcc8a23f1454b80aa4/prefixkv.py).
`PrefixKV.__call__` consumes `past_key_values` and attentions; its default
sequence dimensions and layer count are part of a legacy model-specific patch,
not a backend-neutral API.

The repository does not provide Qwen2.5-VL, InternVL2.5, OneVision, vLLM, or
SGLang integration. `llava-hf/llava-1.5-7b-hf` is related in weights/model
family but is not the authors' vendored execution path.

### Pinned requirements

The README creates Python 3.8. The exact
[requirements file](https://github.com/THU-MIG/PrefixKV/blob/597f1ab032704951550f93bcc8a23f1454b80aa4/requirements.txt)
pins:

| Package | Required version |
|---|---:|
| Python | `3.8` (README) |
| torch / torchvision | `2.1.2` / `0.16.2` |
| transformers / tokenizers | `4.31.0` / `0.13.1` |
| accelerate | `0.21.0` |
| numpy / scikit-learn | `1.23` / `1.2.2` |
| httpx | `0.24.0` |
| einops / einops-exts | `0.6.1` / `0.0.4` |
| timm | `0.6.13` |

This is also incompatible with the current reference stack and must remain
isolated.

### License and decision

The pinned [license](https://github.com/THU-MIG/PrefixKV/blob/597f1ab032704951550f93bcc8a23f1454b80aa4/LICENSE)
is MIT, copyright 2024 Zuyan Liu. Official execution is unsupported in the
common environment. A modern Transformers or different-model port must be
named `prefixkv_reimpl`, and any offline prefix calibration must use a
declared, non-test calibration split to avoid leakage.

The local [PrefixKV specification](baselines/prefixkv_spec.md) now documents
the paper equations, the upstream forget-ratio versus retained-ratio
conversion, exact source APIs, and reimplementation decisions. The unified
`prefixkv_reimpl` can load or generate an offline layer profile, enforces
calibration/evaluation disjointness for native profiles, maintains per-layer
ratios during decode, and uses the common cache/metric interface. Qwen2.5-VL
and other non-LLaVA runs are labeled `generalized_prefixkv_reimpl`. Official
parity is not an active execution path under the single-environment policy.

## VL-Cache

### Availability

The authoritative artifact is
[arXiv:2410.23317](https://arxiv.org/abs/2410.23317) (v1, submitted 2024-10-29;
published at ICLR 2025). The paper page does not link author code. Targeted web
and GitHub searches found surveys marking its code “N/A,” but no official
repository by the paper authors. Therefore implementation availability is
**unsupported** as of the audit date. Do not substitute a similarly named
repository.

### Models, evaluation, and implementation detail

The [paper](https://arxiv.org/html/2410.23317) reports:

- `llava-v1.6-mistral-7b` (GQA) and `llava-v1.6-34b` (MHA), both with
  `openai/clip-vit-large-patch14-336`;
- sampled COCO-Caption, DocVQA, and MathVista tasks through lmms-eval;
- an AWS P4 instance with 8× A100 40 GB GPUs;
- comparison to StreamingLLM, H2O, and PyramidKV with cache budget proportional
  to prompt length and a recent window fixed to a fraction of that budget.

The appendix's efficient implementation is not a drop-in library recipe. It
requires fused attention-statistics kernels because FlashAttention and
PagedAttention do not materialize attention scores in HBM. It describes two
statistics, online softmax, and a second recomputing kernel. The paper does not
state Python, torch, Transformers, FlashAttention, lmms-eval, CUDA, or Triton
versions. Those package versions are therefore **not specified**, not inferred.

The paper source is licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). That is a
paper/content license, not a software license. With no official software,
MosaicKV may implement `vl_cache_reimpl` from the algorithm description while
providing attribution and observing the paper license for copied/adapted
content; project counsel should decide any distribution questions. It must not
be called official VL-Cache code.

The local `vl_cache_reimpl` is now implemented in the common eager-HF cache
interface. Its equation-to-code map, ambiguity log, calibration leakage guard,
formula tests, structural sensitivity analysis, and current non-reproduction
status are documented in [the implementation specification](baselines/vl_cache_spec.md).
No official code parity is possible because no author implementation was found.

## Recommended baseline execution order

1. Implement simple in-tree full-cache, recent-only, uniform block, and random
   baselines under the common HF harness.
2. Validate `prefixkv_reimpl`, `lookm_reimpl`, and `vl_cache_reimpl` from a
   clean SHA with the common environment.
3. Use the implemented `prefixkv_reimpl`, `lookm_reimpl`, and
   `vl_cache_reimpl` only for identical-model comparisons. Never merge official and reimpl
   rows or compare them under unequal prompts, media, tokenization, generation,
   output length, cache budget, precision, or backend configuration.

LOOK-M/PrefixKV source and their local `*_reimpl` code were added after the
original audit. Official execution is disabled in the common environment;
LOOK-M remains unmeasured and prior diagnostics are not paper-eligible.
