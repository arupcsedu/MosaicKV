# V8.1: frozen-proposal output replay (consumed-input diagnostic only)

V8's 32 proposals on two already consumed inputs all failed the unchanged
actual-logit guard, despite improving the recorded draft-attention surrogate.
Its clone/repack controls and source-state isolation passed. Those artifacts do
not establish why the surrogate failed: they did not save actual compressed
attention outputs or projected residual changes. V8.1 gathers this missing
evidence before any proposal redesign.

## Frozen scope

The versioned protocol binds the V8 plan, both diagnostics, raw logits, and
draft-evidence tensors by SHA-256. It copies only the same two label-redacted
inputs, original configurations, and exact proposal descriptors. No new
selection, ordering, search, answer access, or sample materialization occurs.
Each proposal runs only through its originally observed stopping prefix;
the FullKV/anchor and clone/repack controls retain their original 32 prefixes.
Partial rejected trajectories are never called 32-token acceptance passes.

All original controls remain unchanged: pinned LLaVA-1.5 revision, FP16,
HF eager, original prefill, 40 MiB lifetime live-KV boundary, generated-history
reserve, exact K/V, mandatory membership, and logical positions. Guard horizon
32, checkpoints 1/4/16/32, absolute tolerance 1e-7, and relative tolerance 1e-6
are unchanged. FP64 KL remains shadow evidence, not a replacement acceptance
rule. Fresh cohorts, calibration, and paper campaigns remain disabled.

## What is observed

The recorder supports only the inspected Transformers 4.55.2 eager
`transformers.models.llama.modeling_llama.LlamaAttention` and
`LlamaDecoderLayer`. It hashes that installed source file in the diagnostic.
At every executed single-token prefix, all decoder layers record:

| Field | Read-only hook | Meaning |
| --- | --- | --- |
| `attention_heads` | `self_attn.o_proj` forward input | Actual attention output, preserving each query head before concatenation/projection |
| `projected_attention` | `self_attn.o_proj` forward output | Actual attention contribution after the output projection |
| `residual_input` | decoder forward input | Incoming residual stream |
| `normalized_attention_input` | `input_layernorm` forward output | Actual normalized input used to compute attention queries |
| `post_attention_residual` | `post_attention_layernorm` forward input | Observed sum of incoming residual and projected attention |
| `layer_output` | decoder forward output | Residual stream after the MLP contribution |

Hooks detach/copy outputs without returning replacements or changing weights.
They are removed in `finally`, including on exceptions. Every FullKV, anchor,
control, and proposal logit tensor must replay V8 **bit-for-bit**. Both null
controls must additionally reproduce every intermediate tensor bit-for-bit.
Any mismatch fails instrumentation; it is not algorithmic improvement or loss.
Unsupported versions/architectures/attention implementations fail explicitly.

## Sideband predictions and analysis

The original fixed-FullKV-query surrogate is reconstructed separately from
saved probabilities, original source values, retained generated-history terms,
and exact selected memberships. Its per-prefix anchor/candidate head vectors
and the relevant output-projection columns are saved. CPU FP64 matrix products
show their predicted projected change; they never enter model execution or
the registered guard. These remain predictions, not measured compressed
attention outputs. Small differences from the GPU surrogate's accumulation
order are not evidence of a changed algorithm.

The CPU analyzer validates artifact hashes and clean source/config manifests,
then writes per-prefix and per-layer vector errors against FullKV, changes
relative to the anchor, and directional agreement between predicted and
measured changes. It distinguishes:

1. fixed-query prediction versus actual compressed head output;
2. head-output error versus its projection-weighted error;
3. projected attention error versus residual-stream and downstream error;
4. these intermediate observations versus actual KL/logit/token guard failure.

Counts concern observed rejection prefixes, not independent statistical units.
Associations can locate a mismatch but do not prove a causal explanation.
There are no benchmark scores, quality guarantees, memory savings over the
unchanged anchor, or production-latency claims. Hook copies add diagnostic
overhead; captured tensors and private probe KV are reported separately from
the unchanged 40 MiB live-cache boundary.

## Execution

Use the common environment and scratch-only cache setup. Canonical preparation
requires a clean immutable source. The existing admission controller permits
at most six aggregate A100s, but this replay has only two one-A100 workers,
each with 12 CPUs and a two-hour limit under `bi_fox_dgx`.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
python=/scratch/djy8hg/env/mosaickv/bin/python
"$python" mosaickv/scripts/head_substitution_v7.py submit \
  --protocol mosaickv/configs/experiments/bounded_prefix_output_v8_1.yaml \
  --root /scratch/djy8hg/runs/mosaickv/bounded-prefix-output-v8-1-20261026-v1
# After both workers complete; use a new derived directory, never overwrite:
"$python" mosaickv/scripts/analyze_prefix_output_replay.py \
  --root /scratch/djy8hg/runs/mosaickv/bounded-prefix-output-v8-1-20261026-v1 \
  --output /scratch/djy8hg/runs/mosaickv/bounded-prefix-output-v8-1-20261026-v1/derived-analysis-v1
```
