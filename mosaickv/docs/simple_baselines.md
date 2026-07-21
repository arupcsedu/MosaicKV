# Simple exact-cache baselines

The Hugging Face runtime exposes five transparent reference policies through the same explicit
prefill, cache packing, token-by-token decode, trace, and measurement path used by MosaicKV.
These policies are local baselines, not implementations of published systems.

## Methods

| Method | Policy | Stochastic input |
|---|---|---|
| `full_kv` | Keep the untouched FullKV cache. No packing or compression stage runs. | None |
| `random_kv` | Keep mandatory blocks, then traverse a Python MT19937 shuffle of all optional blocks. | Recorded execution seed |
| `uniform_kv` | Keep mandatory blocks, then repeatedly allocate to the layer/KV-head/modality stratum with the smallest retained-cost fraction. | None |
| `prompt_attention_topk` | Rank optional blocks by eager-attention mass from the last configured prompt-window queries. | None |
| `value_topk` | Rank optional blocks by nearest-neighbor cosine novelty of pooled values within the same layer and KV head. | None |

All ties are resolved by canonical source node ID. `uniform_kv` uses a deterministic
farthest-position traversal within each stratum. When indivisible blocks or mandatory blocks make
exactly equal stratum fractions impossible, the allocator selects the least-retained feasible
stratum at each step; the hard global budget always takes precedence.

`fullkv` remains accepted as a legacy spelling for existing FullKV configurations. New baseline
runs should use `full_kv` so manifests and paper-facing method names are unambiguous.

## Shared cache contract

- A single `FullKVState` performs blockization for all compressed policies. Blocks never cross a
  layer or KV head, and every policy uses the configured `cache.block_size`.
- The terminal prompt block is mandatory for each layer/head in the unified HF runtime. Every
  selector admits mandatory blocks first and fails if they cannot fit.
- The effective budget is the smaller of the explicit upper bound and the ceiling of source cost
  times `retention_ratio`. It can be counted in blocks, retained token slots, or bytes.
- Each selected block remains exact. Prototype and residual tiers are empty, and repair is
  disabled. Strict configuration validation rejects a simple baseline if any MosaicKV stage is
  enabled.
- Selection records include every block's score, cost, stratum, rank, decision reason, seed, and
  score provenance. `selected_source_bytes` is the exact sum of selected K/V tensor storage.
- The common generation metric `active_kv_bytes` measures the live packed cache after decoding,
  including generated-token cache entries. This intentionally uses the same storage-accounting
  path as MosaicKV and FullKV.
- At retention ratio `1.0`, compressed baselines select every source block, create no other tiers,
  reconstruct the source state exactly, and pass through the common retention-parity checker.

`prompt_attention_topk` is available only with eager attention because it requires returned
attention tensors. It uses prompt queries only: no future-query forecast or draft rollout is
constructed. `value_topk` computes chunked similarities and does not materialize a global
all-block similarity matrix.

## Configuration and commands

[`configs/hf_simple_baseline.yaml`](../configs/hf_simple_baseline.yaml) is a pinned random-baseline
example. Change only `method` to select `uniform_kv`, `prompt_attention_topk`, or `value_topk`;
keep forecasting, graph, selection, prototypes, residual, and repair disabled.

```bash
mosaickv evaluate \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --dataset-revision schema-v1 \
  --backend hf \
  --attention-backend eager \
  --method random_kv \
  --task synthetic_smoke \
  --retention-ratio 0.5 \
  --block-size 16 \
  --seed 0
```

For a full-cache comparison, use `--method full_kv --retention-ratio 1.0`. Direct CLI
construction disables inapplicable MosaicKV stages automatically. YAML configurations must state
those disabled stages explicitly so the serialized experiment configuration remains auditable.

## Validation boundary

CPU synthetic-cache tests cover common budget resolution, exact byte accounting, mandatory
retention, uniform per-stratum allocation, seeded random determinism, score isolation, byte-budget
feasibility, empty prototype/residual tiers, and lossless retention-1 reconstruction. Tiny random
LLaVA architecture tests exercise the shared HF decoding and trace path without downloading model
weights. These are validation checks, not measured research results.
