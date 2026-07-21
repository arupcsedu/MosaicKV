# PrefixKV baseline specification

Status: `prefixkv_reimpl` is a local, paper-faithful reimplementation. It is
never official PrefixKV code. The official source is pinned as a read-only
submodule at commit `597f1ab032704951550f93bcc8a23f1454b80aa4` under
`third_party/PrefixKV`; its `LICENSE` is MIT, copyright 2024 Zuyan Liu.

Primary sources are the [NeurIPS 2025 paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/93672c60c11d74434f1918aef8a36ab1-Paper-Conference.pdf),
the pinned [`prefixkv.py`](../../../third_party/PrefixKV/prefixkv.py), the
patched [`LlamaAttention.forward`](../../../third_party/PrefixKV/patch_attention_forward.py),
and the official [PPL](../../../third_party/PrefixKV/eval_ppl.py) and
[ROUGE](../../../third_party/PrefixKV/eval_rouge.py) drivers.

## Algorithm

For layer `l`, the official implementation averages prompt self-attention over
query heads, then sums over prompt query positions. This gives one nonnegative
importance value per source position, shared by all KV heads. Sorting those
values in descending order defines the layer's priority sequence. The
cumulative normalized importance of its first `oN` values is `P_l^o`.

The paper defines a retained-cache budget `r` and a common information
threshold `p`. Each layer keeps the shortest priority prefix whose cumulative
importance reaches `p`:

```text
R_l = min { o : P_l^o >= p }
sum_l R_l = r L
```

Algorithm 1 binary-searches `p`. If the discrete layer sizes cannot hit the
budget exactly, the paper says to scale the configuration. The official
`obtain_cdf_num` implementation performs that search in terms of positions to
forget. `prefixkv_reimpl` uses the paper threshold, then performs deterministic
bounded integer apportionment. This preserves the adaptive layer shape while
making the per-layer sizes sum to the exact representable target.

At prefill, each layer retains its protected first `start_size` positions, its
last `protect_size` positions, and the highest-importance remaining positions.
Selection is shared across KV heads, as in the official gather operation. K and
V are copied unchanged; PrefixKV does not merge, quantize, repair, or offload.

During decoding, the official code maintains the layer ratios by checking
`int(active_length - logical_length * R_l)` after each forward. If positive, it
deletes at most one position at a fixed compact-cache offset. Its default
distance is `-25`, meaning 25 positions before the compact cache tail; very
short caches fall back to their midpoint. The unified runtime implements the
same post-forward, pre-next-step schedule and records every layer/step decision.

## Ratio and cache-budget conventions

The paper's `r` is a **retained** ratio. The official Python object's `ratio`
and bundled JSON filenames are a **forgotten/evicted** ratio. For example, a
MosaicKV retention ratio of `0.2` maps to official `--ratio 0.8` and to
`prefixkv_llava-v1.5-7b_0.8.json`. The runtime records both values on every
trace:

```text
official_forget_ratio = 1 - mosaickv_retention_ratio
```

The paper budget counts token positions equally across layers; each selected
position is present in every KV head and in both K and V. The common runtime
therefore requires `block_size: 1`. Under `budget_unit: blocks`, the target is
aligned to a whole layer-shared position. Under `budget_unit: bytes`, the
selector decrements non-protected positions until exact selected K/V payload
bytes do not exceed the hard byte limit. Trace fields distinguish logical
retained bytes from any padding used by the shared HF packed-cache adapter;
systems comparisons must report the actual tensor-storage field.

## Offline profile and search process

The paper estimates one global prefix configuration from ten random training
samples. The official `--profile` path writes one list of per-layer forget
ratios per sample to JSONL. The pinned repository does not contain the script
that aggregates those rows into its bundled `confs/*.json` files.

MosaicKV's native profile is explicit and immutable. It records:

- exact model ID and revision;
- calibration dataset ID, revision, split, seed, and sorted sample IDs;
- the sample-ID digest and profile digest;
- target retained ratio and the corresponding official forget ratio;
- protected boundary settings;
- mean per-layer forget ratios across the per-sample binary-search results.

The generator refuses overlap between calibration and evaluation IDs. The HF
runtime checks each evaluation sample against the profile again. Official raw
list profiles can be loaded for parity, but because the upstream files omit
calibration sample IDs, their disjointness is `unverifiable`; they must not be
used for a paper result unless that missing provenance is recovered.

Capture rows use this JSONL schema:

```json
{"sample_id":"cal-0001","layer_scores":[[0.1,0.2],[0.4,0.3]]}
```

Generate a native profile with:

```bash
python mosaickv/scripts/generate_prefixkv_profile.py \
  --attention-jsonl /scratch/$USER/mosaickv/prefixkv/calibration_scores.jsonl \
  --evaluation-sample-ids /scratch/$USER/mosaickv/prefixkv/evaluation_ids.json \
  --output /scratch/$USER/mosaickv/prefixkv/profile-r050.json \
  --model llava-hf/llava-1.5-7b-hf \
  --model-revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --dataset DATASET_ID --dataset-revision DATASET_SHA \
  --calibration-split 'train[:10]' --retention-ratio 0.5 --seed 0
```

`prefixkv_attention_scores()` produces the required rows directly from eager
prefill attention tensors; the script deliberately does not reload model
weights or select data implicitly.

## Inference interface

Use the adaptive paper path by setting:

```yaml
method: prefixkv_reimpl
cache:
  retention_ratio: 0.5
  block_size: 1
prefixkv:
  enabled: true
  profile_mode: offline_profile
  profile_path: /scratch/USER/mosaickv/prefixkv/profile-r050.json
  start_size: 1
  protect_size: 1
  eviction_distance: -25
```

`profile_mode: fixed_global` keeps the same target fraction in every layer and
still ranks positions with PrefixKV attention. It is an explicit equal-layer
control, not the adaptive paper result.

The implementation enters the same explicit HF prefill/decode loop, cache
packer, timing code, result schema, and manifest writer as MosaicKV. It emits
per-layer sizes, selected physical/logical positions, attention-score digests,
retained bytes, fixed offsets, and decode evictions in the debug trace.

## Supported model assumptions and labels

The public repository directly implements legacy LLaVA-1.5-7B and 13B with a
vendored LLaVA tree, Transformers 4.31.0, legacy tuple caches with sequence axis
2, 32 or 40 Llama layers, and a monkey-patched eager Llama attention forward.
Its README points to the `Zuyan/ElasticCache` LLaVA checkpoint. Although the
paper includes additional model studies, the released runtime is not a generic
Transformers integration.

Results on LLaVA-1.5 use the label `prefixkv_reimpl`. Results on Qwen2.5-VL,
LLaVA-OneVision, InternVL, or any other family are always labeled
`generalized_prefixkv_reimpl`. The parity comparator refuses to call a
generalized artifact official PrefixKV.

## Documented reimplementation decisions

- Equal-score ties use physical position as a stable secondary key; upstream
  `torch.argsort` does not specify this parity rule.
- Integer apportionment hits the exact representable global target; upstream
  independently rounds per-layer profile ratios and can miss it slightly.
- Calibration/evaluation separation is enforced. The upstream driver shuffles
  a caller-provided prefix of one data file and does not enforce a disjoint set.
- The shared HF runtime may physically pad layers to support distinct lengths
  while using stock model classes. Logical retained bytes and actual packed
  tensor bytes are both recorded; only the latter is a systems measurement.
- HF-converted `llava-hf/llava-1.5-7b-hf` is not interchangeable with the
  legacy checkpoint required by the official repository. A cross-check using
  different checkpoints or library stacks is `not_comparable`, not parity.

## Parity protocol

Official and reimplementation artifacts are compared only when model and
tokenizer revisions, calibration and evaluation sample digests, prompt/media
payloads, profile, cache budget, generation settings, precision, attention
implementation, environment, hardware, timing protocol, and seed match. The
comparator reports per-layer cache sizes, retained bytes, PPL, ROUGE-L F1,
answers/token agreement, and latency. Run it with:

```bash
python mosaickv/scripts/compare_prefixkv_parity.py \
  --official /scratch/$USER/mosaickv/prefixkv/official.json \
  --reimplementation /scratch/$USER/mosaickv/prefixkv/reimpl.json \
  --output /scratch/$USER/mosaickv/prefixkv/parity.json
```

See [prefixkv_parity_report.md](prefixkv_parity_report.md) for current execution
status. The one-sample legacy LLaVA diagnostic now has two measured artifacts
and returns `status: comparable`; it also empirically confirms the documented
upstream integer-rounding undershoot. Because that run used a dirty worktree
and one timing trial, it remains ineligible for a paper table. Reproduce it
with:

```bash
sbatch mosaickv/slurm/prefixkv_official_parity.sbatch
```

The job uses `run_prefixkv_llava_parity.py` to generate a profile from a
recorded calibration sample, reject evaluation overlap, and run the pinned
official class and `prefixkv_reimpl` against the same legacy checkpoint and
model instance. No patch is applied to the official submodule.
