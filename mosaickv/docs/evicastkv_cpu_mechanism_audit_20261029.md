# EviCastKV: routing, baseline, and construction audit

EviCastKV is the paper name for MosaicKV. They are the same system. The
`mosaickv_exact` ID remains the historical implementation identity. The frozen
exact candidate and `evidence-output-guarded-v2` are different configurations
of that same system, not different research systems or independent baselines.
No historical raw prediction, configuration, source SHA, or result label is
rewritten by this audit. Prototype and repair variants remain unvalidated for
the performance claims at issue.

## Scope and evidence

This is a CPU-only retrospective mechanism audit, not a new quality experiment.
The original runs, negative gates, and consumed identities remain intact.
No model weights are loaded, no GPU job is submitted, no budget or numerical
threshold is changed, and no new evaluation identity is selected.

Inputs:

- PrefixKV study:
  `/scratch/djy8hg/runs/mosaickv/prefix-native-intermediate-20261029-v2-retry1`,
  producer `ad7c5c6a47c4f00776123a7e88e0e007da202249`.
- Guarded EviCastKV study:
  `/scratch/djy8hg/runs/evicastkv/fox-installed-guard-20261029-v2-retry1`,
  producer `c5749bbe08ffbea5f75c2465102142371fce89b8`.

`scripts/evicastkv_cpu_audit.py` binds the consumed input files by SHA-256,
projects only route/provenance columns from the guard observations, and writes
a separate write-once derived directory. Benchmark answers, references, and
sample scores are not used to construct a policy. Its manifest records the
audit producer separately from the original experiment producers.

## 1. Routing: what was and was not established

All 64 guarded requests (32 inputs at two caps) routed graph optimization to
layer 0, KV head 4. The Mistral model has 256 layer/KV-head routes, and the
frozen fraction is 1/256, so one route per request is intentional.

| Initial cap | Accepted routes | Rolled back | Installed changed positions | Requests with changed positions |
| --- | ---: | ---: | ---: | ---: |
| 20 MiB | 5 | 27 | 166 | 5 |
| 40 MiB | 2 | 30 | 175 | 2 |

There were no whole-cache FullKV fallbacks. These accepted changes did not
change any of the generated token sequences relative to the same-cap Top-K
anchor in the existing analysis. Successful execution and prompt-output
reconstruction do not establish a benchmark-quality improvement.

The inspected route path is:

1. `backends/hf_runtime.py::_prompt_token_attention_by_head`: mean the selected
   prompt window and GQA groups; concatenate layer arrays; reconstruct each
   `(layer, KV head)` score vector using recorded shapes and offsets.
2. `_attention_uncertainty_by_head`: normalize the vector and calculate entropy
   divided by log of its length.
3. `_uncertain_head_routes`: sort by descending uncertainty and then by the
   explicit `(layer, head)` tuple; keep the configured fraction.

The CPU regressions exercise both NumPy and torch paths, a known highest-entropy
head in a different layer, changing winners, exact layer/head coverage, stable
tie-breaking, and rejection of incomplete score maps. They do not reproduce a
hard-coded 0/4 route or a tested mapping/ranking defect.

**Important unresolved distinction:** the original light traces store selected
route identities but not the full per-route entropy arrays. Source inspection
and synthetic tests cannot prove that 0/4 was the correct empirical winner for
these model runs. It would be incorrect to label the observed collapse either
a proven indexing bug or proven genuine entropy behavior.

Before an algorithm change, a separately versioned, bounded consumed-input
diagnostic should record all route scores, candidate ordering, selected score
and rank, and any ties. Preserve the existing fraction and guard initially.
Do not launch fresh quality evaluation merely to recover missing telemetry.

## 2. PrefixKV: the precise boundary

The resolved configurations use:

- `llava-hf/llava-1.5-7b-hf`, revision
  `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, FP16 HF eager;
- upstream profile `prefixkv_llava-v1.5-7b_0.5.json`, reference retention 0.5;
- deterministic global apportionment to each physical-byte cap;
- first/last token protection, no extra question scoring;
- `ragged_llama_eager_v1`, `decode_policy: append_only`;
- fixed 1,024 emitted tokens, with 1,023 cache appends.

Traces identify this as **`prefixkv_selection_only_reimpl`**, with
`paper_native_algorithm: false`. In contrast, pinned upstream
`third_party/PrefixKV/prefixkv.py` at
`597f1ab032704951550f93bcc8a23f1454b80aa4` independently rounds layer forget
counts and can evict during decoding to maintain its ratios. Its legacy model
and stack are not interchangeable with the new HF-converted FP16 regime.

The new CPU differential fixture directly executes the pinned upstream class
on shared synthetic attentions and K/V. It checks full-query attention scores,
protected positions, selected positions, and layer lengths. This is algorithm
fixture evidence, **not new real-checkpoint parity**. Historical official
reproduction and parity artifacts remain valid in their original scope, not
automatically for the byte-rescaled append-only adaptation.

The 192 generation traces retain layer-capacity summaries:

- 80 MiB: minimum layer length 76, maximum 282;
- 120 MiB: minimum 113, maximum 424;
- 160 MiB: across inputs/tasks, minimum 150–174, maximum 531–566.

All audited decode eviction counts are zero. None of these layers is below the
25-position eviction-distance magnitude. Thus a short-layer eviction failure
does not explain these runs. Full selected-position lists are omitted from
these light traces, so their digests cannot reconstruct selection semantics.

The mechanism behind the low-budget ROUGE loss remains **unproven**. Do not
attribute it definitively to attention sinks, prompt loss, rounding, or layer
starvation from aggregate scores alone. PPL evaluates teacher-forced human
targets; reference ROUGE-L evaluates a free-running trajectory against FullKV.
Those different objectives can disagree without an evaluator bug. Neither
should be hidden or combined into one score. Further model-level differential
diagnosis must match checkpoint, precision, input, profile, and decode policy.

## 3. Compression overhead: measured, not guessed

The following means are derived from 20 registered timing repetitions on one
fixed input per task at **120 MiB**. They are technical repetition evidence,
not a population-latency result. Units are seconds.

| Component | EviCastKV: Description | EviCastKV: MM-Vet route | PrefixKV adaptation: Description | PrefixKV adaptation: MM-Vet route |
| --- | ---: | ---: | ---: | ---: |
| Cache-state construction | 2.496 | 2.438 | 2.498 | 2.446 |
| Forecast | 0.020 | 0.021 | 0.000 | 0.000 |
| Planner preparation | 2.982 | 2.911 | 0.000 | 0.000 |
| Graph | 1.600 | 1.625 | 0.000 | 0.000 |
| Utility and selection | 0.107 | 0.106 | 0.016 | 0.016 |
| Tier construction | 8.457 | 8.874 | 0.000 | 0.000 |
| Cache packing | 0.270 | 0.269 | 0.005 | 0.005 |
| Total compression | 15.932 | 16.245 | 2.519 | 2.466 |

Measured zeroes denote phases not executed by that method; they are not missing
measurements. The audit recomputes the non-overlapping compression sum for all
160 timed trials, including the registered 160 MiB PrefixKV arm. The maximum
absolute accounting difference is zero. It does not add nested planner
subphases twice. Trace instrumentation is separately recorded and excluded
from inference for every method.

Tier construction is the dominant measured EviCastKV phase. Its source boundary
includes input validation, source-node mapping, exact gathering,
`MosaicKVState.create`, and active-layout metadata, even though prototypes and
residuals are disabled. It is not a measurement of GPU tensor copying alone.
The related source functions are:

- `backends/hf_runtime.py::_exact_construction`;
- `prototypes/construction.py::construct_three_tier_cache`, `_exact_only`;
- `cache_state.py::FullKVState._gather_descriptors`, `MosaicKVState.create`.

The saved timings do **not** isolate each operation inside that boundary.
Source loops suggest testable hypotheses, not a measured allocation of the
8.5-second cost. Do not claim that another routing fraction or faster selector
will remove this overhead: selection is already about 0.106 seconds.

## Engineering order and stopping conditions

1. Preserve both negative development gates and all consumed identities. No
   missing-data inference retry or large campaign is justified.
2. Close routing telemetry on consumed inputs before revising routing. Keep
   budgets, compatibility controls, and guard thresholds unchanged.
3. Keep the PrefixKV adaptation separate from official native reproduction.
   Audit full-query selection and transfer assumptions without tuning on the
   inspected benchmark labels.
4. Internally profile the tier-construction boundary. An execution-only fast
   path must preserve selected source positions, original K/V, logical decode
   positions, mandatory tokens, padding, and initial/final/peak storage. Apply
   equivalent optimizations to compatible baselines rather than constructing
   an artificially slow competitor.
5. Only a meaningful, safe candidate change would justify a separately frozen
   disjoint development protocol, followed by powered independent validation.
   More samples cannot create superiority over Top-K when predictions are
   identical. No quality or statistical threshold is weakened by this audit.

Reproduce the derived audit without GPU inference:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_cpu_audit.py \
  --output /scratch/djy8hg/runs/evicastkv/NEW_WRITE_ONCE_CPU_AUDIT
```

The output contains `audit.json`, per-trial and aggregate compression Parquet,
PrefixKV layer-capacity Parquet, `report.md`, and an input/output hash manifest.
