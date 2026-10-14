# PrefixKV quality recovery: allocation-first development

Status: implemented CPU-tested options, **not a demonstrated answer-quality fix**.
No GPU run or full campaign is authorized by this document. Frozen historical
profiles, upstream code, and raw results are unchanged.

## Evidence that narrows the problem

The immutable `3b56d89` retry is at
`/scratch/djy8hg/runs/mosaickv/prefixkv-decode-policy-20261014-v2`.
Its `analysis/report.json`, per-task completion records, traces and
`upstream-parity-*.json` are authoritative. Eight examples per OCR task are
development evidence, not a representative paper evaluation.

- The native 50% and adapted 40 MiB plans matched the actual pinned upstream
  selector/evictor for 64 decode steps: identical tokens and zero maximum
  logit difference on the parity fixture.
- At 40 MiB, eviction and append-only variants produced identical answers
  and fixed-length token sequences on the development cohort. Initial selection
  is implicated; changing decode eviction alone did not recover quality.
- The upstream-shape 40 MiB variant scored 0.125 on TextVQA and 0 on DocVQA,
  versus Prompt Top-K's 0.5 and 0.1375. These metrics are not pooled.
- The separate native-50% diagnostic used roughly 152–154 MiB and scored
  0.5 / 0.16477. This is NOT a matched-budget win or a controlled explanation:
  both its profile shape and memory differed.

The paper studies a shared cumulative-priority threshold and offline layer
configuration. Its released description/MM-Vet evaluation is not identical to
40 MiB OCR QA. See the [paper](https://arxiv.org/html/2412.03409v2) and pinned
[`obtain_cdf_num`](../../../third_party/PrefixKV/prefixkv.py). We must not
turn a failed adaptation into a claim that official PrefixKV is broken.

## What the new options address

### 1. Protection-aware calibration, explicitly versioned

For sorted cumulative importance `CDF_l`, the old local search used
`max(boundary_count, prefix_length(CDF_l, threshold))`. The pinned upstream
search computes forgotten count as
`max(0, N - prefix_length(CDF_l, threshold) - boundary_count)`.
Its corresponding retained count includes an ADDITIONAL boundary allowance.
These differ particularly for layers whose attention is concentrated on BOS.

New `search_policy: upstream_protected_cdf` implements that count convention.
It still uses bounded search and exact integer apportionment, whereas upstream
accepts a global error below five positions and rescales/truncates forgotten
counts. Thus it is **not bitwise upstream profiler parity**. The regression
calls the actual pinned upstream function on a synthetic fixture and separately
checks the exact local sum. Defaults remain `legacy_cdf` for reproducibility;
old profiles/hashes are never silently changed.

### 2. Calibrate at the physical byte cap, not an averaged nominal ratio

The previous recalibration averaged `cap / source_bytes` and then searched
every prompt at that average retention. Prompts with different lengths thereby
had different calibration byte targets, although inference used one fixed cap.

`generate_prefixkv_byte_profile` instead uses

```
total_layer_positions = floor(byte_cap / bytes_per_layer_position)
sum_l retained_positions[sample,l] = total_layer_positions
```

for EVERY calibration sample, and freezes mean retained counts across samples.
This preserves the common threshold search and ordinary PrefixKV ranking; it
does not use answers, references, generated tokens, or evaluation scores.
The mean counts, byte cap, layer-byte width and search version are hash-bound
inside the profile. Runtime rejects a different cap or byte width. Short
calibration prompts that cannot exercise the target compression fail explicitly.
Calibration/evaluation identity overlap is rejected as before.

This option uses `prefixkv_byte_calibrated_reimpl`, not official PrefixKV.
Exact integer rounding and the protected decode interior remain local choices.

### 3. Capacity-floor ablation within the SAME budget

An offline profile can assign only a few positions to some layers. High
attention concentration does not prove that all other evidence is dispensable.
Test this hypothesis with the explicit opt-in
`prefixkv.layer_budget_floor_fraction: 0.5`:

```
floor_l = max(existing_protected_minimum,
              floor(0.5 * total_layer_positions / num_layers))
```

All floors are reserved before distributing remaining capacity using the
frozen layer shape. They are not extra bytes. In the 32-layer FP16 LLaVA
configuration, 40 MiB represents 2,560 layer positions, averaging 80 per layer;
the 0.5 option reserves 40 per layer. These are allocation arithmetic, not
measured answer-quality results. Infeasible floors fail closed; no padding or
FullKV fallback is used to manufacture success.

The result label is `prefixkv_capacity_floor_reimpl`. This is a deliberate
algorithmic ablation, **not paper-faithful upstream allocation**. Zero is the
default. Native `upstream_round`, append-only decode, and non-ragged transport
cannot silently activate it. Allocation traces now retain all layer counts,
minimum/maximum, total capacity, and count of short layers even in light traces.

### 4. Equal-layer diagnostic without changing token scoring

`profile_mode: fixed_global` + `decode_policy: effective_profile_ratio` is
explicitly labeled `prefixkv_uniform_layer_reimpl`. It keeps PrefixKV's
all-prompt attention ranking, sharing the same total bytes and decode policy.
If this recovers quality, allocation—not just scoring—is implicated. If all
allocation variants fail while Prompt Top-K passes, inspect the difference
between all-prompt accumulated importance and question-window importance.
Do not silently replace PrefixKV scoring with Top-K or claim such a replacement
as faithful PrefixKV.

## Small next validation, not a full sweep

`run_prefixkv_decode_preflight.allocation_configs` freezes seven arms:

1. FullKV, quality reference only, never a latency comparator;
2. Prompt Top-K;
3. old upstream-profile shape rescaled to 40 MiB (negative control);
4. exact-byte calibrated PrefixKV;
5. the same frozen profile with the preregistered 0.5 capacity floor;
6. equal-layer PrefixKV;
7. MosaicKV exact.

All six compressed arms share checkpoint/revision, processor, media, prompts,
FP16/eager attention, 64 deterministic output tokens, seed, block size one,
first/last mandatory policy, and EXACTLY 40 MiB prefill active storage. The
config helper checks these shared fields and profile identity. Subsequent
decode bytes differ by method and must be reported separately. Only the
PrefixKV variants change allocation. There is no claim of equal decode-time
storage with append-only controls.

Use ten newly selected, disjoint prompt-only calibration IDs and eight new
development evaluation IDs per task (TextVQA, DocVQA). Record the consumed
prior-debug/calibration ID union, exclusion check, ledger hashes and source SHA.
Freeze the profile and all arms BEFORE evaluating. Previously read development
labels may motivate these hypotheses but cannot become untouched validation.
Do not pick the winning arm after seeing evaluation answers and call it frozen.

First run retention-1 and actual-upstream differential checks for the new
adapted profile on one calibration prompt. Then evaluate every planned arm;
retain failed rows and full denominators. Report paired score differences and
intervals, answer/token agreement, actual initial/final KV bytes, and
informational latency. Zero warmups/one trial cannot support a latency claim.
Require the existing Prompt Top-K quality gate and exact-byte checks unchanged.
Any positive small result still requires a preregistered larger validation.

The factory does not submit jobs or change the old preflight CLI's arm list.
New wrappers must explicitly use it; running the old command unchanged would
repeat the old experiment. Use two one-A100 workers only after an authorized
immutable commit, under the existing global four-A100 controller. Do not release
the historical held PrefixKV full array or touch running HF/vLLM jobs.

`mosaickv/scripts/run_prefixkv_allocation_preflight.py` is the explicit runner
for this experiment. Its `--prepare` mode binds two immutable calibration and
evaluation ledgers, source SHA, and Slurm wrapper hashes. `--index 0` runs
TextVQA and `--index 1` runs DocVQA; `--analyze` produces the paired report
without any further inference. All modes require the same clean source and
plan hash. It excludes the first 32 parent IDs consumed by earlier diagnostics,
uses parent positions 32–41 for calibration and 42–49 for evaluation, and labels
the entire run retrospective development because the parent pool was used
historically. This is not an untouched held-out evaluation. No successful
development outcome automatically authorizes a full campaign.

## CPU profile generation interface

For reviewed, hash-recorded calibration attention JSONL and disjoint evaluation
IDs (paths below are input placeholders, not existing execution evidence):

```bash
source mosaickv/scripts/cache_env.sh
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/generate_prefixkv_profile.py \
  --attention-jsonl /scratch/djy8hg/NEW_RUN/calibration_scores.jsonl \
  --evaluation-sample-ids /scratch/djy8hg/NEW_RUN/evaluation_ids.json \
  --output /scratch/djy8hg/NEW_RUN/frozen_profile.json \
  --model llava-hf/llava-1.5-7b-hf \
  --model-revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --dataset DATASET_ID --dataset-revision DATASET_REVISION \
  --calibration-split SPLIT --seed 20271001 \
  --byte-cap 41943040 --bytes-per-layer-position 16384 \
  --start-size 1 --protect-size 1
```

The width is K+V across ALL KV heads for ONE layer-position, not a single head
or a full-stack token. Verify it from actual cache tensors. Profile generation
refuses an existing output; it does not overwrite prior artifacts.
