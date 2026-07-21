# Unified Hugging Face runtime

## Scope and safety status

`mosaickv.backends.HuggingFaceMosaicKVModel` connects one explicit HF prefill to forecasting,
graph construction, block utility, lazy-greedy selection, tier construction, optional repair,
and token-by-token decoding. The MosaicKV path never calls `model.generate()`. Only eager
attention is enabled; SDPA and FlashAttention-2 remain blocked until separate correctness gates
pass.

The audited LLaVA-1.5, Qwen2.5-VL, LLaVA-OneVision, and optional InternVL adapters expose
post-RoPE cached keys. Their capability records therefore set `supports_prototype_merge=false`
and `supports_residual_repair=false`. For these adapters:

- `mosaickv_exact` runs forecasting, graph construction, utility, and exact block selection;
- `mosaickv_proto` is recorded as `mosaickv_proto__mosaickv_exact_safety_fallback`;
- `mosaickv_full` is recorded as `mosaickv_full__mosaickv_exact_safety_fallback`.

The fallback is intentional and visible in every trace. It does not average post-RoPE phases,
create fake residuals, or claim that repair ran. Backend-independent synthetic tests exercise
the real prototype and repair components with explicitly RoPE-free capability metadata.

## Cache representation

Exact selected K/V entries retain their original post-RoPE values. Each entry is ordered by its
original logical position. HF cache tensors require a uniform physical sequence dimension, so
shorter layer/head selections are zero-padded and accompanied by a checked boolean validity map.
An eager-attention pre-hook converts that map to a per-query-head additive mask. The runtime keeps
three lengths distinct:

- packed active cache length;
- original/logical sequence length;
- next decode position.

The next decode position is always the original logical position, not the packed index. Generated
tokens are appended after the packed prompt, and each layer mask gains one valid slot per step.
NaN/Inf checks cover prefill logits, decode logits, packed K/V tensors, and additive masks.

Retention ratio `1.0` selects every source block and bypasses prototypes, residual conversion,
and repair. Because the selected set is uniquely determined, this path uses a deterministic
mandatory-first all-exact selector instead of forcing a partial-budget lazy-greedy heap to its
full cardinality. It still records each block's exact sequential marginal gain and selection
reason. The gate compares its greedy token IDs and per-step logits with the untouched FullKV loop.
The default numerical tolerance remains the one registered in `REPRODUCIBILITY.md`; a checkpoint
is not eligible for scientific runs until the full parity script records both token agreement and
the maximum logit difference.

## Forecast and attention memory

Hybrid and draft forecasting clone the original prefill cache and never run a second prefill.
The clone is discarded after deterministic draft decoding. Eager prompt attention is captured
only for the configured trailing prompt window, and the slice is cloned so it does not retain the
full quadratic attention backing storage. Draft attention has one query position per step.
Utility attention provenance is the actual eager attention probability, so no pre-RoPE query is
incorrectly dotted against a post-RoPE cached key.

## Configuration and commands

The resolved run schema accepts JSON, TOML, YAML, and YML. The versioned example is
`configs/hf_mosaickv.yaml`:

```bash
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/djy8hg/cache/mosaickv/huggingface
export HF_DATASETS_CACHE=/scratch/djy8hg/cache/mosaickv/datasets

python -m mosaickv.cli evaluate \
  --config configs/hf_mosaickv.yaml \
  --task synthetic_smoke \
  --subset-size 1 \
  --output-dir /scratch/djy8hg/runs/mosaickv/qwen-smoke
```

Equivalent direct flags follow the requested interface:

```bash
python -m mosaickv.cli evaluate \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --backend hf \
  --attention-backend eager \
  --method mosaickv_full \
  --task synthetic_smoke \
  --retention-ratio 0.5 \
  --block-size 16 \
  --forecast hybrid \
  --draft-tokens 4 \
  --repair-policy entropy_or_prototype_risk
```

If `--model-revision` is omitted for a registered model, the CLI uses the immutable revision in
the audited model capability matrix. Public benchmark tasks additionally require
`--dataset-revision`. `HF_TOKEN` is read only by the Hugging Face libraries from the environment;
MosaicKV never serializes it. The CLI refuses to load weights unless `HF_HOME` or
`HUGGINGFACE_HUB_CACHE` points outside the user's home directory.

The published-baseline path also accepts `prefixkv_reimpl`. Use
`configs/hf_prefixkv_reimpl.yaml` for the equal-layer `fixed_global` control,
or switch `prefixkv.profile_mode` to `offline_profile` and provide a native,
leakage-checked profile. PrefixKV requires eager attention and token-sized
blocks. It selects the same source positions in every KV head, then maintains
the resolved layer ratios with fixed-distance eviction during decoding.
LLaVA-1.5 traces say `prefixkv_reimpl`; all other model families say
`generalized_prefixkv_reimpl`. See the
[algorithm specification](baselines/prefixkv_spec.md) before running a
comparison, particularly its retained-versus-forgotten ratio warning.

The runtime also accepts the ICLR method label `vl_cache_reimpl`; it is not the
later recurring-image VLCache system and is not official author code. Use
`configs/hf_vl_cache_reimpl.yaml`. The eager prefill attention is restricted to
post-vision language queries, relative-threshold sparsity determines
prompt-specific layer budgets, and accumulated post-vision attention selects
unchanged K/V positions. Optional ambiguity calibration sample IDs are checked
against every evaluation request before prefill. See the
[VL-Cache specification](baselines/vl_cache_spec.md) for exact equations,
GQA/integer-budget interpretations, and the current non-reproduction status.

Each run produces append-only JSONL results, a Parquet aggregate, an immutable provenance
manifest, and one JSON trace per attempted sample. A trace contains selected-block decisions,
prototype records, sparse graph edges, forecast statistics, repair events, packed slots, generated
token IDs, active/residual bytes, and a timing breakdown. Failed samples receive both a failed
result row and a failed trace.

## Validation progression

Run the non-downloading tiny-architecture gate locally in the dedicated HF environment:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src:. \
  /scratch/djy8hg/env/mosaickv/bin/python \
  scripts/verify_hf_runtime.py --max-new-tokens 16
```

On Slurm, `slurm/hf_runtime_progression.sbatch` runs the gates in order. Real checkpoint downloads
and the 20-example dataset run are explicit opt-ins:

```bash
MOSAICKV_HF_PYTHON=/scratch/djy8hg/env/mosaickv/bin/python \
sbatch slurm/hf_runtime_progression.sbatch

MOSAICKV_HF_PYTHON=/scratch/djy8hg/env/mosaickv/bin/python \
MOSAICKV_ALLOW_MODEL_DOWNLOAD=1 \
MOSAICKV_RUN_QWEN3B=1 \
MOSAICKV_RUN_LLAVA15=1 \
sbatch slurm/hf_runtime_progression.sbatch

MOSAICKV_HF_PYTHON=/scratch/djy8hg/env/mosaickv/bin/python \
MOSAICKV_ALLOW_MODEL_DOWNLOAD=1 \
MOSAICKV_RUN_DEV20=1 \
MOSAICKV_DEV_SUBSET_SIZE=20 \
MOSAICKV_MMSTAR_REVISION=bc98d668301da7b14f648724866e57302778ab27 \
sbatch slurm/hf_runtime_progression.sbatch
```

Later gates must not be reported as passed when an earlier job fails. The tiny suite is synthetic
validation, not a paper result.

## Verified integration gates

The following July 19, 2026 A100-SXM4-80GB runs were inspected after completion. They validate
the runtime plumbing at the dirty development worktree recorded by each manifest; they are not
canonical paper results and must not be copied into a measured-results table.

- Slurm job `17114353` passed the no-download tiny LLaVA-1.5, Qwen2.5-VL, and
  LLaVA-OneVision architecture gate for all three method labels, retention monotonicity, trace
  completeness, and 16-token retention-1 parity.
- Slurm job `17114476` completed the pinned Qwen2.5-VL-3B one-image/one-prompt gate. Its raw row,
  Parquet aggregate, trace, and manifest are under
  `/scratch/djy8hg/runs/mosaickv/hf-runtime/qwen3b/655bcb0ce8014ecb85df4443a4b66085`.
- Slurm job `17114491` completed the pinned LLaVA-1.5-7B one-image/one-prompt gate. Its artifacts
  are under
  `/scratch/djy8hg/runs/mosaickv/hf-runtime/llava15/d360de0537f04165b07dc6b2851d7a53`.
- Slurm job `17114628` completed 20 deterministic MMStar examples at revision
  `bc98d668301da7b14f648724866e57302778ab27`: 20 unique completed raw rows, 20 Parquet rows,
  and 20 matching complete traces. Its artifacts are under
  `/scratch/djy8hg/runs/mosaickv/hf-runtime/mmstar-dev20/f65b7cb4d37146d48eabe20385c37ee8`.
- Slurm job `17115048` passed the post-integration Qwen2.5-VL-3B retention-1 gate against
  untouched FullKV for 16 greedy tokens, with token agreement `1.0` and maximum logit difference
  `0.0`. The validation record is
  `/scratch/djy8hg/runs/mosaickv/hf-runtime/qwen3b-retention-one-17115048.json`.
- Slurm job `17115049` passed the corresponding LLaVA-1.5-7B retention-1 gate with the same
  16-token agreement and maximum logit difference. The validation record is
  `/scratch/djy8hg/runs/mosaickv/hf-runtime/llava15-retention-one-17115049.json`.

The dedicated environment used Python 3.11, PyTorch 2.11.0+cu130, Transformers 4.57.6,
Accelerate 1.13.0, Datasets 4.8.4, and lmms-eval 0.7.2. `pip check` passed. FlashAttention-2 is
not installed because its source build did not pass on the login node; no FlashAttention support
is claimed. All gates above used eager attention. Qwen2.5-VL and LLaVA-1.5 expose post-RoPE
cached keys, so their `mosaickv_full` traces explicitly record the exact-selection safety fallback,
with no prototypes or residual promotion.

The lmms-eval route constrains task discovery to the requested task family, delegates MMStar
answer processing and aggregation to the installed official functions, and pins the dataset load
to the manifest revision. A packaged copy of lmms-eval v0.7.2's MIT-licensed MMStar base template
repairs an upstream wheel packaging omission without modifying site-packages. The compatibility
redirect is limited to that known missing template.
