# Unified Hugging Face runtime

## Scope and safety status

`mosaickv.backends.HuggingFaceMosaicKVModel` connects one explicit HF prefill to forecasting,
graph construction, block utility, lazy-greedy selection, tier construction, optional repair,
and token-by-token decoding. The MosaicKV path never calls `model.generate()`.
FullKV supports eager, SDPA, and FlashAttention-2 on adapters that declare the
implementation. Compression prefill remains eager so measured prompt attention
is explicit. A separately configured `decode_attention_implementation=sdpa`
may switch an all-valid exact cache to SDPA after selection; padded per-head
caches and repair fail closed. Both implementations are recorded in the config,
manifest, and debug trace, so this is never silently labeled HF eager.

All audited adapters expose post-RoPE cached keys. Transformers 4.55.2 eager
execution supplies an audited M-RoPE inverse/anchor transform for Qwen2.5-VL
and standard Llama/Qwen2 transforms for LLaVA-1.5 and LLaVA-OneVision. These
three families declare prototype plus residual-repair support. InternVL,
version/position mismatches, and fused prototype paths remain exact-only
fallbacks. Consequently:

- `mosaickv_exact` runs forecasting, graph construction, utility, and exact block selection;
- supported native-adapter `mosaickv_proto` constructs exact and phase-safe prototype tiers;
- supported native-adapter `mosaickv_full` additionally constructs pinned residuals and evaluates
  the configured repair policy; and
- unsupported paths record an explicit `mosaickv_exact_safety_fallback` label.

Every trace records the safety assessment and transform provenance. Fallbacks
do not average post-RoPE phases, create fake residuals, or claim repair ran.

## Cache representation

Exact selected K/V entries retain their original post-RoPE values. Each entry is ordered by its
original logical position. For fixed-block exact selection, the byte target is quantized to a
representable block-subset capacity and lazy greedy fills that capacity independently in every
layer/head. This produces a dense, all-valid cache and avoids per-token mask construction.
Methods whose algorithm requires unequal layer/head lengths are zero-padded and accompanied by a
checked boolean validity map; an eager-attention pre-hook converts that map to a per-query-head
additive mask. The runtime keeps three lengths distinct:

- packed active cache length;
- original/logical sequence length;
- next decode position.

The next decode position is always the original logical position, not the packed index. Generated
tokens are appended after the packed prompt, and each layer mask gains one valid slot per step.
NaN/Inf checks cover prefill logits, decode logits, packed K/V tensors, and additive masks.
The explicit loop grows attention masks and cache-position ranges in amortized device buffers,
does not clone the live cache on ordinary decode steps, and reduces all per-token finite-logit
flags once per sequence. A full cache clone remains mandatory only for a provisional repair step
that may need rollback. Each trace records these policies and whether the packed cache was
all-valid. For append-only no-repair paths, the runtime also preallocates the
configured decode horizon and exposes occupied-prefix cache views, while
logical positions continue to advance from the original prompt length.
The ordinary no-hook path also avoids constructing an empty capture stack and
advances the documented mutable `DecodeState` in place rather than allocating
a state and dictionary copy at every token.

Exact-only construction groups descriptors by layer/head and performs one
vectorized gather per layer, instead of one device gather per selected token
followed by thousands of small concatenations. Dense layer/head capacities are
packed without a model-side mask; the bookkeeping mask remains available for
byte and invariant checks. Controlled dirty-tree A100 job `17346904_2`
completed five warmups and 20 trials at each bounded LLaVA-1.5 point. It
confirmed that cache construction no longer projects beyond the time limit;
for the short output-64 prompt, compression overhead outweighed decode savings.
These exploratory measurements remain ineligible for paper tables until rerun
from a clean commit.

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
result row and a failed trace. Non-diagnostic catalog records are hashed directly
without first materializing full edge dictionaries. Trace construction and
canonical hashing are excluded from inference latency for every method and are
reported separately as `timing_breakdown.trace_instrumentation`; trace file I/O
also occurs outside the inference boundary.

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

## Current integration gates

The common environment uses Python 3.11, PyTorch 2.8.0/CUDA 12.8, and
Transformers 4.55.2. Exploratory A100 environment job `17347802` passed all
251 exact pins, imports, CUDA matrix multiplication, and the specialized
FlashAttention dense/varlen matrix. The synthetic 100%-retention smoke reports
exact equivalence and zero maximum absolute error. The repository build is an
explicitly patched inference-only SM80 multimodal wheel for the audited
64/80/128 head dimensions; it does not claim general FlashAttention or
training support. Qwen2.5-VL-3B SDPA job `17346718` and FlashAttention job
`17348171` each passed 16-token ordinary/reinjection/retention-1 parity with
maximum logit difference zero.

No pretrained-checkpoint run is canonical while the current worktree is dirty,
and no 20-example dataset progression has been accepted under this common
lock. The bounded v17 Qwen2.5-VL-3B component run completed 44 three-tier cells
and four exact no-prototype controls with a passed active-byte/provenance gate.
Its 84 rows are implementation evidence only because the worktree was dirty
and each task had one or two examples. Subsequent exploratory A100 gates
`17855955` (LLaVA-1.5 FP16) and `17856497` (OneVision-7B BF16) each passed
16-token FullKV/reinjection parity, retention-one identity, a real 50%
three-tier cache, and forced residual restoration with at most one re-decode.
They validate implementation support but remain non-canonical dirty-tree
evidence.

The lmms-eval route constrains task discovery to the requested task family, delegates MMStar
answer processing and aggregation to the installed official functions, and pins the dataset load
to the manifest revision. A packaged copy of lmms-eval v0.7.2's MIT-licensed MMStar base template
repairs an upstream wheel packaging omission without modifying site-packages. The compatibility
redirect is limited to that known missing template.
