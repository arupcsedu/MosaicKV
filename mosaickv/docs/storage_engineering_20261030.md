# EviCastKV lossless storage engineering: evidence index

The October 30 work is isolated from the user's dirty main checkout and paper
files. It is a consumed-input HF engineering study, **not** a paper-ready
quality comparison or a default-runtime change. EviCastKV and MosaicKV are
the same system; historical method IDs remain unchanged.

The audited runtime scope is the pinned BF16 LLaVA-1.5 MHA path, eager
attention, batch one and serialized requests. It is not support evidence for
Qwen/GQA, other precisions, optimized attention kernels or native vLLM/SGLang.
The normal evaluation CLI defaults and historical paper campaigns are unchanged.

## Validated mechanisms

- Exact cross-head normalized-source / direct-KV storage, with original
  logical positions and original K/V bit reconstruction.
- Scoped projection arithmetic control, with zero modification of weights or
  the original model's global policy outside reconstruction.
- Fused K/V projection and direct RoPE/gather into one layer lease. Dynamic
  row counts and growing strides avoid a compilation-per-token defect.
- A direct constructor eliminating redundant representations and probes only
  when a zero-byte correction lower bound proves they cannot improve storage.
  Three independent original-K/V reconstruction checks remain mandatory.
- Lossless, bounded four-layer history sealing with shared literal ownership,
  integer round trips and independent backing-storage enumeration.
- Direct reads from the existing encoded source inside projection, charging
  literal checkpoints while eliminating a decoded-source allocation.

The independent dual-kernel diagnostic measured roughly twelve-percent lower
decode latency than the preceding fused path, with unchanged outputs. The
constructor differential measured roughly 57% less EviCastKV representation
construction time. These are separate measured boundaries, **not** an added-up
end-to-end speedup. The packed-source path additionally lowers observed
decode-owned peak at a small persistent-metadata and latency cost.

## Evidence locations

All paths below are beneath `/scratch/djy8hg/runs/evicastkv/`:

| Evidence | Root |
|---|---|
| Exact-storage stages and 128-token history | `exact-storage-stages-20261030-v1-retry1` |
| Six baseline paths, selective LOOK-M repair | `quality-latency-consumed-20261030-v1` |
| Fused direct compact transport | `quality-latency-compact-direct-20261030-v2` |
| Dual projection parity and paired timing | `dual-projection-consumed-20261030-v1-retry3` |
| Same-GPU randomized baseline timing | `paired-storage-baselines-20261030-v1` |
| Common dual/history controls, all six methods | `batched-history-consumed-20261030-v1` |
| Exact-choice constructor differential | `direct-constructor-consumed-20261030-v1` |
| Negative CPU predictor/codec feasibility | `predictive-storage-consumed-20261030-v1-cpu-retry2` |
| Packed-source projection numerical and timing gates | `encoded-projection-consumed-20261030-v1` |
| Common packed-source baseline validation | `encoded-baselines-consumed-20261030-v1` |

Every producer has immutable plans, source/config hashes, raw controls,
failure records and afterany analysis. Retries preserve prior failures. The
CPU-only loader repairs reuse the captured tensors; they do not rerun GPUs.

## Negative results and claim limits

Bulk source-sharing additions and draft-ranked layer additions did not pass
the unchanged quality guard. No selector successor or fresh cohort was
promoted. Fixed row predictors and sparse-byte packets produced no storage
saving on these tensors; the preregistered prospective gate failed and that
direction stopped before GPU decoder work.

All compressed selectors use equal original source-slot capacity, not equal
realized encoded bytes. These storage-layout observations must never be called
shared-physical-byte selection-policy curves. ROUGE-L is only against a
32-token FullKV model reference, not annotation quality or native PPL.
PrefixKV/VL-Cache remain explicitly append-only local adaptations. LOOK-M's
merged values are preserved. PrefixKV is smaller and faster on these inputs;
the study does not establish universal superiority.

The current exact-subset source format does not represent LOOK-M's merge
operation. A compound merge-replay encoding might be possible but is not
implemented or evaluated. Keeping the actual merged K/V through the common
lossless codec is correct, but does not prove LOOK-M's storage is fully optimized.

Raw `full_kv` memory rows in the storage diagnostic are **all-position lossless
controls**, not ordinary uncompressed FullKV memory. Ordinary FullKV remains
the untouched parity/quality reference and is excluded from competitor latency.
Observed peaks charge actual cache-owned tensor backing and reconstruction
scratch, but exclude inactive diagnostic prefill/captures, weights, attention
activations, internal GEMM workspace and allocator reserve. Whole-request HBM
peak is not certified. Prototype/repair and other backends get no new claim.

## Regenerate the report

After the final producer and its CPU analysis finish, from a clean source:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/quality_latency_final_20261030.py \
  --history-root /scratch/djy8hg/runs/evicastkv/encoded-baselines-consumed-20261030-v1 \
  --output /scratch/djy8hg/runs/evicastkv/quality-latency-engineering-final-20261030
```

Use a new write-once output directory. The report verifies producer hashes and
retains negative outcomes. It emits separate Parquet tables and dependency-free
vector PDF/SVG charts, with its source snapshot and manifest. No package
installation or home-directory cache is required.
