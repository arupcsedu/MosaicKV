# MosaicKV paper-readiness gate

## Current decision

The repository has a **paper-eligible exact-selection evidence lane**, but it
is not eligible to claim that every planned MosaicKV tier or every published
baseline has been reproduced. This distinction is release-blocking:

- the evaluated method is `mosaickv_exact`;
- prototype merging and residual repair are disabled by the compatibility gate
  for the evaluated LLaVA cache layout and must not be presented as measured
  components;
- comparisons at a shared physical byte cap are valid selection-policy
  comparisons, while different-cap points are quality--memory curves rather
  than same-budget latency comparisons; and
- paper-native baseline evidence is separate from unified comparison evidence.

This document supersedes development notes that referenced source commit
`0a0d6e5`. It records current immutable evidence at
`acdfb0ec7025b81cb2deeb574034aca70f4ca5c2` and does not reinterpret older
artifacts.

## Passing evidence

### LLaVA-v1.6-Mistral FullKV equivalence

Clean GPU job `21196378` loaded
`llava-hf/llava-v1.6-mistral-7b-hf@2424fdd47412fccc66d91719126b420e9fbd7065`
with HF eager attention and FP16. Its 16-token explicit FullKV decode loop
matched `model.generate()` exactly: token agreement was 16/16 and maximum
logit difference was `0.0`. The immutable report is
`/scratch/djy8hg/runs/mosaickv/release-preflight-acdfb0e/llava-v16-retention1-acdfb0e/report.json`.

The pinned revision is included in the checkpoint-parity registry and GPU test
parametrization. This establishes the required FullKV reference only for that
model/backend/precision/configuration; it is not a blanket parity claim for
other model families or attention kernels.

### Unified exact-byte VL-Cache lane

The immutable analysis artifact
`/scratch/djy8hg/runs/mosaickv/release-preflight-acdfb0e/vl-cache-shared-byte-v2-analysis-acdfb0e`
contains 8,192 completed locally measured rows:

| Task | Metric | Paired samples per cap | Caps | Compressed methods |
|---|---|---:|---|---|
| TextVQA | Accuracy | 256 | 10, 20, 40, 80 MiB | Prompt Top-K, `vl_cache_reimpl`, `mosaickv_exact` |
| DocVQA | ANLS | 256 | 10, 20, 40, 80 MiB | Prompt Top-K, `vl_cache_reimpl`, `mosaickv_exact` |

For every task/sample/cap tuple, the three compressed methods have identical
measured active K/V bytes and remain at or below the cap. FullKV is a paired
quality control and is intentionally not constrained by the compressed cap.
The result validator, `scripts/validate_shared_byte_curve.py`, checks row
completeness, unique pairing, pinned source/model revisions, byte-cap
compliance, exact physical byte equality, task-local metrics, and paired
bootstrap denominators. It does not calculate or permit a mixed Accuracy/ANLS
macro score.

```bash
source mosaickv/scripts/cache_env.sh
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/validate_shared_byte_curve.py \
  --artifact-root /scratch/djy8hg/runs/mosaickv/release-preflight-acdfb0e/vl-cache-shared-byte-v2-analysis-acdfb0e \
  --expected-source-sha acdfb0ec7025b81cb2deeb574034aca70f4ca5c2 \
  --expected-model-revision 2424fdd47412fccc66d91719126b420e9fbd7065
```

This validator passed on 2026-10-11: 2,048 sample/cap byte checks and all
8,192 raw rows passed. This is appropriate evidence for an apples-to-apples
curve. It must not be converted into a claim that MosaicKV uses fewer bytes
than Prompt Top-K *at the same cap*—they are intentionally equal there.

### PrefixKV paper-native operating point

The pinned official repository is
`THU-MIG/PrefixKV@597f1ab032704951550f93bcc8a23f1454b80aa4` (MIT). The
sealed official PPL operating-point artifact evaluated 1,000 examples and
reported PPL `3.40625` versus the registered paper target `3.41`, inside its
pre-registered `0.25` tolerance. Its artifact is
`/scratch/djy8hg/runs/mosaickv/release-preflight-acdfb0e/prefixkv-official-operating-point-21119992/operating_point_comparison.json`.

Clean job `21216557` additionally compared the official PrefixKV path with
`prefixkv_reimpl` on the same LLaVA-Description sample and profile. The methods
differed by 16 KiB over a roughly 164 MiB active cache, one retained position
in one layer, and 0.0127 PPL. This is implementation calibration, but one
sample is **not** a paper-native ROUGE-L reproduction. The artifact remains a
diagnostic only.

## Required result categories

Every paper table must label a row as exactly one of:

1. **Unified comparison, locally measured** — same model, processor, prompts,
   media, tokenization, generation controls, precision, backend, block policy,
   mandatory policy, and measured active-byte cap.
2. **Paper reproduction, locally reproduced** — the paper checkpoint,
   dataset, metric, budget convention, and supported environment match, and a
   registered operating point passes its tolerance.
3. **Paper reported** — a value transcribed from a paper, never displayed as
   if it came from local execution and always accompanied by source metadata.

No row may combine categories 1 and 2 merely because its method name is the
same. `*_reimpl` labels are mandatory for local implementations.

## Remaining paper blockers

| Blocker | Status | Required resolution before the corresponding claim |
|---|:---:|---|
| Exact-only scope | Open | Frame results as `mosaickv_exact`. Develop an audited RoPE-safe prototype/repair path plus parity and quality evidence, or remove three-tier performance claims. |
| PrefixKV native ROUGE-L lane | Execution pending | The common-HF padded transport is now rejected. The isolated pinned upstream runner supports disjoint multi-sample calibration/evaluation sets and token-weighted corpus PPL; run it cleanly with frozen FullKV references and reproduce a registered ROUGE-L point. Do not use anomalous TextVQA/DocVQA `prefixkv_reimpl` scores as PrefixKV evidence. |
| VL-Cache paper-native lane | Execution pending | No official release exists. The COCO Caption CIDEr route is executable and formula-validated; retain labels as `vl_cache_reimpl`, complete the registered COCO/DocVQA reproduction sample set, and compare published trends separately. |
| LOOK-M/MileBench lane | Upstream-blocked | The pinned upstream source imports a missing attention class and assumes author-local paths/original checkpoint. The exact blocker is documented in `docs/baselines/lookm_official_execution_blocker.md`. Retain only `lookm_reimpl` unless the authors provide a compatible pinned revision. Report Accuracy and ROUGE-L separately. |
| Secondary baselines | Open | H2O, StreamingLLM, PyramidKV, Elastic Cache, SnapKV, and RoCo need audited/pinned upstream sources and operating-point reproduction before empirical rows. |
| MathVista | Blocked | Exclude until a pinned local official scorer is validated or authorized judge credentials are supplied through the environment. |
| Native serving claim | Qualified | Exact whole-block transport has development evidence, but vLLM needs a matched canonical campaign before a paper-quality claim. Prototype injection and repair remain unsupported. |
| Timing claim | Qualified | Report planner-inclusive timing as secondary. Do not claim a general decode win from HF eager results; paged-engine and long-context/batch evidence is required. |
| Container verification | Open | Dockerfiles and lock files exist, but target-platform container execution remains required for an artifact claim. |

## Admission rule for a new empirical table

Before adding a locally measured comparison table, all of the following must
pass for the row family:

1. clean immutable source SHA and complete manifest provenance;
2. retention-1 FullKV equivalence for checkpoint/backend/precision;
3. official metric fixture validation and recorded evaluator version;
4. identical paired sample IDs, prompts, media, preprocessing, generation,
   output length, seed, precision, and backend configuration;
5. measured active bytes at the same accounting boundary; and
6. no missing, failed, non-finite, fallback, or provenance-mismatched rows,
   with raw failures and denominators retained.

Paper-native baseline rows additionally require published operating-point
reproduction inside a pre-registered tolerance. A plan, static adapter, or
synthetic test is not evidence of this requirement.
