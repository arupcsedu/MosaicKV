# COCO cross-budget retrospective analysis

This is a **retrospective analysis of existing predictions**, not a new
held-out experiment. The predictions and operating-point estimates had already
been inspected. No GPU inference, selection-policy change, budget change, or
MileBench gate change is involved.

## Evidence and statistical procedure

- Cohort: the same 256 COCO Caption images, with all reference captions retained.
- Model: `llava-hf/llava-v1.6-mistral-7b-hf`, revision
  `2424fdd47412fccc66d91719126b420e9fbd7065`; HF eager, FP16, 64 generated tokens.
- Dataset: `lmms-lab/COCO-Caption`, revision
  `c3213b54e173ae86517d481de17f31439d6f8c72`, validation split.
- Inference source: `edb6edfaf50f4539c9c1969371fefecf0eabccbf`.
- All 3,328 original rows are complete. Source report, plan and derived-file
  hashes are verified. Per-image model/media/prompt/generation/evaluation/GPU
  control fingerprints are identical across compared arms.
- Measured initial-cache grid: 10, 20, 40 and 80 MiB for every compressed method.
  All six lower-cap pairs per baseline are evaluated, including negative points:
  10/20, 10/40, 20/40, 10/80, 20/80 and 40/80.
- Baselines: `vl_cache_reimpl` and Prompt Top-K; these are not official VL-Cache
  measurements. FullKV is not used as a latency comparator.
- The separately planned primary 25/40 MiB pair has no measured 25 MiB arm.
  Both such baseline comparisons remain explicitly unavailable. A 20 MiB point
  is not silently substituted for the missing 25 MiB point.
- Statistics: 10,000 paired draws with seed 20271014. Each draw recomputes the
  **complete corpus CIDEr**, including reference document frequencies, TF-IDF,
  clipped similarities and corpus aggregation. Frozen per-image/shard CIDEr
  scores are never averaged. The accelerated calculation matches the unchanged
  pinned official reducer on 162 full-corpus fixtures (maximum absolute
  difference below `1e-12`).
- End-to-end and decode latency use those same paired draws. A latency ratio is
  the ratio of resampled cohort means, not the mean of per-image ratios.
- Both unadjusted 95% intervals and simultaneous Bonferroni-adjusted intervals
  are stored. The conservative global family includes 14 planned pairs times
  five endpoints: CIDEr, end-to-end difference/ratio and decode difference/ratio.
  The two unavailable pairs count toward correction. Each adjusted interval has
  99.9286% marginal confidence, targeting familywise 95% coverage.

## Results against VL-Cache reimplementation

Positive CIDEr differences favor MosaicKV. All intervals below are adjusted.
Memory reductions use **measured backing storage**, not nominal retention.

| Mosaic / VL-Cache initial MiB | Initial / final / peak reduction | CIDEr difference [CI] | End-to-end ratio [CI] | Preserved quality? |
|---|---:|---:|---:|---|
| 10 / 20 | 50.00% / 35.87% / 35.87% | −0.454 [−5.167, +4.577] | 0.767 [0.759, 0.774] | Not established |
| 10 / 40 | 75.00% / 62.66% / 62.66% | −4.682 [−12.536, +1.555] | 0.691 [0.683, 0.697] | Not established |
| 20 / 40 | 50.00% / 41.78% / 41.78% | +0.121 [−4.069, +4.407] | 0.724 [0.717, 0.731] | Not established |
| 10 / 80 | 87.50% / 79.66% / 79.66% | −6.876 [−14.999, +0.168] | 0.590 [0.586, 0.595] | Not established |
| 20 / 80 | 75.00% / 68.28% / 68.28% | −2.074 [−6.357, +1.729] | 0.619 [0.615, 0.623] | Not established |
| 40 / 80 | 50.00% / 45.52% / 45.52% | +1.441 [−1.924, +6.017] | 0.687 [0.681, 0.692] | Not established |

For the promising 20/40 point, measured storage is 20 versus 40 MiB initially
and 27.875 versus 47.875 MiB at both final and peak boundaries. End-to-end time
is 15.029 versus 20.753 seconds: difference −5.724 seconds, adjusted interval
[−5.994, −5.484]. CIDEr is 107.965 versus 107.844, but its quality interval
crosses zero even before correction: unadjusted 95% interval [−2.196, +2.396].

For 40/80, final and peak storage are 47.875 versus 87.875 MiB. End-to-end time
is 16.676 versus 24.278 seconds: difference −7.602 seconds, adjusted interval
[−7.839, −7.384]. CIDEr is 111.480 versus 110.039; the unadjusted quality
interval is still [−0.658, +3.945].

## Verdict

**12/12 available pairs pass the ≥30% memory and end-to-end no-slower tests;
0/12 pass the zero-margin quality test.** Therefore none establishes the complete
lower-memory, preserved-quality, no-slower claim. No cross-budget quality test
passes even at unadjusted 95%, so the verdict is not caused by the conservative
multiple-testing correction.

The narrower defensible statement is descriptive: on this previously inspected
COCO cohort, some MosaicKV points have both lower memory/time and slightly
higher CIDEr point estimates than `vl_cache_reimpl`. Statistical quality
preservation is unresolved. A confidence interval crossing zero is not proof
of equivalence. Prompt Top-K comparisons remain in the complete report; no
nearest-competitor superiority claim follows. The failed MileBench gate remains
failed and cannot be rescued by this COCO analysis.

## Limitations and artifacts

The cohort came from previously evaluated COCO identities. The underlying
timing protocol has five warmups and one measured quality trial per image,
not a randomized 20-repeat latency experiment. Paired example bootstrapping
does not characterize run-to-run GPU variability. The global adjusted tails
contain only about 3.57 of 10,000 draws, so extreme quantiles have Monte Carlo
uncertainty. Peak active KV storage excludes model weights and process-wide
allocator reservations. Analysis runs are explicitly labeled retrospective
and dirty-tree-derived; original inference remains bound to its clean SHA.

- [Complete report, including all Prompt Top-K pairs and decode latency](/scratch/djy8hg/runs/mosaickv/coco-cross-budget-retrospective-20261022-v2/report.md)
- [Machine-readable report and provenance](/scratch/djy8hg/runs/mosaickv/coco-cross-budget-retrospective-20261022-v2/report.json)
- [All paired tests, Parquet](/scratch/djy8hg/runs/mosaickv/coco-cross-budget-retrospective-20261022-v2/paired_tests.parquet)
- [All paired tests, CSV](/scratch/djy8hg/runs/mosaickv/coco-cross-budget-retrospective-20261022-v2/paired_tests.csv)
- [All available and unavailable pairs](/scratch/djy8hg/runs/mosaickv/coco-cross-budget-retrospective-20261022-v2/pair_catalog.parquet)
- [Resampled corpus scores and paired draw counts](/scratch/djy8hg/runs/mosaickv/coco-cross-budget-retrospective-20261022-v2/bootstrap_corpus_scores.npz)
- [Preserved analysis/evaluator source snapshot](/scratch/djy8hg/runs/mosaickv/coco-cross-budget-retrospective-20261022-v2/analysis_source.zip)
- [Analysis script](../scripts/analyze_coco_cross_budget_retrospective.py)

No original prediction/result file or paper text was overwritten. No jobs were
submitted and no commit was created.
