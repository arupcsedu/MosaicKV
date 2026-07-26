# Selector calibration and oracle gate

This document defines the exploratory calibration stage used before any new
MosaicKV quality claim. Calibration rows are not paper results. Their sample
IDs must be frozen before variant outcomes are inspected, and every later
evaluation ledger must be disjoint.

## Primary target

The intended claim is matched-quality active-memory reduction. If the strongest
baseline uses `B` measured active KV bytes, MosaicKV must use at most `0.60 B`
while satisfying a preregistered paired quality non-inferiority margin. CPU
residual bytes, transfer latency, and repair re-decodes are reported separately
and must not be hidden from the comparison.

## Gated workflow

1. Run the variants in
   `configs/experiments/selector_calibration.yaml` on one immutable calibration
   ledger. These rows compare normalization, prompt/draft evidence, and the
   per-layer/KV-head reserve at identical physical bytes.
2. On fixed diagnostic calibration samples, collect evaluation-only FullKV
   future queries and attention with `diagnostics.forecast_oracle_steps > 0`.
   Record query cosine similarity, attention-rank correlation, selected-block
   recall, and forecast regret. Oracle outputs are written only to the trace and
   are never passed to the online compressor.
3. Freeze exactly one configuration using a rule declared before reading the
   evaluation ledger. Evaluate it once on disjoint sample IDs alongside FullKV
   and the strongest baseline.
4. Test the frozen configuration at `0.1` retention against the baseline at
   `0.2`. A nominal ratio is only a scheduling label; the claim uses measured
   active bytes and requires the MosaicKV/baseline byte ratio to be at most
   `0.60`.

The fixed variant-selection rule is: maximize mean paired calibration task
score; break ties by lower mean compression time, then lower mean oracle
forecast regret, then lexicographic variant name. The immutable ledgers are
`configs/ledgers/selector_calibration_docvqa.json` and
`configs/ledgers/selector_evaluation_docvqa.json`; they must have no sample-ID
overlap.

## Utility scaling decision

Forecast attention is a probability distribution within every layer/KV head.
The previous implementation combined it with unnormalized value norms and
cosine-distance novelty, making the configured coefficients depend on sequence
length and feature magnitude. `layer_head_l1` now puts attention, expected value
contribution, and uniqueness on the same nonnegative unit-sum scale within each
layer/KV-head stratum before applying the preregistered signed equation.

The dense selector may reserve a configured fraction of each layer/KV-head
capacity before global lazy greedy selection. Reserved blocks still count
against the same hard byte budget and are recorded with the
`stratum_reserve` reason.

## Repair gate

Qwen2.5-VL stores post-M-RoPE keys and its adapter currently declares both
prototype merging and residual repair unsupported. Do not enable
`mosaickv_full`, simulate repair, or publish native repair rows for this model.
Plan C may proceed only after an adapter-specific correctness proof establishes
safe block replacement and retention-1 parity. Until then, the calibrated
method is an explicitly exact-only MosaicKV variant.

## Exploratory outcome (2026-07-25)

The calibration rule froze `normalized_hybrid1_reserve0`. All six compressed
variants scored `1.0` on the five-ID calibration ledger at the same measured
active-byte budget; the frozen variant had the lowest MosaicKV compression
time. Slurm job `17302674` then ran the evaluation-only oracle audit. Its
five-sample means were query cosine `0.8083`, attention-rank correlation
`0.9079`, selected-block recall `0.8079`, and forecast regret `0.01027`. Every
trace records `evaluation_only_output_not_passed_to_compressor`.

The disjoint five-ID evaluation ledger produced the following measured rows.
These are exploratory validation measurements, not paper results or evidence
of benchmark-wide statistical significance.

| Method | Retention | Mean active KV (MiB) | Mean score | Answer agreement with FullKV | Compression (s) | End to end (s) |
|---|---:|---:|---:|---:|---:|---:|
| `full_kv` | 1.0 | 72.309 | 1.0 | 1.0 | 0.000 | 4.282 |
| `prompt_attention_topk` | 0.2 | 14.442 | 1.0 | 1.0 | 5.586 | 11.365 |
| `mosaickv_exact` | 0.1 | 7.207 | 1.0 | 1.0 | 5.354 | 11.477 |

Thus, on this frozen slice, `mosaickv_exact` at 0.1 used `49.90%` of the
active bytes of the nearest validated competitor at 0.2: a `50.10%` active-KV
reduction at identical answers and task score. This clears the exploratory 40%
memory target. It does **not** clear a latency target: the 0.1 MosaicKV row was
`0.98%` slower end to end than prompt-attention Top-K, which is within the noise
of these single passes and must not be called a speedup or slowdown. Broader
tasks, more samples, independent seeds, paired confidence intervals, and the
preregistered 5-warmup/20-repetition timing protocol remain mandatory before a
paper claim.

The latest paired method rows were sequential Slurm jobs `17325857` and
`17325858` on the same A100 node; the FullKV row remains the earlier frozen
reference. All rows had five results, zero sample errors, complete timing
fields, zero repair events, and dirty-tree manifests with
`canonical_eligible=false` as required for this exploratory stage.

## Systems optimizations

The online streaming-centroid implementation previously synchronized CUDA for
every query norm and centroid assignment. CUDA query sources are now staged
once per source and the same deterministic online update runs on the host.
Unrestricted same-layer/same-head graph compatibility and top-k ordering are
vectorized, fallback candidates are capped before edge insertion, block pooling
is dispatched in equal-width batches, descriptor validation avoids duplicate
array conversion, and symmetric value similarities are evaluated once. The
low-memory centroid fast path retains bitwise reference-equivalent centroids.
One fixed 2,033-token calibration sample preserved graph, selection, forecast,
packed-slot, generated-token, answer, score, and byte identities while changing:

- forecast construction: `0.625 s` to `0.487 s` (`22.0%` lower);
- graph construction: `2.452 s` to `1.805 s` (`26.4%` lower);
- compression: `8.528 s` to `7.659 s` (`10.2%` lower);
- end-to-end time: `15.929 s` to `14.926 s` (`6.3%` lower).

Non-diagnostic traces now retain deterministic counts and SHA-256 hashes for
large record catalogs; only frozen diagnostic samples retain the full catalogs.
Catalog digests are computed without first materializing graph-edge dictionaries.
Trace construction and hashing are measured separately and excluded from
inference latency for every method; the final held-out means were `0.324 s` for
Prompt Top-K and `0.743 s` for MosaicKV. Trace I/O was already outside the
latency boundary. These timings are single exploratory passes, not
20-repetition microbenchmarks.
