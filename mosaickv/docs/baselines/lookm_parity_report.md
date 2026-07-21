# LOOK-M official versus `lookm_reimpl` parity report

Report date: 2026-07-20

Official source pin: `SUSTechBruce/LOOK-M@ecf0f51a9c416c2d85e47faf2638502f01a6d748`

Status: **official parity not run; `lookm_reimpl` development smoke passed but is not comparable**

This is a parity-readiness report, not a measured-results table. No synthetic,
estimated, or unequal-control number is substituted for a missing
measurement.

## Requested controlled run

The intended comparison is official LOOK-M versus `lookm_reimpl` with one
immutable LLaVA-1.5-7B checkpoint/revision, one content-addressed sample set,
identical prompts and media, tokenizer, pivotal compression ratios,
temperature-zero generation, fixed output-length policy, model precision,
eager backend configuration, and seed. Per sample, the report will compare:

- selected physical positions for every layer/head;
- realized active KV bytes;
- generated token IDs;
- benchmark score; and
- synchronized end-to-end latency.

The comparison command is:

```bash
python mosaickv/scripts/compare_lookm_parity.py \
  --official artifacts/lookm/official/artifact.json \
  --reimpl artifacts/lookm/reimpl/artifact.json \
  --output artifacts/lookm/parity_report.json
```

The validator emits sample comparisons only when every control is identical.
This includes content hashes for the Python/CUDA environment, hardware state,
and a measurement protocol defining the active-byte boundary, latency
boundary, synchronization, warmups, and repetitions. Otherwise it returns
`status: not_comparable`, lists the mismatched fields, emits no sample metrics,
and exits with status 2.

## Inspection outcome and blocker evidence

| Gate | Official LOOK-M | Unified `lookm_reimpl` | Outcome |
|---|---|---|---|
| Code provenance | Pinned submodule, unmodified, MIT | MosaicKV working tree, always labeled reimplementation | Ready |
| Model executable | Vendored original LLaVA, configured as `liuhaotian/llava-v1.5-7b` | Transformers adapter for `llava-hf/llava-1.5-7b-hf` | **Mismatch** |
| Model asset available locally | Not present in the audited caches | HF conversion snapshot is present | **Blocked** |
| Dataset/sample asset | Official scripts expect external MileBench data | MosaicKV synthetic task is available | **Mismatch** |
| Environment | Official requirements demand torch 2.1.1+cu118 and Transformers 4.37.0 | Verified HF environment uses a newer, incompatible stack | **Mismatch** |
| Portable official launcher | Worker contains author-machine absolute import paths | Unified package is portable | Requires isolated patch |
| Algorithm detail | Source hard-masks 576-position image spans | Reimplementation follows paper max-score prior | Reported deviation |

Running the two available paths would therefore violate the repository rule
requiring identical checkpoint, tokenization, sample, and backend
configuration. It would generate numbers, but not valid parity evidence. The
official 7B run and the matched `lookm_reimpl` run were deliberately not
submitted under unequal controls.

## Standalone reimplementation validation

Slurm job `17118553` completed the current `lookm_reimpl` path on the reserved
A100 allocation using the cached
`llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`,
the packaged synthetic image/prompt task, BF16, eager attention, pivotal merge,
and the 0.1 recent plus 0.1 important ratios. It wrote a complete result row,
Parquet aggregate, manifest, and compact debug trace at:

```text
/scratch/djy8hg/runs/mosaickv/lookm/reimpl-hf7b/lookm-reimpl-hf7b-17118553/
/scratch/djy8hg/runs/mosaickv/lookm/reimpl-hf7b/traces/lookm-reimpl-hf7b-17118553/
```

The trace identifies `implementation: lookm_reimpl`, `official_code: false`,
the pinned upstream SHA, selected positions, score/assignment digests,
generated-token IDs, and the timing breakdown. Its manifest correctly records
`baseline_reimpl_measured` and `canonical_eligible: false` because the source
working tree was dirty. This smoke is synthetic, has no matched official row,
and is excluded from paper results and from every numerical parity comparison.

## Current comparison fields

| Field | Official | `lookm_reimpl` | Comparison |
|---|---:|---:|---|
| Selected positions | not measured | not measured under parity controls | unavailable |
| Active KV bytes | not measured | not measured under parity controls | unavailable |
| Generated tokens | not measured | not measured under parity controls | unavailable |
| Benchmark score | not measured | not measured under parity controls | unavailable |
| Latency | not measured | not measured under parity controls | unavailable |

These cells are status labels, not numeric experimental observations.

## Required path to a valid parity row

1. Materialize an immutable sample/media set and record its SHA-256 manifest.
2. Materialize one immutable LLaVA-1.5-7B checkpoint usable by both executables,
   or construct and validate a deterministic conversion whose tensor hashes,
   tokenizer output, prompt embeddings, and FullKV generated tokens match.
3. Create the official legacy environment from its pinned requirements and
   record a successful import/CUDA smoke manifest. Do not retrofit the primary
   HF environment and still call it official.
4. Apply any portability/instrumentation changes only from an isolated patch
   under `third_party/patches/LOOK-M`; record the patch SHA in the official
   manifest.
5. Run both methods on the same reserved GPU allocation with identical power,
   clock, concurrency, warmup, and repetition controls.
6. Export `official_lookm` and `lookm_reimpl` artifact JSON and run the strict
   comparator above.

Until all six steps pass, paper tables must place official-source results and
reimplementation results in separate, explicitly non-comparable sections.
