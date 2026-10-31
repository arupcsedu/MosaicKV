# Common mixed-boundary/refit codec development study

This separately versioned study stops the failed draft and value-aggregation
directions; no prototype, repair, new forecast, training or model change is used.
Use only the same 16 consumed COCO images, not a fresh cohort. Freeze settings
before observing the new codec's scores. Do not tune from the 112-image answers
or failure identities. No independent or paper-level quality claim is permitted.

The new **common** format combines existing operations: eight-bit K/V in the
first/last two layers; three-bit K and two-bit V in middle layers; actual
two-bit-error scalar refitting. Text and first-four/last-eight selected positions
retain the existing exact-protection policy. Keys are never phase-averaged.
No MSE range clipping is used. Every method can use this identical format.

Seven measured points: mixed format for EviCastKV, Top-K, LOOK-M, VL-Cache and
PrefixKV; plus balanced exact-boundary two-bit refit controls for EviCastKV and
Top-K. EviCastKV/Top-K use balanced capacities, other methods retain their native
local allocation (LOOK-M already balances). All spend the same source-slot cap,
but encoded sizes differ: **not shared-byte-matched comparisons**. Preserve
LOOK-M's actual merged K/V, all mandatory/logical positions, and lossless
append-only generated history. All rows remain explicitly local adaptations.

Reuse all 656 older rows, including negative candidates and stronger smaller
LOOK-M points; add only 112 requests, in 28 four-image shards. Each worker uses
one A100, 12 CPUs, 48 GiB host memory and at most 30 minutes, maximum six A100s
in aggregate. Full CPU verification precedes one execution canary, then finite
afterany promotion/analysis. Unchanged union3 model-logit hashes and ordinary
FullKV generate/reinjection checks must pass. No arbitrary fallback is accepted.

Use corpus CIDEr with 10,000 paired official-reducer bootstrap replicates.
There are 48 compressed points after adding seven, plus FullKV quality-only.
Each of two EviCastKV candidates has 47 contrasts: 94 Bonferroni-adjusted tests.
Report initial/final/decode-owned peak bytes, construction/compression overhead,
actual output truncation flags and complete provenance. Whole-request KV peak
is not certified; no general no-slower claim is made. Original five-warmup rows
are not directly latency-comparable with these one-warmup ablations.
