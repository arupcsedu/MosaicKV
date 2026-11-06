# User-requested 256-image COCO operating-point study

This extends the frozen November 6 cohort to FullKV and all four local baseline
precision/storage adaptations. It does not change any selection, codec, source
capacity, prompt, model, or generation policy. The candidate is the EviCastKV
shared-source precision/storage variant, **not the inactive six-mechanism path**.

All 256 image IDs, all reference captions, processor/chat template, media hashes,
LLaVA-1.5-7B revision, BF16 eager backend, 128 fixed predictions, and EOS-aware
caption scoring are inherited from the hash-bound parent cohort. This is a
previously inspected/development cohort, not a newly untouched confirmation.

Frozen methods have different encoded sizes. Report the measurements as
**separate operating points**, not a shared-byte comparison or proof of a better
selection policy. A nominal source-slot ratio is not an encoded-byte target.
Every new unified comparison still requires the shared-byte v2 admission rules.
The four `_reimpl`/Top-K precision adaptations are not official paper executions.

## Finite execution

1. Full CPU suite.
2. Six one-image canaries, one method each. Compare all 128 exported logits
   bitwise with and without memory instrumentation and compare measured initial
   and final backing. Check existing FullKV/EviCastKV token parity.
3. Ninety-six memory workers: six methods × sixteen 16-image shards.
4. Sixteen timing workers: all methods on each of sixteen 16-image shards,
   randomized method order within each image. Five warmups per method/worker;
   one uninstrumented full-request trial per method per image (256 trials).
5. Six paired microbenchmark workers: five compressed methods, five warmups and
   twenty repetitions each, on six fixed cohort images. FullKV is a quality
   reference and excluded from compressed-method latency-ratio tests.
6. CPU corpus CIDEr, 10,000 paired whole-corpus bootstrap replicates, unadjusted
   and familywise-adjusted quality intervals, paired repeated-timing ratios,
   raw predictions/manifests and separate Parquet measurement/summary artifacts.

Each stage is gated on execution correctness and producer completeness, **not**
on hiding unfavorable quality. `afterany` analyzers preserve failed/missing
denominators and block progression. At most six A100s in aggregate, one A100 and
twelve CPUs per worker, two-hour worker limits, reservation `bi_fox_dgx`. Arrays
are held and released only by the common aggregate admission controller. All
caches remain on scratch. No external dependencies are installed or changed.

## Measurement boundaries

- Initial/final: unique persistent decode-owned backing, including codec data,
  protected exact words, metadata and generated history—not tensor view sizes.
- Audited decode-owned peak: backing plus observed reconstruction/append
  overlap. FullKV uses its native DynamicCache append observer; compressed
  methods use their existing reconstruction/lease observer. Record those
  method-specific scope labels. Neither includes model weights, ordinary
  attention activations, vendor workspace, or prefill/construction overlap.
- Whole-request exact KV peak remains uncertified and must not be inferred from
  a decode-only peak. Previously measured allocator envelopes remain separately
  labeled conservative upper bounds.
- Latency: separate **uninstrumented** synchronized request wall time, including
  media loading, processor, prefill, construction, decode, CPU logits export,
  EOS-aware detokenization and packet retirement. Excludes model loading,
  artifact I/O and scoring. Never reuse instrumented timings as performance.
- CIDEr: complete 256-image corpus with every original reference; resampling
  recomputes document frequencies and corpus reduction, never mean shard scores.

Launch from a clean immutable SHA:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_coco_all_methods.py prepare \
  --root /scratch/djy8hg/runs/evicastkv/coco-256-all-methods-20261106-v1
```

Final paths: `summary/report.md`, `summary/summary.json`,
`summary/summary.parquet`, `summary/quality_comparisons.parquet`; each phase keeps
hashed raw predictions, complete manifests and `analysis/measurements.parquet`.
If blocked, its write-once `analysis/decision.json` identifies missing/failing
producers rather than generating an incomplete final comparison.
