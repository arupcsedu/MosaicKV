# Fox-review evidence closure, October 28–29, 2026

This is a CPU-only reporting and provenance correction. It does not modify the
runtime, coefficients, budgets, tolerances, evaluation metrics, or raw results.
The user's six-A100 aggregate ceiling remains in force; this chain requests no
GPUs. Failed scientific gates do not justify filling idle GPUs with larger runs.

## Inputs and stopping rules

- Reproduce the original 1/256 opportunity/demand development analysis and the
  separately registered 1/16 follow-up from their actual GPU producer records.
  Both cover the same 64 consumed inputs, not 128 independent examples.
- Rerun the registered 10,000-replicate corpus-CIDEr bootstrap and require exact
  numerical equality with the saved statistics. Preserve the multiple-comparison
  family and strict positive-lower-bound superiority rule.
- Seal both studies as negative/inconclusive. Do not search additional routing
  fractions or tune on these answers. The writing deadline is not a reason to
  weaken a numerical, quality, or freshness gate.
- Analyze the completed PrefixKV retry independently. Its byte-rescaled,
  append-only local baseline is an adaptation, not the official native decoding
  policy. PPL and ROUGE-L remain separate. Initial memory savings do not replace
  measured final/peak savings.

## COCO audit boundary

Read only the ID columns of the pinned local Arrow dataset after verifying its
recorded content hashes. COCO's auxiliary `id` is a row index; it is **not** the
source-image number. Canonical `question_id == file_name` supplies that identity.
No images, reference captions, predictions, new cohort, or model are read by the
ID projection. The historical ledger is read for its existing IDs and hashes.

Resolve historical directories only using original execution evidence: a sealed
COCO retry, the scheduler's extraction-only command, and the saved MileBench
media-only report. Preserve the incomplete historical media certificate.
Unused IDs establish availability, not certified independence from every prior
source/media group. This audit must not authorize confirmation by itself.

## Reproduction

From this clean committed worktree:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_review_closure_v2.py submit \
  --root /scratch/djy8hg/runs/evicastkv/fox-review-20261028-writing-closure-v2
```

Five independent CPU producers (`r256`, `r16`, `audit`, `prefix`, `tests`) feed one finite
`afterany` reporting job. Every producer has two CPUs, 12 GiB, and ten minutes in
`bii`; no continuous log watcher or automatic expansion is installed. Failures
write structured records. A missing/failed producer cannot become a passed
writing packet. A retry must use a new immutable root; raw predictions are never
rewritten.

Outputs include complete measured tables, adjusted-interval PDF, PGFPlots source,
CSV/Parquet, negative-result seals, command/job ledger, actual source hashes,
claim/evidence restrictions, and the PrefixKV gate. Descriptive quality-run
timings must not be used as repeated-trial headline latency results. The packet
is reporting-ready evidence, not a claim that positive paper confirmation passed.
