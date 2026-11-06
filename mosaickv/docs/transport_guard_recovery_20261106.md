# Numerical guard correction and selective recovery

The first 256-image quality execution exposed a genuine constructor bug, not a
reason to change tolerances: the correction mask used
`isclose(expected.float(), actual.float())`, whereas verification used
`isclose(actual, expected)` in BF16. Relative tolerance is asymmetric, and BF16
arithmetic rounds differently. A synthetic CPU example (not a benchmark result)
is expected=0.375 and actual=0.39453125: the old mask accepts it, creates zero
repairs, and the registered verifier rejects it.

The correction **retains every old correction and adds every violation of the
already registered BF16 actual/reference predicate**. It does not change rtol,
atol, protected-row rejection, logical positions, selection, codec bits, reader
modes, per-layer numerical limits, or the original FullKV reference. Protected
violations are still rejected rather than hidden by patches.

For any previously successful construction, the new mask equals the old mask:
any unpatched old entry already passed the final guard, and all patched entries
were already in the old mask. CPU regressions cover this property, chunking,
the asymmetric boundary and protected rows. No benchmark label is needed.
Existing valid quality predictions are therefore preserved with their actual
producer SHAs and manifests; **their timings are not promoted as measurements
of the corrected implementation**.

The same restriction applies to whole-request allocator peaks, which include
the new transient predicate workspaces: preserve and label each historical
producer's actual observations, but do not claim that reused peaks certify the
corrected implementation on all 256 images. Initial/final and decode-owned KV
storage remain reusable only where the unchanged-valid-packet property applies.
Any conditional timing subset is measured anew on the corrected source and is
descriptive, not a substitute for whole-cohort peak certification.

The bounded recovery starts only after the existing 64-worker array finishes
and the corrected source passes the complete CPU suite. It verifies parent
artifact hashes, admits only the specific guard failure (at most 32 failed
quality samples), runs one consumed failing-input model-level canary, then
retries only those invalid quality rows. Unknown failures or a failed canary
stop progression. The new canary is a diagnostic, not independent evidence.

Keep all original warmup/failure/success attempts and new attempts in a separate
raw JSONL. A selected-row index replaces only the registered failed sample/
method keys; it does not rewrite or relabel old producers. Recompute CIDEr over
all 256 paired samples and within all 10,000 bootstrap draws. Do not average
shard scores or omit failed examples from the denominator. Report the numerical
protocol correction and original/recovered failure counts explicitly; do not
call this globally untouched confirmation or an unchanged-source replication.

Only a strict candidate-minus-FullKV lower CI >=0 plus reduced initial/final
cache and whole-request allocated HBM permits the already fixed descriptive
timing subset on the corrected source. There is no FullKV latency comparison,
compressed-baseline superiority, whole-request KV-only peak, or six-component
system claim. All newer compressed-baseline comparisons remain separately
blocked on their byte-matched lanes. Six A100s aggregate, one GPU/12 CPUs/48GiB
per worker, two-hour limits, scratch caches only. No running source is edited.
