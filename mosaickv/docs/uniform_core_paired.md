# Frozen uniform-core paired latency confirmation

The two consumed-input 5-warmup/3-repetition request screens passed exact
128-vector parent-logit words, original packet checks, initial/final cache
storage and explicitly charged 297006/298550-byte workspace. Both exceeded
the registered 4% point-estimate improvement against the two optimized
PrefixKV controls. Preserve those results, including their non-confirmatory
labels. This follow-up changes repetition count and analysis, not the runtime.

Use the same two inputs and four arms, five warmups and twenty randomized
paired timed repetitions per arm. Keep the original actual parent output
checks, snapshot-audited allocation companion, zero private graph pool,
adaptive original-eager passthrough, cold capture/retirement and per-request
constructor timing. CPU proof precedes two one-A100/12-CPU/48GiB workers with
40-minute time limits; afterany CPU analysis preserves failed producers.

Bootstrap 10000 paired trial-index draws for mean complete-request ratios and
differences, with a two-sided Bonferroni family bound of 64 contrasts. Require
the adjusted ratio upper bound <=0.96 against BOTH PrefixKV controls on BOTH
consumed inputs and smaller initial/final/decode-owned peak cache. <=1.00 is
reported separately; <=1.05 is not acceptance. Do not call three repetitions
or these repetition-level intervals independent task-quality evidence.

This confirms only the registered engineering comparison with PrefixKV.
Prompt Top-K, LOOK-M and VL-Cache are not newly measured here. This precision/
storage variant does not activate the six MosaicKV algorithm components and
does not certify whole-request GPU peak, FullKV answer quality or paper-level
superiority. Independent quality N=0. Fresh cohorts and larger paper campaigns
remain blocked; any broader experiment requires its own registered controls.
