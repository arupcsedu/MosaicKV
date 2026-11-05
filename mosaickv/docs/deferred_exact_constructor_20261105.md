# Deferred exact-body encoding: consumed-input constructor diagnostic

This default-off execution variant avoids losslessly encoding intermediate BF16
bodies that a later text-overlay constructor discards. Intermediate bodies own
raw contiguous clones, bounded to 8 MiB each; no original prefill KV view is kept
alive. Surviving bodies receive the **unchanged** original lossless codec before
per-layer retirement, validation, ownership accounting, and decoding. Aliases
are preserved. The weak memo holds no GPU owners. The context is request-private.

Storage-funded correction selection and refitting explicitly finalize inputs
before inspecting bytes. They must make precisely the same allocation decisions
as the parent. Final tensor words, metadata, selected values, logical positions,
and all initial/final/decode-owned peak bytes must match their own parent.
Intermediate construction allocations are not omitted from PyTorch's allocator;
whole-request KV or GPU peak remains uncertified and is not a superiority claim.

Offer this constructor to PrefixKV controls as well as EviCastKV. The latter also
uses the previously tested aligned text-row constructor. Compare against both
original and optimized PrefixKV readers and retain the original EviCastKV graph
reader. Each source gets a clean full CPU suite before two consumed-input A100
workers (one GPU/12 CPUs each, maximum six aggregate, 20-minute limits).

Audited requests require identical final payload hashes and all 128 parent-logit
vectors bitwise. Then five warmups and three randomized timed repetitions form
a cheap **screen only**, never formal acceptance. Terminal analysis runs afterany,
even on failures. A 4% point estimate cannot replace the registered 20-repetition,
multiplicity-adjusted upper ratio <=0.96 against every original/optimized control.
No automatic formal run, new identities, calibration or paper campaign follows.
No training, labels, generated-output tuning or threshold changes are allowed.
This variant is a full-prompt shared-source storage variant, not validation of
the six-component algorithm. Independent benchmark-quality N remains zero.
