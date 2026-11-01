# Paired complete source-request timing on the same consumed inputs

Require the immutable streaming-prefill canary's payload, 128-token and complete
teacher-logit identity and lower observed prefill/cache-transaction peak gates.
Bind all its evidence. No additional images, labels, coefficients or selections.
This compares streaming execution to its OWN frozen offline source constructor,
not to paper baselines, FullKV latency or an independently validated algorithm.

Use identical BF16 eager model, media, chat template, exact text, profile,
capacity, headers, decoder and 128 output tokens. Both use validated ONE-head
lossless generated-history projection. The four-head history experiment was
numerically correct but did not establish a speed improvement; do not promote it.

Give the offline constructor the smallest mathematically equivalent evidence:
save only the last 16 prompt attention rows and the full BF16-head-mean then
FP32-query-mean prior. Its existing `priors` API supports this without changing
scoring. A CPU fixture and FIRST measured payload/hash per arm must agree with
the frozen full-capture constructor. Do not exaggerate the streaming gain by
forcing the offline control to retain all 32 dense attention matrices or build
discarded codecs. Both produce one original full prefill per request.

Five paired warmups and twenty timed paired requests, randomized method order
per repetition from the fixed seed plus input index. Every call must produce
bitwise frozen 128-vocabulary-logit trajectories and identical initial bytes.
Persist warmup/trial rows incrementally. Failed/missing/duplicate measurements
fail closed. Use CUDA events/synchronization and monotonic wall time.

The clock includes processor work, vision encoder/projector, language prefill,
actual source construction, source retirement/GC and decoding, including normal
measurement/storage-summary costs. PIL media loading from disk and model loading
are outside; warm media and weights are disclosed. Hashing/evaluating predictions
is outside timing. Record preprocessing, combined prefill+construction+retirement,
decode, and total times, plus allocations. Do not call this a cold-start benchmark.

Tensor-result auditing is disabled in BOTH arms. Use the prior audited canary
for cache/encoder peak evidence; unaudited counters are not substituted as peaks.
Report p5/p50/p95, mean, standard deviation, 10,000-replicate paired bootstrap
intervals for total difference and ratio. Engineering target: paired upper ratio
bound <=1.05. Preserve and disclose a slower result rather than changing criteria.

Two one-A100/12-CPU workers, 45-minute bounds, after a clean complete CPU suite;
finite afterany analysis. No fresh cohort, paper promotion, strongest-baseline
superiority claim or averaging PPL with ROUGE. Baseline-compatible streaming
construction and independent quality evidence remain separate requirements.
