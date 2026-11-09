# Direct storage comparison and bounded readout profile

First require complete positive paired fused-versus-unfused timing from
`mixed-history-projection-20261108-v1/paired-timing`. Do not multiply that ratio
by a storage comparison from another job. The next registered experiment runs
storage-only/common-headers, selective/unfused and selective/mixed on the SAME
two consumed inputs, GPU allocations, process, prompts, weights, output horizon,
common optimizations and controls. Five warmups and twenty repetitions per arm;
randomize arm order within each repetition. This remains development evidence.

All 128 storage-only logit vectors must equal the frozen storage parent. Both
selective arms must satisfy the unchanged FullKV-parent 32/128-token numerical
guard AND equal the frozen unfused-policy logits bitwise. No precision weights,
history window, memory budget, numerical tolerance, reference or benchmark label
changes are allowed. Preserve every raw row and failure. Initial/final storage
is recorded; a timed request does not inherit another run's peak certificate.

There are twelve registered timing endpoints: two inputs, three pairwise
comparisons and two latency metrics. Bootstrap twenty paired repetitions with
10,000 replicates and Bonferroni family-wise 95% intervals. Equal-or-faster
requires an adjusted upper ratio <=1.00 against the actual storage comparator.
<=1.05 remains only a trade-off. Do not select a favorable comparator afterward.

Only the FIRST EXCLUDED mixed warmup captures optional phase intervals. Sample
actual reads—not construction replays—at sealed lengths 32/64/112 in layers
0/16/31, for K/V separately. Record source unpack, native projection, output
copy/RoPE and numerical correction host times and CUDA stream intervals. Stream
intervals include host dispatch gaps and must not be called pure kernel time.
Record all 192 constructor host durations separately. Default profiling is off;
no event instrumentation runs in timed repetitions. No new persistent GPU source
workspace is introduced. Model/KV arithmetic and all checks remain unchanged.

Use the dominant measured phase to justify any next implementation. The direct
comparison and phase audit never authorize fresh cohorts, calibration, a paper
claim or initial-memory superiority. One A100/12 CPUs per worker, two workers,
one-hour limit, aggregate admission <=6, finite `afterany` analysis.
