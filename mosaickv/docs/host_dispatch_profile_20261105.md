# Host decode attribution, not another cache-policy sweep

Event-only model profiling confirmed that halving the native reader interval
did not materially shorten the complete decode step. The next diagnostic
therefore identifies CPU-call costs before changing a runtime. It uses only
the same two consumed inputs, original/native EviCastKV and original PrefixKV.

One excluded full warmup per policy, then cProfile only inside decode updates
1–126. Initial graph capture is excluded. Preserve every numerical predicate,
all 128 own-parent vocabulary-logit words, values, positions and cache geometry.
No CUDA tensor workspace or model-runtime policy is added. Profiling overhead
and C-function GPU waits make these **attribution times, not acceptance times**.
Do not sum nested cumulative durations. Keep the complete binary profile and
its SHA alongside the top self/cumulative call tables.

Require a clean source/full CPU suite, then two one-A100/12-CPU workers under
the six-GPU aggregate ceiling and 20-minute limits. Terminal CPU analysis is
afterany. No fresh identities, automatic timing retry, calibration or paper run.
The registered 4% full-request latency gate remains unchanged and unpassed.
Independent benchmark-quality N is zero; whole-request peak is uncertified.
