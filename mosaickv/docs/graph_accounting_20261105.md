# Single-pass and event-updated graph-cache accounting

Actual consumed-input host profiling found 516,096 lease-byte property calls
over 126 profiled updates. PromptGraphCache.observe redundantly enumerated all
layers twice. This execution-only ablation retains the original path and class
as a control, offers one-pass sums, and separately offers event-updated sums.

Event sums bind only unique private fixed graph layers. Resident-byte setters
notify changes; effective lease charges follow the original shared-alias rule.
New layers bind once. Shared and graph-private bytes are still read at every
observation, and extra storage is still independently enumerated. Audited GPU
requests compare the event total with the original full layer sum at **every
observation**. All requests retain the independent final tensor-storage check.
Missing notifications, foreign/replaced layers or underflow abort. No numerical
check is removed; no bytes, model operations, values, logical positions or
allocation boundary changes. CPU counters hold no additional CUDA storage.

The ordinary class does not gain a resident-byte property getter, so original
controls are not slowed to create a win. Offer event sums to PrefixKV original
and optimized graph controls too. Compare seven arms with five warmups and
three randomized timed repetitions per consumed input, after full CPU tests and
audited complete-payload, all-128-logit and three-boundary byte equality gates.
Two one-A100/12-CPU workers, maximum six A100s aggregate, 20-minute job limits;
afterany terminal analysis. Three repetitions are screening, not formal latency
acceptance. No automatic larger/quality run. Independent quality N is zero.
The unchanged formal target is an adjusted upper full-request ratio <=0.96
against every original and optimized baseline. Whole-request peak is uncertified.
