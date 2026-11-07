# Native guard-tail execution diagnostic

This opt-in path replays the original final normalization, two vocabulary
readouts, FP64 softmax/log-softmax/KL/max/argmax operations and hidden-state
selection in a request-local CUDA graph. No fused approximation, changed
threshold, modified model weight or benchmark label is used. A nonfinite
record still fails the request. The reference path remains the default.

The path disallows the distinct risk-repair and readout-reuse variants rather
than silently dropping their branches. First-step entropy repair is unchanged.
Explicit activation backing and the observed graph allocator increment are
reported separately. They are not K/V savings or whole-request peak certification.
Graph construction and retirement remain in measured request latency.

Each worker first checks bitwise operator outputs, unchanged FP64 records,
A-B-A state isolation and nonfinite handling. Synthetic operator requests
must improve by at least 5% including graph construction before loading the
pretrained model for paired verification. Synthetic weights and operator times
are not benchmark quality or paper performance evidence. Model promotion still
requires all 128 frozen logit vectors, exact storage, full traces and adjusted
equal-or-faster request latency against both storage controls. Two consumed
engineering inputs do not establish independent task-quality superiority.
