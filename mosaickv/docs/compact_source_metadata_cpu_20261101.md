# Compact source metadata: CPU-only feasibility

Three-bit per-channel shared projection-input encoding uses FP32 minimum and
scale for every feature/time group. Test FP16 storage of those two headers,
leaving code bits, grouping, source positions, original phase and lossless
protected K/V overlays unchanged. This is an explicitly lossy metadata
variant, not a lossless-codec claim. Reject nonfinite metadata and zero or
negative scales after conversion; never clip, hide a fallback or raise caps.
Any later comparative use must offer the same option to applicable baselines.

Reuse only the two consumed CPU normalized captures and frozen weights. Do
not use model outputs, benchmark answers, annotations or scores for fitting.
Repeat the registered protected 3/4-bit x time/feature source reconstruction
study. Primary engineering candidate is still 3-bit time-axis encoding.
In addition to the original 20% reconstruction improvement and p95/cap gate,
require its global MSE and p95 to be no more than 1.05 times the unmodified
FP32-header source on EACH input, with at least 1 MiB prefix savings. This
5% numerical allowance is specific to this CPU feasibility question; the
32-token model KL/logit/token thresholds and FullKV parity do not change.

All headers, lossless overlays, row maps and shared phase count. Prefix-only
CPU bytes exclude history, reconstruction leases and construction workspace;
no GPU/whole-request peak or latency conclusion follows. Even a pass only
justifies a separately registered same-input GPU test; no fresh cohort or
paper campaign. Full CPU suite first, two CPU threads, 16 GiB, 20 min limit.
