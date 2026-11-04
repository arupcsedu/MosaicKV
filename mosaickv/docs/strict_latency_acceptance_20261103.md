# Strict consumed-input latency acceptance amendment

The user clarified on November 3, 2026 that **equal or faster is required**.
A method up to 5% slower is only a reported trade-off, never a pass. This
amendment changes analysis/progression only; running runtime snapshots, model
weights, reader policies, numerical guards, storage boundaries and raw
measurements are unchanged. Preserve legacy analyses and negative runs.

The candidate's simultaneous-adjusted paired repeat-bootstrap upper latency
ratio must be <=1.00 against every measured compressed-baseline arm. <=0.95
is the stronger 5%-faster result. <=1.05 is secondary descriptive telemetry.
Five warmups and twenty timed repetitions per arm are mandatory. Bootstrap
10,000 paired repeat resamples; Bonferroni two-sided intervals use a fixed
maximum family of 64 contrasts across two inputs and two engineering stages.
These intervals quantify repetition uncertainty, not independent-example
quality or generalization. The completed cached-text stage is retrospectively
reclassified in a separate derived directory, without GPU reruns.

The corrected direct-construction study has its own frozen runtime SHA. A
separate clean analyzer SHA validates producer hashes, runtime/plan identities
and the runtime CPU-suite proof. Its finite afterany chain launches the same
registered paired timing only after tensor/logit/storage controls pass. It
launches the already registered all-five timing only if strict paired latency
passes. No fresh examples, calibration, paper campaign or reader promotion is
authorized. The runtime source is never rewritten by the analyzer.

All five methods are local precision/storage adaptations, not newly certified
official baseline implementations. These are two consumed inputs, with zero
independent task-quality observations. Parent vocabulary-word parity does not
repair existing benchmark-quality deficits. Different parent cache sizes are
not byte-matched algorithm comparisons. Peak storage remains decode-owned
tensor backing, not certified whole-request GPU peak; allocated/reserved CUDA
peaks are separately recorded.
