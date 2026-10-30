# Prompt precision-tier diagnostic (not the exact-only method)

This separately labeled lossy variant retains original selected memberships,
post-RoPE phases and logical positions. Keys use time-axis groups128, values
feature-axis groups128, 3/4-bit affine min/max codes with FP32 parameters.
First four, last eight, and every selected text position stay exact BF16. A
conservative union of text slots across heads is used, including its byte cost.
The first two and last two layers retain their original exact representation.
Time-group padding is not stored in the code payload; partial-group extrema
use repeated real values, not artificial zeros. Generated history
uses the validated lossless source-history constructor. No weights are changed.
The motivation is evidence retention at lower storage without repeated prompt
projection GEMMs. This is not a paper-faithful KIVI implementation; asymmetric
key/value grouping follows the published idea, not a claimed new principle:
https://arxiv.org/abs/2402.02750 . All code here is a local implementation.

The retention-one control disables quantization and must remain bitwise exact.
The explicit lossy arms must match a dequantize/repack control bitwise and retain
all32 original greedy predictions with KL<=0.02 and max logit difference<=1.0.
No threshold changes after outcomes. CPU/GPU transport and strided-lease tests
must pass. Two consumed inputs only; no annotations or new evaluation identities.

Only if both four-bit arms pass numerical and lower initial/final/observed-peak
storage gates may twelve method/input workers execute, offering the exact same
precision policy to all six paths. Five warmups and twenty randomized paired
technical repetitions compare each method against its own exact representation.
These are not statistically independent benchmark-quality evidence or a common
realized-byte selection comparison. Whole-request HBM peak remains uncertified.
Raw model output, failures, manifests and all storage parameters are retained.
No fresh cohort, calibration or paper campaign is automatically authorized.
