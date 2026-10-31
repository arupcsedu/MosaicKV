# Frozen small task diagnostic: range fit and boundary precision

After both consumed-input numerical diagnostics complete with no execution
errors, assess actual COCO captions on the SAME 16 consumed images.
Compressed-token guard failures remain negative evidence; task scoring is a
separate diagnostic question, never a conversion of those failures into passes.

Freeze these arms before reading the new task predictions:

- `head3k2v_mse`: original lossless boundary layers, three-bit K, two-bit V
  with fixed cache-local affine MSE fitting and the same funded exceptions.
- `head3_boundary8`: eight-bit K/V in the first two/last two layers, three-bit
  K/V in middle layers. Text and selected first4/last8 rows remain lossless.

Both options are offered identically to all five selectors, with actual LOOK-M
merged K/V preserved. No weights, positions, mandatory tokens, source capacity,
history or generation changes. Reuse every original union3/scalar-refit/text-row
prediction (240 rows), preserving producer identities; add 160 new observations.
No losing point is removed. Complete 128-token requests, one warmup/one timed
trial, EOS-aware scoring, same pinned model/processor/inputs/environment. These
are rate/distortion points, NOT equal encoded-byte comparisons. Latency is
descriptive, not a powered repeated-trial claim; KV-only peak remains explicitly
limited to decode-owned storage, not whole-request HBM.

Require a complete clean CPU suite, identical-packet CPU/CUDA decoder controls,
ordinary generation/reinjection parity, per-worker union3 logit-hash controls,
isolated repeated requests, and complete provenance/accounting. First run four
images for EviCastKV under both arms. Promote only this bounded study after
execution controls pass. At most six aggregate A100s, 12 CPUs/one GPU/30 minutes
per worker; longest remaining selector first. Finite afterany analysis recomputes
corpus CIDEr in 10,000 paired replicates. Correct simultaneously over 48
comparisons (two EviCastKV arms against each of the other 24 compressed points).
Neither these consumed images nor the two numerical-guard inputs are independent
paper evidence. No fresh, calibration, larger or paper promotion is automatic.
