# Fused transport-predicate operator gate

The completed main/background attribution identifies foreground correction
construction as a remaining Source-specific cost. Host nonzero intervals
include queued GPU work and synchronization: they are not pure compaction
kernel time. Earlier static and atomic compaction attempts were negative and
remain rejected. Do not repeat or silently enable either.

This separately versioned operator retains PyTorch's canonical dynamic nonzero,
index order, int32 indices and original BF16 word copies. Test only replacing
the existing full-span FP32 closeness mask and actual-value finiteness scan
with one bounded Triton kernel. Keep rtol=0.01, atol=2^-6, equal_nan=False;
disable FMA contraction. Handle same-sign infinities, signed zero, NaN,
subnormals and partial blocks explicitly. No BF16 assertion predicate or
protected-position, transport-envelope, native-repair or logit guard changes.

Before timing, require exact mask/finite flags on every BF16 bit pattern,
multiple pairings and threshold cases, plus partial blocks and the 4-Mi-element
bound. Canonical correction indices and word patterns must match too. Any
mismatch is a negative numerical gate; do not weaken tolerances or time it.

Then use five warmups and twenty randomized paired repetitions for the complete
mask+finite+nonzero+word-copy stage at 600/604 source tokens, alone and after a
synthetic queued native projection. Include synchronized host wall and CUDA
event times, every trial and all scratch. Operator progression requires no
regression at either shape/scenario and at least one ratio <=0.96. This does not
establish request speedup. No model weights or datasets are accessed, independent
quality N=0. One A100 follows a full same-SHA CPU suite, then afterany analysis.
No automatic inference integration, fresh cohort or quality campaign is allowed
by this operator result. Runtime/packets remain entirely unchanged.
