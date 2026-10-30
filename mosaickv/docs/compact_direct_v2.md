# Compact direct-buffer transport V2

Frozen memberships and original bits are unchanged. This follows the completed
V1 compact proof: replay-only FP32 accumulation eliminated all BF16 correction
entries for the two consumed EviCastKV inputs. It did not establish quality
superiority or useful latency. Do not change model/prefill/decode precision.

V2 retains the compact projections but writes original-position staged-BF16
RoPE/gather directly to the attention lease. Release projected K before
projecting V. Replace repeated searchsorted with a charged int32 original-token
to compact-row map. Replace flattened int64 direct-slot indices with two int32
coordinates, at the same payload size. Decode low-occupancy original K/V
straight into the lease using integer-only scatter, without decoded-K/V or
index-arithmetic temporaries. Correct any remaining numerical differences with
the existing lossless delta stream. All formats remain available to all exact
selectors; transformed LOOK-M values remain ineligible for original projection.

Require synthetic CUDA strided-window/sentinel tests for original-position
RoPE, V gather, raw scatter and encoded scatter, then per-layer original-bit,
retention-1 and 32-token greedy equality. Only after those pass, run 5 warmups
and 20 randomized paired repetitions. Timed cache instances have independent
history. Memory evidence is separately instrumented. Charge the extra int32
source maps and all backing/workspace. Whole-request peak remains uncertified.

The first two phases preserve memberships and test numerical correctness and
repeated timing. A separately labeled third phase tests restore-only additions
on the same consumed identities after V1's bulk additions failed. Reuse the
original prefill for four isolated FullKV-generated draft steps. Rank omitted
tokens independently per head by signed, normalized attention-output error
reduction, accounting for renormalization and preserving all draft history.
Generate at most eight positive-gain proposals, each adding exactly one slot
per head in only one layer. No donor or original evidence is removed. The
proposal API never receives annotation answers or 32-prefix evaluation logits.
The primary prospective policy is the top-ranked signed-gain proposal. Any
other plan passing the evaluation guard is diagnostic oracle information, not
permission to choose it using future evaluation logits.
Retain full layer/head score telemetry to detect route-index/tie defects.

Evaluate all proposals with unchanged 32-token KL/logit/token guards, initial
<=62 MiB and declared decode-owned peak <=96 MiB, plus independent own-greedy
selected-cache bitwise controls. A diagnostic success is not a benchmark score
or fresh confirmation. No new cohort, calibration or paper campaign is
authorized. The V1 paper-baseline diagnostics continue unchanged.
