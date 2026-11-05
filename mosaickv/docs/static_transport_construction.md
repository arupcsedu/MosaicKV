# Bounded constructor compaction, with canonical output unchanged

The optimized consumed constructor profile attributes approximately 0.11 s to
64 dynamic transport-correction nonzero operations. This is separate from
steady-state history-patch dispatch. Do not sum nested cumulative timings or
promise that this alone closes the remaining latency gap.

The opt-in constructor uses installed Torch 2.8's fixed-size nonzero operation:
4,096 sorted slots per component, plus an on-device true count. Temporary
masked writes preserve every original correction while independent prefill
verification and text-overlay construction run. All original finite,
protected-row and close predicates remain. A single count transfer before
commit checks all component capacities; overflow FAILS, never truncates or
silently relaxes a check. Temporary numeric data are not benchmark references.

Before payload hashing, budgeting, or decode, copy the original compact int32
indices/BF16 values into ordinary SparseBitPatch objects. Original packets may
retain provisional tensors until prefill retirement; verify their weak owner
references all die after that retirement/GC and before decoding. Final and
initial decode storage therefore cannot benefit by ignoring a staging buffer.
Record staging backing and count-transfer bytes separately. Whole-request or
construction peak is NOT certified by these counters.

CPU regressions cover empty/exact-capacity/overflow cases, original sparse
ordering/value words, protected mismatches, wrappers, mandatory positions,
foreign/dropped patches, scopes and retirement. CUDA fixtures compare actual
masked writes and canonical output with ordinary nonzero at 0/1/127 padding,
including an expected rejected overflow. Check all 128 own-parent model-logit
vectors, packet hash, no fallback and identical initial/final/decode-owned peak
backing on the same two consumed inputs. No new benchmark labels or samples.

Every arm receives the already separately tested compiled history-patch reader
and event-sum accounting. Compare original and static construction for EviCastKV
and both PrefixKV readers; baseline methods with no eligible transport patches
must report zero eligibility honestly. Failed lookahead and native conversion
are excluded, not relabeled successes. Five warmups/three randomized requests
are a SCREEN, not the adjusted twenty-repeat <=0.96 acceptance gate.

Finite full CPU -> two one-A100 workers -> terminal CPU analysis. Depend on the
preceding completed dispatch screen, refuse a short daily-reservation window,
one GPU/12 CPUs/48 GiB/twenty minutes per worker, at most six aggregate A100s.
No larger/fresh campaign follows. These six-mechanism-disabled precision/storage
variants do not establish independent quality, benchmark fidelity or a paper
superiority claim; independent-quality N=0.
