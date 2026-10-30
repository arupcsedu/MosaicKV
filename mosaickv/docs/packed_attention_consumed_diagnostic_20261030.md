# Packed K/V attention feasibility — separate experimental variant

Source history preserved bits but cost about 30% more decode work in the
two consumed-input diagnostic. This successor tests lossless exponent-coded
K/V read directly inside a decode attention kernel: no source projection and
no materialized full K/V lease. It is not enabled by the normal CLI.

Stage one is synthetic CUDA numerical and latency feasibility only, with the
same raw/encoded kernel and original eager reference, BF16 batch-one full-MHA
dimension128. Exact codec reads must produce bitwise identical raw/encoded
kernel outputs. Arithmetic reduction ordering may differ from original eager;
that is measured, not silently called bitwise parity. Registered attention
output absolute-error control is 0.015625 on random fixtures. This does not
replace model-level controls: all original 32 greedy tokens, KL <=0.02,
maximum logit difference <=1.0, retention-one, positions, masks, state
isolation, mandatory membership and complete accounting remain required.

Five warmups, twenty randomized technical repetitions; all metadata counts
including literal checkpoints. A finite one-GPU20-minute job; scratch caches
only. No new evaluation identity or large model is needed for stage one.
Proceed only to the same two consumed inputs if numerical controls pass.
Offer identical kernels to Prompt Top-K, PrefixKV/VL-Cache local adaptations,
LOOK-M actual merged K/V and EviCastKV. Improvements against an unoptimized
baseline alone must not become a selection-policy superiority claim.

No current result is claimed. No published algorithm is reproduced by this
kernel, and no benchmark annotations enter format/proposal decisions. All
negative earlier selection and source-history results remain preserved.

## Preserved initial failures and stable-accumulation successor

V1 failed raw/encoded arithmetic equality at 300 positions; V2's common
tensor-core reduction passed that case but failed at 2048. V2's independent
reader matched all 65,536 BF16 patterns and each fixture's original K/V bits.
Its largest raw/encoded probability difference on the failing case was
4.76837158203125e-7; output difference was 0.0001220703125. This separates
floating reduction/layout differences from a codec corruption claim.

The separately labeled V3 kernel uses FP64 intermediates for its small dot
and weighted-sum reductions, then the declared BF16 eager boundaries. The
same equality gate and original eager error bound remain, and both controls
are rerun. This is not claimed to be original eager arithmetic. All original
model-level token/logit gates must still pass before any quality promotion.

V3 passed every numerical control, including raw/encoded arithmetic equality,
but not the all-length <=1.05 latency gate. The 300-position packet path was
faster; 604 and 2048 were slower. Preserve those results. V4 parallelizes the
long-sequence weighted-value reduction into fixed128-position partitions,
then combines FP64 partials. Its workspace is attention scratch, not a hidden
persistent K/V allocation, and whole-request peak remains uncertified. Both
raw/encoded numerical controls and the unchanged original eager fixture bound
must rerun. No model-level correctness or quality result is inferred.

The bounded model canary is two consumed inputs only. It records 32-token
retention-one greedy agreement and maximum logit difference <=0.001; selected
cache runs must retain all greedy tokens and KL<=0.02/logit<=1.0 against the
ordinary implementation of the identical selection. A real ordinary HF cache
is the control: its append allocation overlap is instrumented at actual cat
operations. Encoded headers/checkpoints and raw history are all charged.
Model attention masks are checked asynchronously once per distinct mask/step,
never silently ignored; dispatch is request-local via ContextVar. A six-path
128-token technical comparison is permitted only after both canaries pass;
it does not authorize fresh evaluation or any paper campaign. Timing includes
separate constructor and decode phases and their sum, excluding prefill and
the frozen original selector, identical for every compared path.
