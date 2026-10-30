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
