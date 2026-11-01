# Diagnose the source-projection operator before changing numerical controls

The Description shared-source worker failed construction: one of 2,457,600
scalars differed by 0.03125 from native BF16 CUBLAS projection, outside the
registered rtol=.01/atol=2^-6 tolerance. Do not raise that tolerance. The MM-Vet
worker completed, including bitwise protected rows, retirement and isolation;
both source arms passed its original 32-prefix model guard. This is not an
overall pass: Description is incomplete, source peak storage was higher and
decoding slower. Preserve that result.

Run one operator-only one-A100 study on the two consumed normalized-input
captures, frozen projection weights and original RoPE phases. No model output,
annotation, answer, reference, target token or score enters this study. For
all 32 layers and both source-range choices, compare unchanged fused output
to native BF16, BF16 without reduced-precision accumulation, and FP32 input/
weight projection rounded back to BF16 before the original RoPE. When any
comparison fails, add FP64 projection as a numerical oracle. All diagnostic
backend flags are restored after each projection; production execution does
not change. Protect the same original rows in every comparison. Preserve
per-component mismatch counts and example coordinates. Do not stop at the
first divergent layer or hide later failures.

The output only diagnoses the operator. It cannot approve a new reference,
change tolerances, or promote any method/cohort. A corrected runtime must
separately pass the unchanged model guard, memory and actual decode tests.
Full CPU suite first; one A100, 12 CPUs, 30 minutes; finite afterany CPU
analysis. No fresh identities or larger campaign.

The first operator worker stopped before any numerical case: checkpoint files
use FP16, while the registered runtime explicitly loads BF16. The audit loader
must perform the same BF16 cast; preserve the rejected mismatched-dtype run.
A regression covers FP16-on-disk to BF16-runtime conversion. This correction
changes neither the representation nor any numerical threshold.
