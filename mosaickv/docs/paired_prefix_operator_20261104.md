# Paired K/V prefix operator: bounded feasibility, not a superiority result

The original-row-major nibble scratch failed its preregistered operator gate
in `row4-operator-20261104-retry1`: bitwise controls passed, but measured paired
ratios were 1.1592 and 1.1664. Preserve it as negative; do not enable it.

This successor keeps the original three-bit code layout and all positions.
For a fixed 256-row prefix, one CTA shares source decoding across K and V.
K writes to its original lease; V writes to a separately charged contiguous
2 MiB prefix buffer. K and V tails use their original global row/RoPE positions;
the V prefix is copied to the existing V lease. No weights, precision,
mandatory tokens, corrections, or attention operations change. Source patches
and protected values remain outside this primitive and would still need model
parity before any integration. Joint projection uses sixteen warps to stay
within A100 shared-memory limits. Unknown/nonzero spills fail closed.

The finite CPU -> two one-A100 operator workers -> CPU analysis chain tests
600/604 rows, 32 heads, dimension 128. Five warmups and twenty randomized paired
trials include the prefix copy. All prompt/output/history-sentinel bits must
match the original two projections. The common binary-arm reducer retains its
historical `row4` key, explicitly labeled as a key, not a layout here. The
registered adjusted upper wall-ratio gate is 0.80 (10,000 bootstrap replicates).
An operator pass is not request latency, answer quality, or paper evidence.
There is no automatic model/cohort submission.

The same source also adds a default-off private graph execution policy:
capture lossless ordered code reconstruction, and borrow the idle graph lease
for history verification. Borrowed scratch is poisoned before all three
independent verification fills; persistent-history aliasing is rejected. The
ordinary reader also releases its previous verification output before its next
fill, giving baselines the same common lifetime improvement. Graph generator
owners remain charged and verified. These changes require real CUDA and model
controls; CPU stand-ins alone do not establish numerical equivalence.

Full-request acceptance remains adjusted paired ratio <=0.96 against every
original AND optimized baseline option. Retention-one/parent packet/logit
checks and initial/final/decode-owned peak storage gates remain unchanged.
The broader six-mechanism algorithm is disabled in this precision/storage
variant. Independent quality N=0, and whole-request ownership peak is not
certified. Fresh cohorts and paper campaigns remain blocked.
