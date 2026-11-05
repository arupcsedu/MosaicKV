# Construct the verified joint plan, not its numerical search

The saved joint-KV construction profile attributes 28.8 ms to its two native
engine factories, including 27.5 ms in `KernelPlans` and 14.7 ms verifying the
same immutable installed header. These are instrumented CPU attributions, not
a promised request speedup. Runtime still enumerates up to 128 IDs and 32
candidate configurations even though the all-layer gate froze one previously.

Instantiate that exact algorithm/configuration (31, 15, 1, 0, 0, 0, 18, 4)
directly. Check its actual attributes, numerical flags and zero-workspace
feasibility on every new private handle. Keep the original vendor choices for
ordinary full/tail rows. Construct the one necessary descriptor set, not an
unused original set followed by another set. Cache only installed-header proof,
consistent with the existing immutable library ABI cache: never cache handles,
CUDA tensors, weights, cache data, requests or outputs. Other geometries fail
explicitly; only the two all-layer-audited tail shapes are enabled.

Before checkpoint loading, compare old/new actual GEMM output words at full,
tail and joint row counts for both geometries; verify actual configuration
attributes, not opaque descriptor bytes. Then require all 128 parent vocabulary
vectors bitwise and identical own-control initial/final/decode-owned peak bytes.
Exactly two constructors per Source request engage; ineligible baseline requests
report zero, with no artificial added work. Common epilogue/rotary optimizations
apply identically to all methods when their prior numerical/storage and paired
own-control timing checks passed. Retain source-copy reuse only if it passed
both consumed-input own controls; this choice never uses benchmark labels.

Four arms, five warmups, three randomized paired trials, two consumed inputs,
two one-A100 workers with 12 CPUs and 20-minute limits; full CPU suite and
afterany analysis. This small constructor improvement cannot by itself promise
the 4% full-request gate. Independent quality N=0, six-mechanism-disabled
storage variant, whole-request GPU peak uncertified, no fresh/larger campaign.

Before the randomized warmups and timings, execute one excluded extra request for each arm.
Capture balanced construction/decode CPU profiles for the frozen Source reader
and optimized PrefixKV graph reader; the other arms execute the same extra
request without profiling. These instrumented calls are never acceptance trials
or substitutes for GPU kernel timing. Use them only to locate the remaining
dominant operations under the common epilogue and rotary policies.
