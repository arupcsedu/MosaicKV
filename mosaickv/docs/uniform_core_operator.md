# Uniform-only eager-core capability: retain the negative universal gate

The previous universal operator passed all six word/ownership controls and
384-KiB checks, but FAILED its registered two-layout performance gate. Uniform
replay took about 5.31 ms versus 8.84 ms; adaptive replay and reusable-out took
about 11.08 ms versus 8.86 ms. Preserve that failed universal result. Do not call
the narrower study a pass of it, or claim adaptive acceleration.

This separate capability applies only to verified 32-layer uniform prompt
geometry, within the same 1,024-position physical bound. It is selected once
from layer lengths and the registered horizon, never from answers, task names
or method names. Uniform Prompt Top-K/LOOK-M must receive it too. For adaptive
or unverified geometry, do NOT allocate a Core, copy queries, use the slower
out variant, or capture anything: use the exact original eager implementation.
Report this ineligibility and zero GPU owners/graphs truthfully.

Cache only immutable CPU view descriptors per live length, aliasing the same
fixed owners. Keep an uncached uniform-graph ablation beside the frozen cached
candidate, so geometry routing and descriptor reuse are not conflated. The
candidate uses caching before outcomes are inspected; it is not the post-hoc
best of the two. Both adaptive policies instantiate no Core and preserve the
original attention path. Clear descriptor aliases only after graph retirement.

Keep every prior numerical, input-isolation, retirement, memory and 16-use rule.
Run the same real-CUDA controls before timing. Repeat the controls at
301/305/365/369/427/431 positions too, so the shorter
uniform baseline layout is not assumed to inherit the Source-length proof.
Run five warmups/twenty randomized paired repetitions on BOTH layouts,
including cold capture/copies/retirement.
The registered feature gate requires <=0.96 latency for supported uniform
geometry plus exact original adaptive passthrough with zero graphs. Adaptive
timing is informational, not a claim of speedup or a relaxed pass of the old
universal gate. No model/cache compression, data, or metric changed.

Count the fixed tensor workspace, capture-time transient allocation, generator
metadata and separately exposed two-MiB private pool reservation where present.
Common cuBLAS backend warm allocation remains explicitly exposed. A future model
integration must charge the actual new footprint and preserve smaller measured
initial/final/decode-owned peak storage; operator feasibility cannot certify it.
Model numerical/full-request validation remains separate. No quality campaign,
fresh identities, broad latency claim, or paper-ready conclusion follows this
synthetic study. Independent benchmark-quality N=0.
