# Move verified tiling ahead of lossless text-storage validation

The consumed subphase study identifies three mandatory independent
reconstructions as the real critical path: asynchronous exact flags reduce
host comparison time but shift the stream wait to the final reduction.
Preserve that finding; do not describe it as elimination of the GPU work.

The current source constructor fits transport patches using the original
one-head/16-row kernel, optimizes text storage and checks three independent
reconstructions, THEN switches to the four-head decode tiling. This bounded
successor moves only the verified hardware geometry ahead of text optimization,
AFTER unchanged transport fitting. It does not retune quantization, protections,
patch thresholds, weights or memberships, and removes no reconstruction/check.

Use only the geometry previously passing all-layer K/V and complete 128-token
checks; if no alternative passes, test the existing (16,4) decode geometry
earlier in construction. Require the same original packet hash and allocation
bytes for every method/input. A changed text patch rejects the whole candidate,
even if it appears numerically close. Audit constructor memory and compare
paired-projection spills against the original (16,1) construction kernel.
The declared memory boundary still excludes opaque GPU workspace and is not
whole-request peak memory. No new buffers are added to persistent packets.

Run only when the preceding complete-request timing gate remains negative.
Preserve default construction_geometry=None for historical calls. Apply the
existing shared exact-state CPU/GPU gates, then conditionally run the same five
methods, five warmups and twenty full requests per input. A finite afterany DAG
waits for row-geometry decisions and timing without polling or modifying active
snapshots. Stop on incomplete/unsafe upstream data or positive timing gates.
No fresh cohorts, calibration, paper promotion, quality-threshold changes,
universal-superiority conclusion, or performance claims about all six disabled
mechanisms follow from these two consumed source-precision inputs.
