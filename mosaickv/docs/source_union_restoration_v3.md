# V3: restore evidence already present in a shared projection archive

V2 array 22183915 and analysis 22183916 passed all real-model 32-token
representation controls. Retention one also passed. The workspace audit makes
the limitation visible: source recomputation saves persistent storage but
adds temporary backing and computation. No task-quality superiority follows.

## Why this intervention, not another donor swap

Earlier delimiter/donor controls failed. A head may have discarded a token
whose original normalized input is nevertheless already archived because a
different head retained it. For each layer, form the union U of the frozen
per-head selected positions, and let every head attend to U. This removes no
old selected position and requires no donor eviction, special delimiter quota,
new prompt, future query or benchmark answer.

The normalized source rows and their BF16 bits are unchanged. Reconstruct
original K/V with the original projections and RoPE. When all heads share U,
one GPU union-index allocation can back a stride-zero head view instead of H
duplicate maps. Charge actual backing. Larger layer leases and reconstruction
workspace can still offset persistent savings: do NOT call this physically
free or a matched-byte comparison.

The pure union rule sees only saved position sets. Actual future-prefix logits
are evaluation-only and never enter the online-eligible expansion or format
selection. Apply the same union rule and source format to BOTH selectors.
Preserve original selections' KV-only/hybrid/bounded-source controls.

## Fixed consumed-input test

Bind complete V2 and reuse exactly the same two consumed identities, checkpoint,
BF16 precision, processor/media/prompt, eager backend and seed. Verify the new
shared-map implementation through retention-one and original-selection
32-token own-greedy controls first.

For union restoration, independently gather original post-RoPE K/V at the
expanded positions and compare all 32 forced-prefix logits bitwise to the
source-backed path. Targets are the previously saved FullKV model-generated
prefix, not benchmark annotations. Source cache masks cover the maximum
physical layer extent even when layer zero is not longest; logical/RoPE
positions remain original.

Compare candidate prefix logits to the original anchor and FullKV using the
UNCHANGED 32-token guard, KL/logit thresholds and token conditions. Preserve
all prefix failures. Report per-selector outcomes and measured
initial/final/decode-owned tensor-backing peak, including extra workspace.
Do not compare distinct own-greedy trajectories as if they shared a prefix.

No benchmark score, independent quality claim, fixed-byte superiority, fresh
cohort, calibration or paper-scale campaign is authorized. A positive causal
diagnostic would justify a separately frozen development validation, not a
headline. Negative results stop this simple expansion, without enlarging
quotas or weakening the guard. This MHA-specific source format is not claimed
to help grouped-query architectures.

Use two one-A100/12-CPU workers, ten-minute limits, maximum six aggregate A100s,
scratch-only common environment and finite CPU afterany analysis.
