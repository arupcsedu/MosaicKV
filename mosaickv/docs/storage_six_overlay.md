# Building mechanisms on the successful storage execution path

The prior feasibility commit f3ef9fa descended from the successful storage
producer a6f689f, but executed the older dense HF cache path. Its results cannot
be described as an incremental extension of the winning storage representation.
This experiment instead calls the exact same `evicastkv_paper_precision.execute`
and `common_streaming_timing.request` used by the 256-image storage study. Frozen
reader options, 603-token preprocessing, checkpoint, BF16/eager math, original
prefill retirement, encoded source, protected exact rows, 128 predictions,
sequential reconstruction and lossless history remain active.

## Explicit new hybrid, not original six-component compaction

Forecasting, a sparse cross-modal graph, value utility and lazy submodular
selection choose four additional exact-protected rows in one routed head of
layer 31. This selection never removes the parent's exact protection or tokens.
The original prototype constructor creates one two-member image prototype by
inverting captured RoPE phases. The shared prototype is reconstructed at each
member's **original** position; it does not collapse the attention sequence.
Its actual CPU residuals permit a first-step entropy-guided promotion. This is
a **six-mechanism precision-overlay variant**, not a silent substitution for the
historical slot-compaction algorithm. No general six-component speed/quality
claim follows from this diagnostic.

Every added index, phase, key/value, phase-adjusted key, mixed buffer, promotion
reservation and device flag is charged. The bounded auxiliary GPU allowance
is 16 KiB. Original full K/V owners are not retained. CPU residuals are pinned
and reported separately. Repair writes original residuals into charged stable
addresses so previously captured graphs observe the new data. Mandatory values
and original logical sequence length remain unchanged.

The initial prototype used a full provisional trajectory because the parent's
engagement checks were hard-coded to 127 updates. The separately registered
two-prediction probe now requires exactly one update through every layer,
with all cold/capture/replay/copy counts checked. The default main-request
contract still requires all 127 updates; numerical operators and tolerances
are unchanged. A 128-prediction provisional control must reproduce all final
logits bitwise. The private probe reuses the same prefill/packets and discards
its history. Any promotion recomputes its first prediction once in the final
request. General arbitrary-step repair is not implemented by this extension.

Valid comparator rows from the first attempt are reused with their original
producer/manifest and hashes. Only missing or failed comparators are retried.
Per-worker scratch Triton caches avoid shared compiler-publication races.

## Finite verification

Two already consumed COCO inputs compare parent, disabled-hook parity, hybrid,
forced repair, and the four historical compressed comparator adaptations.
Zero activation must reproduce parent logits bitwise. Enabled arms must produce
real prototypes and pinned residuals, never exact-only fallback. The forced
control must transfer and promote real original K/V. The 32-token KL/logit/token
diagnostic uses unchanged 0.02/1.0 thresholds; divergent own-prefix comparisons
are identified, not mislabeled as fixed-prefix KL. Numerical repair is distinct
from uncertainty-driven residual repair.

This is not a fresh quality cohort or a byte-matched campaign. A finite two-GPU
array and afterany CPU analyzer preserve failures; no larger campaign is
automatically launched. The prior storage study and user paper remain untouched.
