# Generated-history source packets: prospective feasibility only

This bounded diagnostic addresses append-only history growth without evicting
any history or changing answers. It uses the same two consumed inputs and six
baseline paths already tested. It does not integrate an online codec, claim
reduced whole-request memory, or authorize fresh evaluation.

The pinned LLaVA full-MHA architecture projects one normalized hidden vector
into K and V. Read-only attention hooks capture that actual input, its original
RoPE phase and the actual post-RoPE K/V written to each history tail. The
unmodified 128-token transport runs before capture; all logits and the existing
32-token prefix must remain bitwise identical. Hooks are removed on failure.

For each sixteen-token sealed chunk, build the existing exact source projection
plus sparse original-bit corrections and check K/V three times. Preserve the
actual phases; history-local addressing must not renumber RoPE. Compare actual
packet buffers with the independent per-chunk lossless K/V packet alternative.
Account for source maps, slots, checkpoints, patches and shared phases once.

Critically, the prospective source design also requires a mutable normalized
source tail. Charge sixteen BF16 hidden vectors per layer **in addition to**
the unchanged sixteen-position K/V tail. Omitting this overhead would exaggerate
saving, especially for 32-token outputs. Report both 31 and 127 updates, exactly
matching the first-prediction-plus-decode convention.

The feasibility threshold is at least 10% packet saving at 127 updates for every
method and both inputs. Failure stops this direction. Success only justifies
later online integration: it does not certify transient/constructor peak,
latency, batched-pool byte equality, or a benchmark-quality improvement.

Execution uses two one-A100 workers, twelve CPUs each, thirty-minute limits,
and a finite afterany CPU analyzer. Each method/horizon has a 300-second
construction limit. No predictive fitting, quantization, model-weight changes,
history eviction or benchmark-label access occurs.
