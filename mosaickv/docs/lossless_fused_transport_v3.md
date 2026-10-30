# V3: bounded shared-transport optimization

The completed V2 diagnostic at source `0aa7cb8` verified bitwise 32-token
greedy/logit parity for the two consumed Description/MM-Vet inputs. It reduced
post-construction decode-owned peak storage by approximately 25%, but decode
was approximately 6–7% slower in single diagnostic observations. Both exact
selectors benefited almost equally. Neither selector superiority nor
whole-request peak certification was established. Preserve those artifacts.

V3 changes **only execution of the same lossless representation**. One integer
GPU launch unpacks K and V and copies their used append-only history suffixes.
The previous path dispatched two unpack and two history-copy kernels per layer.
The new path does not change selected positions, stored BF16 bits, logical
positions, mandatory tokens, capacity, attention arithmetic, or model weights.
Apply it equally to raw and encoded transports and to both selectors.

Validate every BF16 bit pattern and partial blocks, all four combinations of
raw/encoded K/V, and history lengths 1/16/31. Require retention-one and selected
cache greedy parity for 32 tokens. Keep previous absolute FullKV-prefix guard
failures visible; representation parity is not a selection-fidelity pass.

For each consumed input and saved selector, benchmark raw/separate,
raw/fused, encoded/separate, and encoded/fused. Preregister five warmup blocks
and twenty timed blocks, randomized order within every block, seed 20261029.
Construct a fresh cache every trial. Transfer logit tensors to CPU only after
the synchronized decode timing boundary; require all trial logits/tokens to
match the ordinary selected-cache reference bitwise. Store individual trials.
Report median/p95, constructor time, and a 10,000-resample paired timing-ratio
interval. The representation-overhead engineering gate is an upper interval
bound at most 1.05 against the raw version of the **same** transport. This
is not a benchmark-quality, selector, or end-to-end speedup gate.

Storage remains unique tensor backing including codec metadata, raw histories,
growth overlap, and the current layer's complete materialized K/V. Separate
and fused executions must have identical accounting at all three boundaries.
The inactive source, constructor workspaces, weights and activations remain
outside this post-construction boundary. Whole-request KV/HBM peak is
uncertified; no claimed whole-request memory reduction follows.

Use two one-A100/12-CPU workers from one clean immutable source, 20-minute
limits, scratch-only caches and a per-worker compiler cache. Chain finite CPU
analysis with `afterany`; preserve producer failures. No fresh identities,
calibration, large campaign, or paper claim is authorized by this protocol.
