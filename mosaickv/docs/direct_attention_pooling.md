# Bounded-memory attention co-activation pooling

The old Torch path first selected a full query-by-key matrix for EVERY block
and then gathered only that block's key positions. For one-token blocks, the
intermediate had shape `[blocks, query_window, all_keys]`: quadratic in prompt
length even though the final graph was sparse.

Gather the paired head/query/physical-key coordinates directly. The selected
tensor now has shape `[blocks, query_window, block_width]`; keep the original
mean reduction and deferred CPU transfer unchanged. Validate negative and
out-of-range coordinates explicitly so advanced indexing cannot wrap them.
CPU tests require bitwise agreement with the original repeat-then-gather
operator for multiple heads and widths. A dispatch-level allocation regression
rejects the old quadratic intermediate at the 603-position geometry.

This reduces a construction workspace, NOT persistent active KV storage. Do
not add those temporary bytes to a claimed cache compression improvement.
The pretrained gate binds the prior source's actual COO hashes and full
128-prediction logits; comparing two new paths alone would not be sufficient.
Keep all source fields, selections, prototypes, logical positions and guards
unchanged. Queue verification after an active timing array to avoid changing
our own GPU/CPU concurrency during that paired measurement. No new cohort,
independent quality evidence or larger campaign is authorized.
