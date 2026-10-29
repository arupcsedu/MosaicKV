# Shared compact exact-tier representation

EviCastKV is the paper name for MosaicKV. This is an execution-only optimization
shared by EviCastKV and Prompt Top-K, not a new selection policy or quality claim.
Fresh evaluation and larger campaigns remain blocked.

## Motivation and representation

The October 29 consumed-input diagnostic identified eager creation and repeated
validation of one tiny K and V tensor view per selected block as the largest
exact-tier construction cost. Exact-only HF packing already gathers from the
original source tensors and selected descriptors; it does not consume those
individual view objects.

`source_views` now stores the unchanged ordered descriptor tuple, one frozen
source alias per layer and K/V tensor, validated geometry and payload-byte
metadata, and immutable sequence facades. Indexing or iterating the sequences
creates ordinary block views on demand. Repeated indexing returns the same
cached object. Construction and exact-only HF packing need no block views.
Concrete PyTorch tensors and NumPy arrays use this path; subclasses and custom
tensor protocols retain the eager, individually validated implementation.

`source_views_reference` explicitly retains the old eager representation for
diagnostic comparisons. Other gather modes and the default configuration are
unchanged. Both modes preserve source alias semantics; neither copies or changes
selected K/V values. Selection, descriptor ordering, logical positions, mandatory
policy, physical capacity, masks, and the HF packing algorithm are unchanged.

## Validation and accounting

The compact bank validates descriptor provenance, source bounds, contiguous
physical positions, dtype/device, and descriptor bytes against source geometry.
The existing membership index still rejects duplicates. MosaicKVState retains
source-subset, cross-tier conflict, mandatory-position, and storage-statistic
checks. Materialized views are cached, revalidated, and included in cross-tier
tensor-object identity checks. General public ExactTier construction continues
to validate each supplied tensor. A frozen layer alias preserves an existing
tier's storage binding if an external caller later rebinds a source tensor.

Selected view-payload bytes are **not** the total backing allocation retained by
source aliases. This optimization reduces host descriptors/view objects, not
the selected KV budget. The HF runtime still materializes its cache and records
actual backing storage, including dense padding and generated-history growth.

CPU regressions cover randomized Torch/NumPy tensors, mixed dtypes and widths,
noncontiguous sources, arbitrary block sizes, alternate tensor axes, logical
gaps, retention-1 reconstruction, mandatory preservation, duplicate/foreign
descriptors, view mutation, source rebinding, alias conflicts, and identical
packed tensors, masks, slots, and allocated bytes.

## Bounded diagnostic procedure

Use `evicastkv_consumed_instrumentation.py prepare --compact-source-views` in a
clean immutable checkout, then its existing submit command. It compares the
reference and compact representations on exactly the same already-consumed
Description and COCO inputs as the prior instrumentation diagnostic. It does
not materialize new data or score against benchmark references.

Both sides use synchronized profiling, unchanged controls, complete generated
tokens/logits, selected-block/slot catalogs, initial/final/peak physical storage,
and FullKV retention-1 generate/reinjection checks. Every request stores its
actual execution config/hash plus the unchanged historical configuration hash.
The finite CPU `afterany` callback preserves failures and writes the equivalence
gate, view counts, phase timings, and artifact hashes. Single-pair profiled
timings are diagnostic only, not production-latency or statistical evidence.
