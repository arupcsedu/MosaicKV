# Exact structural-edge compilation

The remaining graph profile includes repeated spatial grid search, OCR/layout,
temporal/alignment grouping and fallback-neighbor work. These edges depend only
on validated structural metadata and positions, not K/V features or query
demand. The opt-in engine-scoped template cache compiles exactly those edges
once per complete structural signature. Semantic/coactivation edges, pooling,
forecasting, utility, selection, prototypes and repair still execute per request.

The SHA-256 key contains the complete graph configuration, execution policy,
effective eligible-node set, ordered node identities, original logical
positions, modalities, layer/head identities, and every structural metadata
field. It does not contain or retain a query, K/V tensor, model owner, prediction
or reference. Changed geometry, source layout, grouping, budget-induced
eligibility or compatibility rules cannot reuse a mismatched template.

Every cached segment is independently validated, immutable and retains its
original weighted score. It is not re-clipped or weighted twice. The final
COO sort and diagnostics are unchanged. Source-type degree caps and independent
edge ablation remain intact. Public source-state/metadata validation is not
skipped. The LRU accounts for CPU arrays and shallow retained containers within
a four-MiB budget; this is not a whole-CPU-heap peak certificate. It allocates
no GPU storage. Closing an engine scope drops all entries.

CPU tests compare complete graphs across different K/V inputs with identical
structure, and verify invalidation for positions, modality, metadata, config
and effective eligibility; dynamic edges cannot enter the cache. Failed,
oversized or malformed entries do not bypass any storage guard.

Model timing remains blocked until the native-readout handoff has complete
pretrained parity/storage evidence. The four-arm comparison then isolates
template reuse from an otherwise identical handoff hybrid, retaining both
storage-only controls. Cold/profile work and compiled CPU backing are reported
separately. Warmed request results must be labeled geometry-template reuse,
not cold-start speedups, independent task-quality evidence or all-six superiority.
