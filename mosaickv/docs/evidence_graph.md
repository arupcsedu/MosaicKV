# Sparse cross-modal evidence graph

The graph implementation is backend-independent and operates on
`FullKVState` block descriptors. It does not prune or otherwise modify the KV
cache. The public entry point is `mosaickv.graph.build_evidence_graph`.

## Node descriptors

Each node is exactly one `KVBlockDescriptor`, so it belongs to one layer and
one KV head. Keys and values are pooled over the block's batch, head, and
physical-position axes using the layout recorded by `KVLayerStorage`. Optional
per-layer hidden states are pooled over the same positions. Each pooled source
is L2-normalized, concatenated, and normalized again to form the semantic
descriptor. Descriptor tensors are detached, copied to CPU float32, and made
read-only; graph construction does not retain accelerator storage or autograd
state.

`BlockEvidenceMetadata` carries optional graph-only annotations. Boxes must be
normalized `(left, top, right, bottom)` coordinates in `[0, 1]`. Image, frame,
and page identities default to the cache descriptor. Unnormalized regions in
legacy cache metadata are not silently interpreted as normalized boxes.
For Qwen2.5-VL, image/video token runs are validated against processor-owned
`image_grid_thw`/`video_grid_thw` and `spatial_merge_size`. Each block receives
its exact normalized raster box and image/frame identity. Grid/token-count
mismatches raise instead of generating guessed spatial evidence.

## Independent edge sources

The stored directed edge types are semantic similarity, prompt-attention
co-activation, spatial adjacency, OCR/layout, temporal adjacency, same evidence
region, cross-modal alignment, and positional fallback. Every source has its
own configuration weight. A weight of zero disables that source.

Compatibility is checked before storing every edge. By default, nodes must
share a layer and KV head, and the directed modality pair must appear in
`allowed_modality_pairs`. Semantic, attention, and fallback edges also support
source-specific maximum logical-position spans.

- Semantic and attention edges use positive cosine similarity. Raw prompt
  attention maps may be supplied as `[batch, query_heads, query_positions,
  key_positions]`; query heads are grouped by KV head and each block is pooled
  over batch, grouped query heads, and its key positions. Callers may instead
  provide already pooled per-node co-activation vectors, but cannot provide
  both forms in one build.
- Spatial edges use rectangle distance over normalized boxes within the same
  image/page container.
- Layout edges use OCR-box reading order, equal OCR strings, rows, columns, and
  named page regions.
- Temporal edges use frame distance within a clip and a configured frame
  window.
- Evidence-region and alignment edges use explicit stable identifiers;
  alignment edges require different modalities.
- Positional fallback is an independent sparse source for every node, including
  nodes that also have metadata. It cannot disappear merely because one
  structural source is present. Its configured weight and diagnostic counts
  remain separate from all real evidence types.

Each source independently retains at most `max_neighbors` outgoing neighbors
per node with deterministic target-ID tie-breaking. Consequently, disabling
one source cannot change the candidates or weights retained by another source.

## Sparse storage and memory bound

The result is typed COO storage and can be converted to CSR with `to_csr()`.
No dense adjacency matrix is created. With `s` enabled sources, `n` nodes, and
degree cap `d`, stored edges are bounded by `s * n * d`; `s` is at most eight,
so graph storage is `O(n*d)`. Similarity is evaluated as
`similarity_chunk_size`-by-candidate score tiles rather than an `n`-by-`n`
matrix. Metadata sources use spatial indices or bounded local/group
neighborhoods.

## Diagnostics

`GraphDiagnostics` records node and directed-edge counts, weakly connected
components, the fraction of stored edges that cross modalities, average
out-degree, maximum out-degree, counts by edge type, fallback use, and evidence
cluster coverage. Evidence coverage is the fraction of per-layer/per-head
named evidence regions whose induced graph is connected; singleton regions
count as covered, and the field is `None` when no evidence-region IDs exist.

These diagnostics are structural outputs, not measured model-quality results.
In the bounded v17 run, the full graph increased facility coverage over the
no-graph control by 0.8326 on DocVQA/ChartQA and 0.8114 on MMStar. Task scores
did not change at the one-/two-example resolution; the structured-task
attention-output diagnostic improved by 0.0008 on average while MMStar worsened
by 0.0651. This is directional exploratory evidence, not a structured-task
quality claim.
