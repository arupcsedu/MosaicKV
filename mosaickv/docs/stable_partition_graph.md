# Exact stable top-d cosine edges without sorting every target

After indexing positional neighbors, cosine-neighbor selection remains a
dominant CPU operation. On the existing segmented path, use a partition to
find the d-th score cutoff. Keep all scores above it; resolve equal-cutoff
scores by prefix rank in original sorted node-ID order. Exclude the source
and nonpositive scores exactly as before. Do not use argpartition's arbitrary
tie order. COO sorting, edge weights, compatibility groups, span filters,
facility coverage and the selected cache remain unchanged.

The stable full-sort implementation remains the default/reference. The new
explicit `graph_cosine_topk=partition` mode requires segmented accumulation.
Randomized tests compare exact positive top-d sets for many tied score tiles,
single-node groups, oversized d and self exclusions. Complete graph tests
compare every edge/type/weight/diagnostic. A pretrained cross-source gate must
also match the frozen graph COO hashes and every one of the 128 logits.

The paired benchmark takes its two policy configurations FROM the completed
graph gate and rejects any differences except the three registered execution
fields. It cannot silently change a budget, prototype, repair threshold or
selection policy. Keep the same five warmups, twenty pairs, storage matching,
bitwise predictions and adjusted latency criteria. This optimization is not
new independent quality evidence or permission for a larger campaign.
