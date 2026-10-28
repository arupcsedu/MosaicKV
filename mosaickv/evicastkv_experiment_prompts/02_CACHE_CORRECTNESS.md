# Prompt 2 — Validate packed-cache correctness and physical allocation

Apply 00_MASTER.md. Require Prompt 1's manifests. Implement and execute E0 cache tests on synthetic tensors, then on the 32 validation inputs per workload. Retain all diagnostic errors, including failures.

A. Build an independent attention reference that does not reuse the production packer or mask builder. For identical source K/V, queries, retained IDs and original positions, compare (i) the packed selected entries with validity masks and (ii) original-layout entries with excluded positions masked. The normalization must be over exactly the same retained valid keys. Do not compare a lossy subset with unrestricted FullKV.

B. Test nonconsecutive positions, unequal head lengths, GQA ownership where applicable, all-valid/all-retained cases, mandatory-only feasible selection, infeasible mandatory state, duplicate-ID rejection, tail/padding handling, dtype/device errors, and aliasing. Use small float64 cases against direct calculations (proposed tolerances atol=1e-10, rtol=1e-8); record these as a new test contract rather than a historical claim.

C. Check retained K/V bitwise equality to the selected source entries, ascending original-position order, masks, logical lengths, and the next cache position. Test rejection without partial publication. Test shortlists too small for feasible filling and ensure no silent capacity underfill or FullKV fallback.

D. Explicitly test the padded two-head cases (5,5) and (9,1): ten selected head-position entries must occupy ten versus eighteen padded head-position slots. Include K and V, dtype width, dimensions and batch size in byte calculations. Check actual backing allocations, alias deduplication, views retaining larger bases, and preallocation. Never certify memory using numel alone.

E. On real checkpoints, compare ordinary FullKV generation with extraction/reinjection at retention one for at least 16 tokens. Preserve the paper's existing 1e-4 maximum absolute logit criterion; report any failure without silently relaxing it. For irregular-layout numerical equivalence, record existing dtype-specific tolerances or establish a separate tolerance contract on independent numerical controls before examining quality/performance. A failure must be diagnosed, not waived.

F. For compressed layout checks, force identical continuation token IDs in both reference and packed executions for at least 16 steps, comparing outputs at every step. This prevents autoregressive divergence from obscuring a representation bug. Compare free-running tokens separately; do not count first-prefill-token agreement as evidence about compression. Track actual cache append count: an emitted token has no KV entry until it is processed by a forward step.

G. Exercise both checkpoints and the tight/moderate declared budgets. Debug snapshots belong in separate diagnostic runs, not the performance process: lingering full-cache references can alter measured memory.

Deliver tests, correctness_cases.jsonl, capacity_cases.jsonl, numerical_error_summary.json, correctness_report.md and GATE_correctness.json. Gate passes only for the tested valid paths with all structural/storage checks satisfied and declared numerical tests passing. A failed or unavailable test blocks dependent claims and runs.
