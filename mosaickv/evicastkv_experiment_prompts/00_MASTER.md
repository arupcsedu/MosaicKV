# Master instruction — EviCastKV experimental validation

Use this instruction before every numbered task. Work in the existing EviCastKV/MosaicKV repository, not a new reimplementation. Read existing AGENTS.md guidance first. Names below designate NEW proposed output artifacts unless repository discovery identifies an existing equivalent.

## Objective and scope
Produce measured, reproducible evidence for the EviCastKV paper and the corrections in Prof. Fox's October 24, 2026 assessment. The writing deadline is October 29, 2026, America/New_York. Do not assume that earlier scheduled work was completed. Discover the real code, configuration, data and allocated hardware before running anything.

Keep the evaluated exact-selection path: prompt observations, unchanged retained KV entries, original logical positions, and append-only generation. Do not implement draft generation, prototypes, repair, pair-synergy rewards, new serving backends, or a new batching scheduler for this campaign.

## Scientific and implementation contract
- Preserve existing work and historical artifacts. Use a separate branch/worktree or an isolated change set; never reset or clean the user's work. Record source commit AND a hash of relevant uncommitted changes. Do not upload private code, data, logs, credentials, or model files.
- Recover actual coefficients, routing, adapter/model IDs, rounding, input preprocessing, EOS/padding, scorer and physical layout from code/manifests. Unknown fields are blockers, not permission to guess. Do not upgrade PyTorch/Transformers or change precision/backend silently.
- Never replace layout-aware capacity reconciliation with an unconstrained global top-K. Count unique active KV backing allocations, padding and preallocation; preserve mandatory positions and per-head minima. Independently normalized head scores are not automatically globally comparable.
- Separate representation correctness from task quality. Compare packed subsets to an equivalent masked-source reference, not to unrestricted FullKV outputs.
- Distinguish (a) matched-capacity compressed-method comparisons, (b) uncompressed FullKV references, (c) scoring/boundary diagnostics, and (d) native-baseline fidelity checks. Do not mix their rows or make one answer another's question.
- Reuse common runtime/packing optimizations in Top-K and EviCastKV. Each method should execute only its needed policy work. FullKV must not execute compression or expensive unused attention collection.
- A single job owns its allocated GPU. Follow existing scheduler and account restrictions; never run GPU workloads on login nodes or interfere with other users' jobs. Verify allocation rather than assuming eight GPUs are available.
- Never fabricate outputs, latency, uncertainty, completed tests or job completion. Keep failures, OOM stage, attempted/completed/scored denominators and all prespecified comparisons. A skipped optional task is not a pass.
- Development and confirmation samples must be disjoint by dataset ID and reusable media/document/video grouping where relevant. Never select samples based on FullKV correctness, method disagreement or favorable scores. Confirmation labels/scores are not available to model selection.
- Performance runs must recompute their own complete request. Shared prefills/cached graphs are permissible only in labeled diagnostic or selector-only studies, not end-to-end headline timing.
- Preserve negative and inconclusive results. No numerical performance improvement is promised. Do not equate a confidence interval crossing zero with equivalence.

## Reporting and stage gates
At the end of each task, write: changed paths, actual commands executed, tests with pass/fail/not-run status, output paths/hashes, observed results, and blockers. Distinguish implementation completion from an executed experiment. If GPUs/data/permissions are missing, prepare runnable commands and record BLOCKED; do not report a run.

Use append-only per-attempt records and atomic shard outputs. Resume only when source/configuration/model/data/scorer/protocol hashes match. Reject duplicate successful keys; retain a retry ledger. Record the full executed configuration rather than relying on prose.

Every gate record must identify the source revision, configuration hash, tests actually executed, and any untested cases. Never bypass a failed correctness, capacity, pairing, or scoring gate to meet the deadline. Stop at the task's designated output; do not launch the next stage automatically.
