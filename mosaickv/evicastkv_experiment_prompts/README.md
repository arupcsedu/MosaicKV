# EviCastKV — second-priority experimental campaign

Prepared for the October 29, 2026 writing deadline. This package contains EXPERIMENT DESIGNS AND CODEX INSTRUCTIONS, not completed measurements or a runnable implementation of the user's private runtime. Repository commands must be discovered or implemented and tested by Codex.

## Basis and distinction from proposed choices

Source requirements are from `EviCastKV__QUERY_FORECASTED_INTER_MODAL_KV_CACHE_COMPRESSION_v1.4(1).pdf` and `Revised Revised Rescue Plan 3.pdf` (Prof. Fox's assessment dated October 24, 2026). The paper describes prompt-window exact selection, the two LLaVA checkpoints, physical capacity accounting, COCO/MileBench protocols, and old experimental results. The review requires profiling the whole path, preserving capacity semantics, testing graph utility with shared code, a random control for entropy routing, validating first-token/scoring behavior, measuring FullKV, and establishing complete-path batching feasibility.

The sample sizes, 1/16 and 1/4 screening fractions, random seeds, repeated-run counts, confirmation budgets, statistical family and new artifact names in this package are PROPOSED protocol choices. They are not already executed settings or results. Coefficients, actual routing rules, processor revisions and internal paths must be recovered, not invented.

## Use

Extract `evicastkv_experiment_prompts/` into the existing research repository. Do not overwrite project AGENTS.md. Begin each Codex task with:

> Read `evicastkv_experiment_prompts/00_MASTER.md` and the selected numbered prompt. Follow existing repository instructions. Implement and execute that task within the available resources, then report the required evidence and blockers. Do not automatically run subsequent stages.

Read stages 1–7 in order. Stage 8 is conditional and must not delay confirmation. Stages 9 and 10 analyze and package completed data. A failed correctness gate blocks dependent runs; a negative scientific result does not.

| Stage | Task | Required outcome |
|---|---|---|
| 01 | Discover/lock | Actual code mapping, frozen protocol, disjoint manifests, implemented runner |
| 02 | Cache correctness | Numerical/structural checks and actual-byte capacity validation |
| 03 | Scoring/boundary | Pinned scorer checks, port audit, clearly separated boundary sensitivity |
| 04 | Profile/measure | FullKV/Top-K/EviCastKV real E2E and full-path hotspot breakdown |
| 05 | Safe optimization | Preserved algorithm or explicitly versioned algorithm change; paired before/after |
| 06 | Mechanism ablations | All scheduled routing fractions, scoring controls, and uncertainty/trade-offs |
| 07 | Fresh confirmation | Frozen configuration on predetermined untouched samples |
| 08 | Optional batching | Complete request feasibility including prefill/transients, or NOT_RUN |
| 09 | Statistics/export | Auditable reductions, intervals, plots, tables and supported-claim mapping |
| 10 | Reproduction audit | Recreated tables/figures and a writing packet without invented outcomes |

## Proposed data matrix

| Use | COCO | MileBench |
|---|---:|---:|
| Development | 64 inputs | 64 inputs, 16 per subtask |
| Cache/scoring validation | 32 development inputs | 32, 8 per subtask |
| Profiler traces | 8 development inputs | 8, 2 per subtask |
| Repeated development timing | 16 development inputs | 16, 4 per subtask |
| Fresh confirmation | 256 inputs | 256, 64 per subtask |
| Repeated confirmation timing | 32 input-selected confirmation cases | 32, 8 per subtask |

The subsets may be nested within the development set; the fresh set must not overlap any prior tuning/inspection or historical evaluation exclusion ledger. Do not use outcome-based sampling. If multiple examples share media/document/video content, use that grouping when splitting and resampling.

COCO uses the discovered LLaVA-v1.6-Mistral-7B adapter, initial KV budgets 20/40 MiB, corpus CIDEr, Prompt Top-K and the validated VL-Cache port. MileBench uses the discovered LLaVA-1.5-7B adapter, 16/40 MiB, the existing four subtasks and separate Accuracy/ROUGE-L metric families, Prompt Top-K and the validated LOOK-M port. Use FullKV in both as an uncompressed quality/timing reference. These are two checkpoints, not two independently validated model families.

Do not change the source output/EOS policy silently. If quality runs permit variable completion lengths, match initial bytes and compare step-aligned decode storage; do not claim equal final memory. A fixed-step timing microbenchmark is a separate protocol. A final emitted token need not itself have been processed into the cache.

## Request-count accounting

Per workload: 256 FullKV + 256 x 2 budgets x 3 compressed methods = 1,792 primary request evaluations. Across two workloads: 3,584. A tight-budget controlled complete-versus-routed-prompt study adds at most 256 x 2 methods x 2 workloads = 1,024, producing a maximum core total of 4,608. Reuse a complete row only under identical full protocol/candidate/quota hashes. Repeated timing, development, diagnostics and optional batching are ADDITIONAL work.

A 256-input cohort is a bounded workload proposal, not a power guarantee. For MileBench, each metric family has 128 applicable inputs, not 256. No fresh result can be claimed when exclusion history is incomplete.

## Critical path and resource limits

Use October 27 for validity checks/profiling and a bounded development study; October 28 for freezing the selected method and finishing confirmation; reserve October 29 for analysis, figure/table regeneration and writing. Do not pretend the review's October 25–26 tasks are complete. Actual feasibility depends on the code state, queue and measured pilot cost.

Compute required GPU-hours from actual remaining request counts and pilot times for each workload/method, with allowance for repeats and failures. Respect the actual scheduler allocation, not an assumed eight-GPU reservation. Drop optional entropy/full-graph/batching extensions before cutting correctness or starting confirmation with a changing method. Do not invent time estimates or numerical speedup promises.

Method selection is a research decision: inspect the development Pareto front using distinct native quality metrics, complete latency and physical memory. Record the practical criterion before outcomes are used, then save an author-approved selected_config.json before opening confirmation scores. If no useful mechanism signal appears, freeze a diagnostic configuration only with its negative/inconclusive rationale; do not manufacture a winner or silently redefine success.

## Minimum per-attempt provenance

Required identities: source commit and diff hash; resolved config; model/processor/data/scorer revision/hash; sample/group/task/split; method; actual routing/candidates/quota contract; input and output policy; backend/dtype; GPU/job identity; attempt/repeat/measurement mode. Measurements: raw output/token IDs; scorer output; emitted/processed token counts; initial/final/cache-boundary peak bytes; peak allocated/reserved memory and sampled process memory when available; wall-clock E2E and phase times; actual emission timestamps; status/error/fallback/OOM stage.

Mark measurement_origin as real_model, synthetic_correctness_test or dry_run. Dry runs and synthetic tests never become experimental quality/timing rows. Missing numerical fields are null with a reason, not 0. Retain every prespecified row and an attempted/completed/scored/paired denominator.

## Expected paper outputs

- Validation table: case class, number tested, numerical error, invariants, status.
- Main quality/resource table: native score and uncertainty, actual bytes, complete timing and memory for each declared method/budget.
- Routing curve: actual routed pairs/fraction versus quality and planner/E2E cost, one metric per chart.
- Component table: complete, attention-only routed control, coverage-off and value/novelty-off under explicit candidate/quota controls.
- Runtime decomposition: reference and optimized Top-K/EviCastKV with FullKV reference, costs charged where executed.
- Boundary diagnostic: A/B protocols, first-token/answer sensitivity and representational checks; not labeled leakage.
- Optional batching table only after full-path validation.
- Complete appendix matrices and negative/inconclusive outcomes.
- Claim-to-evidence map with raw hashes, row filters, aggregation and confidence intervals.

## External implementation documentation consulted

The source PDFs are the research basis. These official pages support implementation details only and do not validate experimental claims:

- PyTorch 2.8 profiler: https://docs.pytorch.org/docs/2.8/profiler.html
- PyTorch 2.8 CUDA timing and allocator semantics: https://docs.pytorch.org/docs/2.8/notes/cuda.html
- PyTorch 2.8 allocator fraction: https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.set_per_process_memory_fraction.html
- OpenAI Codex persistent instruction/workflow guidance: https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex

The presence of instructions, test code or scheduler submission logs does not establish completed experiments. Codex must return actual artifacts and statuses.
