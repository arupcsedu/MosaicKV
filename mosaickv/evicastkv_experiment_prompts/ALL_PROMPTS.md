# EviCastKV experimental campaign — complete Codex prompts

Read README.md for the experimental matrix and scope. Apply the master instruction to every stage.

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


---

# Prompt 1 — Discover the runtime and lock the experiment contract

Apply 00_MASTER.md. Implement the campaign scaffold using the existing repository structure; do not refactor the model runtime yet.

1. Locate the actual model adapters, prompt-query capture, stable/routed selector, allocator, packer, decoding loop, scorer, data loaders, manifests, tests and scheduler scripts. Read the first-priority audit outputs if present. Write repository_map.md with real paths/symbols. The public name is EviCastKV; preserve internal legacy identifiers where renaming would break provenance.
2. Verify runtime versions, full checkpoint/processor identifiers and revisions, real GPU allocation, CPUs per worker, and existing batch support. Record whether each request is single-GPU or model-parallel. Preserve the historical revision as a reference. Use the paper's PyTorch 2.8.0/Transformers 4.55.2 as the documented setting, not as a reason to overwrite a different environment without reporting it.
3. Create protocol_lock.yaml and a validator. Recover the historical observation window 16, routing fraction 1/256, shortlist 8%, graph degree 8, two correction rounds, signed coefficients, protection and rounding rules from code. Report every discrepancy; never hardcode a missing value from an assumption.
4. Define COCO/LLaVA-v1.6-Mistral-7B budgets [20,40] MiB and the four existing MileBench subtasks/LLaVA-1.5-7B budgets [16,40] MiB. Preserve FP16, HF eager and the validated output policy. Recover whether '64 tokens' means a fixed decode schedule or max_new_tokens with EOS; store both emitted-token and forward-step counts. Do not silently change scoring or stopping.
5. Deterministically choose 64 development inputs per workload (16 per MileBench subtask), from previously used/development pools. Within them, designate 32 validation inputs (8 per subtask), 16 timing inputs (4 per subtask), and 8 profiler inputs (2 per subtask). Use only input characteristics for length stratification.
6. Reserve 256 fresh COCO and 256 fresh MileBench examples (64 per subtask), excluding all historical/development sample and media groups. Create exclusion_report.json. If historical exclusion records are incomplete, say freshness is unverified and block a 'fresh confirmation' claim. No outcome filtering. Store separate files/hashes; do not score confirmation data.
7. Create a typed result schema: run/sample/media IDs; attempt/repeat; source/config/model/processor/data/scorer hashes; task and split; method and effective fallback; budget and actual per-layer/head quotas; route counts; token counts and positions; initial/final/boundary-peak KV; peak allocated/reserved GPU memory; stage and full-request timings; raw prediction/token IDs; score; status/error/OOM stage; GPU UUID/job ID. Store hashes of large metadata, not large GPU tensors, in performance records.
8. Provide runnable CLI/job wrappers using actual repository commands, with dry-run, validation, small pilot, development, confirmation, analysis and resume modes. Use a new lightweight wrapper only if needed. Names for commands must be implemented and tested, not invented in documentation.

Deliver repository_map.md, protocol_lock.yaml, manifests/, result schema, command_examples.md, readiness.json and unit tests. Run CPU/schema tests and a two-input smoke run only if an allocated GPU is available. Do not open confirmation scores or launch a large sweep.


---

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


---

# Prompt 3 — Validate scoring, baseline fidelity and first-token sensitivity

Apply 00_MASTER.md. Require cache correctness for the tested paths. Use the fixed 32 validation inputs per workload, not confirmation data. Do not choose examples based on FullKV answers.

1. Verify the pinned COCO tokenizer/CIDEr reducer and MileBench per-subtask Accuracy/ROUGE-L reducers with existing evaluator tests and explicit fixtures: empty outputs, whitespace, option labels, answer prefixes, EOS/padding, and multiple references where applicable. Inspect the actual metric contract rather than substituting a generic implementation. Preserve raw outputs and parsed answers. Count invalid/empty/unparseable outputs and document inclusion rules.
2. Run FullKV, shared-runtime Prompt Top-K, the current EviCastKV exact selector, and the existing workload-specific local baseline. For baseline fidelity, compare the port against its pinned upstream path on the first 8 fixed validation inputs or representative saved attention fixtures when available. Compare scoring, quotas, mandatory/recent windows, retained IDs and LOOK-M merged tensors as applicable. Separate native fidelity tests from byte-adapted main comparisons. If upstream execution is unavailable, report fidelity unverified, not 'official reproduced.' Do not invent the cause of LOOK-M degradation.
3. Implement boundary A: current full-prompt prefill -> compression -> first token from original prefill logits -> compressed-cache continuation. Record when the first token is computed and when it is actually emitted.
4. Implement diagnostic boundary B only for adapter-valid inputs: prefill all but the final original prompt token; collect observations only from that available prefix; compress; process the held-back token exactly once at its original logical position; obtain the first answer token from that computation. Preserve visual expansion, position IDs and masks. Do not use full-prompt observations or logits from a separate run. Do not feed the held-back token or first answer token twice. If the final token cannot be safely separated by the adapter, report BLOCKED for B rather than silently dropping examples.
5. Under B, adjust mandatory positions to the last available prefix position under the same declared rule for all methods. Record the actual observation window and cache boundary. Reserve the held-back step's measured representable allocation when needed to match cache size just before the first answer token; if equal cross-boundary capacity cannot be guaranteed, report separate strata and do not attribute all A/B changes to one factor.
6. Validate FullKV A/B chunked-prefill numerical behavior independently. Run all comparable methods under both boundaries on the fixed subset. Record first-token probabilities/IDs, the first answer-bearing output prefix, parsed-answer changes and task scores. Use the native parser to diagnose short-answer sensitivity; never feed references into the selector. Treat B as a changed evaluation protocol, not an information-leak fix or a new headline score.
7. If a scorer/adapter defect is found, version the correction and regenerate ALL affected methods on the same inputs. Preserve pre-fix results as invalidated history. Do not retune on confirmation data.

Deliver scoring_fixtures.json, predictions.jsonl, parsing_audit.csv, baseline_fidelity.md, first_token_sensitivity.csv, boundary_protocols.yaml and GATE_scoring.json. Expected outcome: correct measurement and a documented sensitivity result, not guaranteed higher scores. If materially affected short-answer outcomes remain unvalidated, restrict the corresponding claim.


---

# Prompt 4 — Profile complete requests and establish FullKV/Top-K/EviCastKV timings

Apply 00_MASTER.md. Require correctness and scoring gates for each tested configuration. Work on the declared development subsets; keep all confirmation outputs closed.

Create two modes: diagnostic profiling and unprofiled timing. Use the 8 profiler inputs per workload for diagnosis, and 16 timing inputs (4 per MileBench subtask) for repeated performance measurements. Run tight and moderate budgets. FullKV runs once per input/output policy, not once per nominal compression budget.

Instrument the real executed path: input preparation and transfer; vision/prefill; query/attention capture; descriptor/candidate gathering; normalization; graph construction; selection/coverage corrections; physical-capacity reconciliation; packing/publication; decoding; output detokenization. Avoid materializing full attention matrices solely for measurement if the real path does not require them. Attribute observation cost to prefill when it executes there.

Use PyTorch CPU/CUDA profiler ranges to locate hotspots, with shapes/stacks/memory tracing only on bounded diagnostic cases. Export traces and self/inclusive times; do not sum nested inclusive ranges or CPU/GPU intervals as if disjoint. Headline latency must come from unprofiled runs.

Warm the loaded model using five representative warmups per relevant method/shape regime. On the same GPU, run each request/method three times in a counterbalanced order, without simultaneous timed workloads. Reuse only loaded model weights and declared input-file caching; every timed request recomputes preprocessing, prefill, policy work and decode. Disable unused compression hooks for FullKV. Verify that changing method hooks does not contaminate subsequent runs.

Use synchronized monotonic wall time for completed request latency; CUDA events only for device phases with correct stream dependencies. Do not set CUDA_LAUNCH_BLOCKING=1 or insert synchronization inside every kernel for headline runs. Define E2E as beginning before request input preparation and ending after completed output detokenization; exclude model loading and metric scoring and report setup separately. Record narrower model-only timing separately if useful, never relabel it E2E.

Record first-logit-ready, actual first-token-emitted, second-token-emitted and final-output times. TTFT is actual emission, not retrospective availability of a prefill logit. Report emitted lengths, forward steps, mean request latency, p50 and sample-count-qualified p95, and between-repeat variability. Match the locked EOS/output policy; do not force equal workload retrospectively when lengths differ.

Record active unique KV storage separately from whole-request PyTorch peak allocated/reserved memory and baseline model allocation. Reset peak counters only before a complete request; use separate diagnostic runs for stage-local peaks. Log NVML/process GPU memory if available with its sampling interval and visibility limitations, and CPU RSS. No measured process-HBM claim based solely on KV or allocator statistics.

Deliver raw timing JSONL, profiler traces, phase_breakdown.csv, fullkv_reference.csv, memory_boundaries.csv, measurement_contract.md, hotspot_report.md and GATE_measurement.json. State which components dominate from traces, not assumptions. Use pilot timing to predict remaining GPU-hours, without promising a speedup or a 50-ms threshold.


---

# Prompt 5 — Optimize only measured bottlenecks and preserve semantics

Apply 00_MASTER.md. Require Prompt 4's actual hotspot report. Preserve a runnable reference implementation and its configuration.

Choose at most two highest-cost, tractable implementation bottlenecks. Possible changes include tensorized descriptors, batched gathers, segmented operations, removing redundant transfers, caching request-invariant metadata, or reducing synchronization. Implement only changes justified by the profiler. Do not build a new serving engine or change attention backend, dtype, objective, routing, budgets, candidate domain, or stopping policy in this stage.

Maintain the existing physical allocation policy, mandatory entries, per-head minima, layout padding, rounding and filling rules. Never use unconstrained global top-K. Preserve deterministic tie-breaking; do not add epsilon perturbations to disguise changed rankings. If vectorized reduction order changes selections, either recover the reference selections or register the implementation as a distinct algorithmic variant requiring quality evaluation.

On identical saved DEVELOPMENT inputs/observations, compare reference and optimized selections, routed heads, candidate membership, allocation quotas, retained K/V, original positions, validity masks, target/actual bytes and output traces. Saved inputs may be reused for this diagnostic; full performance requests must be recomputed. Re-run structural and numerical gates after each change. Changes to common packing or score collection must benefit Top-K and other compatible methods too, without adding unnecessary graph work to them.

Measure the paired factorial: reference Top-K, optimized Top-K, reference EviCastKV at 1/256, optimized EviCastKV at 1/256. Include FullKV as an uncompressed timing reference. Use the same 16 timing inputs per workload, declared budgets, five warmups and three repeats, all timed unprofiled. Compare selected IDs and outputs before calling this semantics-preserving. Report regression or no gain as valid outcomes.

Release source-cache/query-hook/debug references at the same documented lifecycle boundary across comparable configurations. Do not make memory look lower by excluding live allocations or latency look lower by moving computation outside the timer.

Deliver optimization_diff.patch, equivalence_report.json, before_after_runtime.csv, raw repeated timing files, regression_tests/, optimization_report.md and GATE_optimization.json. Classify each change as semantics-preserving, corrected bug, or algorithm change. Mark historical results affected by a discovered bug; do not compare their invalid numbers with new measurements. A speedup in common infrastructure is not evidence of a graph-selection quality benefit.


---

# Prompt 6 — Test graph participation, scoring components and routing controls

Apply 00_MASTER.md. Require valid common runtime and allocation. Use ONLY the 64 development inputs per workload, with budgets COCO [20,40] and MileBench [16,40] MiB. All counts and rules are new protocol choices, not historical facts.

A. Route-fraction screening.
Run the recovered current deterministic routing policy at fractions [0, 1/256, 1/16, 1/4]. Fraction 0 must bypass graph construction and its outputs must be checked against shared-runtime Top-K; do not assume they are identical. Recover/define the count rounding explicitly. Pilot fraction 1 on 8 development inputs and expand only if the prespecified resource cap permits it. No result-based choice of which examples finish. Hold weights, window, shortlist fraction, degree, protection, allocation semantics and generation fixed. Report actual routed pairs, candidate count and bytes. If the original policy cannot rank additional heads, document a simple deterministic extension as a NEW policy rather than pretending it was already implemented.

B. Same-candidate scoring ablations at a development-selected fraction r* (or the preregistered diagnostic default 1/16 if no selection has been authorized).
Use the same prompt observations, routed head IDs, candidate universe, graph descriptors and per-head capacities in a labeled fixed-candidate/fixed-capacity mechanism study:
  complete: recovered coefficients unchanged;
  routed_prompt_control: attention utility only, value/novelty/graph/modality coefficients zero;
  coverage_off: graph and modality coverage coefficients zero, value/novelty unchanged;
  value_novelty_off: value and novelty coefficients zero, coverage unchanged.
Do not retune coefficients in ablated variants. Do not call coverage_off 'no graph': novelty/candidate formation may still need it. Keep individual signal normalization and score domains fixed. If the deployment allocator is treatment-dependent, freeze a documented prompt-derived feasible quota plan for ALL variants and run a complete-policy control under that plan. Label results as that controlled study; do not substitute its timing for native deployment timing.

C. If pursuing entropy routing, use a declared normalized entropy over valid prompt attention: mean(-sum p*log(p)/log(n_valid)), with 0 log 0=0 and entropy 0 when n_valid<=1; aggregate mapped query heads within each KV head. Use no future token observations and no renormalization over shortlists. Compare entropy-selected heads with random heads at the SAME routed count and layer quotas, using seeds [13,29,47]. Measure entropy computation cost. Existing deterministic routing remains a comparator. Do not pool seeds as independent task examples.

D. If making an inter-modal-specific benefit claim, add a bounded cross_modal_edges_off control at r*: remove cross-modal edges while freezing candidate IDs, quotas and all other settings. Report whether the ablation actually changes edges, scores or selections. If no cross-modal edges were active, explicitly mark the claim unsupported; do not fabricate modality metadata.

Save native quality, planning and full-request time, memory, selected-set overlap, raw/marginal score terms, realized graph degrees and active edge-source counts. Profile and quality-only cached diagnostics are not headline timing rows.

Deliver routing_sweep.csv, component_ablation.csv, routing_controls.csv (when run), configuration_activity.json, development_predictions.jsonl and development_decision.md. Include every scheduled condition, failure and timeout. Write a Pareto/trade-off report without combining CIDEr, Accuracy and ROUGE-L into one score. Do not select r* using confirmation data, p-values alone, or best random seed. A human-reviewed selected_config.json and a prespecified claim/analysis family are required before confirmation; otherwise report NO_CONFIG_FROZEN and stop.


---

# Prompt 7 — Freeze the method and run fresh paired confirmation

Apply 00_MASTER.md. Require successful correctness/scoring/measurement gates plus selected_config.json from development. Do not choose a new configuration in this task.

1. Freeze code, uncommitted-diff hash, dependencies, model/processor revisions, routing rule/fraction, coefficients, capacity rules, preprocessing, observation boundary, EOS/padding, evaluator and analysis contract. Copy immutable resolved files to a new run directory. Reject configuration drift. No confirmation-derived repairs, tuning, cohort enlargement or sample replacement. A genuine defect invalidates the affected version and requires a separately recorded rerun; already inspected data are no longer untouched confirmation for tuning.
2. Check exclusion manifests and availability of 256 fresh COCO examples and 256 fresh MileBench examples (64 each ActionPrediction, ActionSequence, CLEVR-Change, Spot-the-Diff). Do not filter by FullKV performance. Missing metadata/sample history blocks a 'fresh' claim.
3. Main matrix per workload: FullKV once per input; shared-runtime Prompt Top-K, frozen EviCastKV, and the validated workload-specific local baseline at both budgets. COCO: [20,40] MiB and VL-Cache (R). MileBench: [16,40] MiB and LOOK-M (R). Total is 3,584 request executions before repeated timing or extra controls. FullKV is an uncompressed reference, not a matched-byte competitor. An unverified port remains explicitly unverified and cannot support a strong superiority claim; never silently count it as validated.
4. Confirm the mechanism at the tight budget: add complete and routed_prompt_control under the SAME frozen candidate/quota mechanism contract on all 256 examples per workload. This is at most 1,024 extra executions, yielding a maximum core total of 4,608. Reuse a complete-policy row only if its entire protocol hash and actual candidate/quota contract match, not merely its method name. Keep this controlled mechanism study distinct from deployment timing.
5. Recompute full requests for headline measurements. Check initial storage equality per paired input; verify final/peak equality only under genuinely matched forward-step schedules. Otherwise report actual growth and step-aligned storage instead of imposing artificial equality. Never use aggregate mean-byte equality as the pairing gate.
6. Choose 32 timing-repeat inputs per workload by the frozen input-only manifest (8 per MileBench subtask). Run three repeats per primary method/budget in counterbalanced order on the same device, after warmups. Record median/mean and variability; repeats do not increase unique quality N. The first timing observation can count only if it followed the same timing protocol. Run repeated tests unprofiled.
7. Export raw tokens, parsed answers, predictions, scores, timings, memory and per-attempt status. Preserve failure denominators and OOM stage. Do not drop failed samples to create a favorable complete intersection. Block complete-cohort claims when required rows are missing. Recover only failures authorized by the frozen retry policy; log every attempt.
8. Scheduler jobs may run independent sample blocks on available GPUs. Keep a sample's compared methods on the same GPU and vary method order. No eight-GPU acceleration claim from embarrassingly parallel experiment execution.

Deliver confirmation_manifest.json, predictions.jsonl, measurements.jsonl, pairing_report.json, repeated_timing.jsonl, completion_report.md and GATE_confirmation.json. Do not analyze interim quality to stop early or extend the sample. A null or unfavorable confirmation is an acceptable scientific result.


---

# Prompt 8 — Optional end-to-end batching under an A100 allocator budget

Apply 00_MASTER.md. This is NOT on the critical path. Run only after the frozen runtime has real existing batch support and mandatory confirmation jobs have sufficient reserved resources. Otherwise write NOT_RUN with the reason. Do not implement a new microbatch-prefill/decode-assembly scheduler for this experiment.

Validate batching first on 8 fixed development inputs: compare individual and batched FullKV with correct padding/positions and compare EviCastKV with its declared per-request allocation. Preserve per-example logical positions, masks and scoring. Do not silently distribute one request's cache budget across others.

Choose 32 representative development requests independently of scores and lock identical batch groupings for all methods. Test batch sizes [1,2,4]; batch 8 is optional only if already supported. Test FullKV, shared-runtime Top-K and frozen EviCastKV. A compressed per-request cap C means total target B*C plus explicitly measured inter-request padding, not a smaller total pool. Match actual compressed allocation at a given B.

Measure the WHOLE path from prefill through transient compression to completed decode. Report prefill, compression and decode-stage memory/latency, peak allocated/reserved GPU memory, sampled process GPU memory if available, completion counts, E2E wall time, output lengths, task quality, and OOM stage. A decode-only fit does not count as a feasible batch. For throughput use completed valid requests or actually emitted non-padding tokens divided by wall-clock makespan, not sum of request times.

Choose allocator limits from three fractions [1.0,0.75,0.50] of measured visible device memory, locked before comparison. Run each method/batch/limit in a fresh process to avoid retained allocations affecting OOM. Apply torch.cuda.memory.set_per_process_memory_fraction before model load and log the effective byte limit and allocator backend. Report model-load OOM separately. The cap controls the PyTorch allocator; it is not a total-process GPU-memory guarantee, an emulation of L4/RTX 4090, or a different A100 compute/bandwidth partition. Do not touch other GPU processes.

Use three repeats where feasible; retain every scheduled OOM/failure. Call the largest passing B only the largest TESTED feasible batch. No quality-equivalence or goodput claim without its separately declared quality condition. Report no batching advantage when that is the result.

Deliver batching_results.csv, batch_parity.json, memory_feasibility.csv, batching_report.md and status. This experiment must not consume the October 29 writing window.


---

# Prompt 9 — Analyze frozen measurements and generate paper-ready evidence

Apply 00_MASTER.md. Analyze only completed, hash-validated artifacts; no model inference or method tuning in this task. Historical, development, confirmation, first-token-boundary, fixed-quota and batching rows are separate strata.

1. Verify planned/attempted/completed/scored/paired counts, duplicates, hashes, original sample/media group identity, actual bytes, output policy, and workload consistency. Do not silently use only favorable surviving pairs. Export the full missingness/failure ledger; incomplete required groups cannot support complete-cohort claims. Preserve original metrics and record all conversions.
2. Recompute COCO corpus CIDEr using the pinned tokenizer/reducer and its documented document-frequency/reference policy; do not average shard CIDEr or silently switch DF conventions during bootstrap. Duplicate resampled items need unique resample identities. For MileBench, preserve equal-subtask averaging within each metric family. Accuracy and ROUGE-L remain separate; 256 total examples means 128 applicable examples per family. Cluster resampling by shared underlying media/document/video where required; use input-derived pairing groups, not method outcomes.
3. Use 10,000 paired bootstrap replicates with a fixed recorded seed. For timing, summarize repeats per input first and bootstrap paired input-level estimates, or use the preregistered nested scheme; repetitions are not independent quality observations. Report between-repeat variability separately. Do not use bootstrap uncertainty as a substitute for fresh data.
4. Prespecify the primary six EviCastKV-minus-Top-K quality contrasts: two COCO budgets and two budgets for each of the two MileBench metric families. Report ordinary 95% intervals descriptively and Bonferroni-adjusted simultaneous intervals for this six-contrast family (coverage 1-0.05/6 per interval). Component, published-port and timing contrasts are secondary declared families. Clearly label any additional analysis exploratory. Do not add or remove tests to improve significance.
5. Do not assert equality, noninferiority or 'same quality' from an interval crossing zero. Unless independently justified margins were locked before confirmation, report signed differences and uncertainty only. Confidence intervals with a lower bound of exactly zero after rounding do not establish strictly positive improvement. Random routing seeds are not extra independent dataset samples; report each seed and the chosen aggregation separately, never the best seed alone.
6. Compute speedup as mean baseline latency / mean candidate latency, and reduction as 100*(1 - mean candidate / mean baseline), within the same declared cohort/protocol; also supply paired absolute differences. Recompute these ratios in bootstrap replicates when reporting uncertainty. Distinguish mean-of-ratios from ratio-of-means explicitly. No percentage based on display-rounded inputs. Report peak process/allocator memory separately from active KV; distinguish actual final from peak and sampled from continuous maxima.
7. Generate LaTeX tables with model, task, N per metric, method, cap, realized bytes, quality, compression time, TTFT/second-token gap where relevant, decode/E2E times, memory peaks and uncertainty. Add a paired-difference table, component table, validation table and optional batching table only when data exist. Render one chart per metric with measured capacities or routing fractions and uncertainty; no invented points, interpolation claims, chosen favorable axes, or fabricated ablation curves. Include CSV/JSON inputs and deterministic plotting scripts. Keep unwieldy complete matrices in the appendix, not suppressed.
8. Write claim_evidence.md linking each proposed claim to raw file hash, row filter, test definition, point estimate and interval. Classify claims as supported in scope, inconclusive, contradicted, invalid measurement or not tested. Separate whole-implementation improvements from controlled mechanism effects.

Deliver analysis_manifest.json, complete result CSVs/JSON, tables/*.tex, plots with source scripts, a limitations/completeness report and GATE_analysis.json. Missing results must be marked NOT_RUN/INCONCLUSIVE, not estimated from old tables or hardware peaks.


---

# Prompt 10 — Reproduce the analysis and freeze the writing packet

Apply 00_MASTER.md. This task is read-only with respect to runtime algorithms, locked configurations and raw measurements.

From a clean analysis working directory, regenerate every number, LaTeX table and figure using saved artifacts and documented commands. Run schema/pairing tests and compare output hashes or numeric contents. Verify the 3,584 main-matrix expectation or an explicitly versioned, pre-run matrix change; do not require a number if the campaign legitimately remained blocked/incomplete. Report actual totals without replacing missing rows.

Audit: no confirmation/development overlap; no best-seed selection; no profiler-contaminated headline timings; FullKV without unused compressor work; no hidden source-cache reuse in E2E; no compressed-budget fallback mislabeled success; no nominal-token memory claim replacing physical bytes; no baseline/source version mismatch; no runtime advantage attributed to graph components without the corresponding controlled result.

Produce final_writing_packet.md containing:
- Exactly which old tables/claims are superseded, with old and new protocol hashes.
- New claims that may be written, with numerical evidence and scope.
- Claims that remain inconclusive, negative or untested.
- The actual method settings, model/data/preprocessing, timing and statistical protocol.
- Main-paper versus appendix table/figure mapping.
- Artifact availability and blocked reproduction steps.
- Experiment-command ledger, test results, source hashes and a minimal README.

Do not edit the entire manuscript or strengthen a claim because a deadline is near. On October 29, regenerate already completed results and finalize this packet; do not start another hyperparameter search. Return the exact artifact paths and verification status. Creating scripts or submitting scheduler jobs is not the same as obtaining completed experimental results.
