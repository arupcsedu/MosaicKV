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
