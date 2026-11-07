# Prompt 8 — Optional end-to-end batching under an A100 allocator budget

Apply 00_MASTER.md. This is NOT on the critical path. Run only after the frozen runtime has real existing batch support and mandatory confirmation jobs have sufficient reserved resources. Otherwise write NOT_RUN with the reason. Do not implement a new microbatch-prefill/decode-assembly scheduler for this experiment.

Validate batching first on 8 fixed development inputs: compare individual and batched FullKV with correct padding/positions and compare EviCastKV with its declared per-request allocation. Preserve per-example logical positions, masks and scoring. Do not silently distribute one request's cache budget across others.

Choose 32 representative development requests independently of scores and lock identical batch groupings for all methods. Test batch sizes [1,2,4]; batch 8 is optional only if already supported. Test FullKV, shared-runtime Top-K and frozen EviCastKV. A compressed per-request cap C means total target B*C plus explicitly measured inter-request padding, not a smaller total pool. Match actual compressed allocation at a given B.

Measure the WHOLE path from prefill through transient compression to completed decode. Report prefill, compression and decode-stage memory/latency, peak allocated/reserved GPU memory, sampled process GPU memory if available, completion counts, E2E wall time, output lengths, task quality, and OOM stage. A decode-only fit does not count as a feasible batch. For throughput use completed valid requests or actually emitted non-padding tokens divided by wall-clock makespan, not sum of request times.

Choose allocator limits from three fractions [1.0,0.75,0.50] of measured visible device memory, locked before comparison. Run each method/batch/limit in a fresh process to avoid retained allocations affecting OOM. Apply torch.cuda.memory.set_per_process_memory_fraction before model load and log the effective byte limit and allocator backend. Report model-load OOM separately. The cap controls the PyTorch allocator; it is not a total-process GPU-memory guarantee, an emulation of L4/RTX 4090, or a different A100 compute/bandwidth partition. Do not touch other GPU processes.

Use three repeats where feasible; retain every scheduled OOM/failure. Call the largest passing B only the largest TESTED feasible batch. No quality-equivalence or goodput claim without its separately declared quality condition. Report no batching advantage when that is the result.

Deliver batching_results.csv, batch_parity.json, memory_feasibility.csv, batching_report.md and status. This experiment must not consume the October 29 writing window.
