# Preregistered dispatch-affinity diagnostic and separate CPU bookkeeping audit

The earlier C++ dispatch run passed source-word checks but failed its two-worker
wall-time gate. One measured candidate burst reached about 72 microseconds per
launch; it remains in the original negative result. The cause is unresolved.
This is **one new, explicitly registered diagnostic**, not a selective retry or
permission to replace the original result.

Before submission, freeze the protocol JSON, source SHA and original result
hashes. Keep the same source format/shape/seeds, binaries, geometry, numerical
controls, five warmups, twenty timed pairs, fifty launches per burst, 10,000
bootstrap draws and four-endpoint adjustment. Pin both methods' main thread to
the lowest CPU in its Slurm-allowed mask and restore affinity on exit. The worker
first uses `srun --cpu-bind=cores` to restrict that mask to allocated cores;
record its bind list, process and thread IDs as well as the selected CPU.

Record every burst, including warmups, in write-once JSONL: wall time, thread
CPU time, CUDA stream interval, voluntary/involuntary context switches and page
faults. Preserve raw clocks and negative wall-minus-thread bracket differences.
Wall minus thread time is only a proxy for time not charged to this thread; it
does not uniquely identify scheduler preemption. CUDA intervals include host
gaps and timing instrumentation, not just device kernel execution. No outlier
may be dropped or used to adapt the fixed sample count.

The strict gate is still the adjusted upper **wall-time** ratio <=1.00 in both
workers. Thread CPU time cannot substitute for it. No model, fresh cohort,
combined candidate or promotion follows automatically, even if it passes.

Separately profile frozen `f3c7902` History-V-first control flow on CPU. Mock
GPU/projector work but execute the actual scheduling module's metadata guards,
weak-reference checks, signature construction and bookkeeping. The profile does
not exercise CUDA stream-query cost, projector work or original format kernels.
Retain raw cProfile files and call tables; do not interpret this fixture's timing
as GPU or end-to-end latency. No implementation is changed by this audit.

Use the common environment and scratch-only caches. CPU work uses `bii`. The
single two-element GPU array uses one A100/12 CPUs/48 GiB per worker and a
ten-minute limit, admitted under the global six-A100 controller. A single
admission tick releases eligible tasks and exits; no long-running queue/log
poller is needed for this fixed pair. Every producer has an `afterany` analysis.
