# Consumed-input construction profiling

Source parent 944098b passed all ten packet/storage/prefill GPU controls but
the subsequent repeated complete-request study failed the all-baseline 1.05x
latency gate. Preserve both results. Source construction exceeds Top-K by
approximately 0.81--0.88 seconds; do not change bytes or selection to hide it.

Profile only the same two consumed Description/MM-Vet inputs. All five frozen
precision/storage adaptations use the same model, processor and prefill.
Use two warmups and three profiled repetitions per method, randomized inside
each engine. No decode or new task scoring. All original packet hashes, byte
counts, initial logits, retirement and beneficial source-code storage must
still match their immutable parents. Hashes/scoring are outside phase timers.

Nested Python monotonic timers report inclusive and child-subtracted exclusive
host wall time. CUDA events are recorded without per-call synchronization and
resolved after a final synchronization. They measure current-stream elapsed
time including host launch gaps, NOT kernel-only active time. Instrumentation
perturbs execution: never use this diagnostic as uninstrumented performance
evidence. Preserve raw per-call traces and averaged phase counts/times.

Two one-A100/12-CPU/48-GiB workers, 15-minute limits; at most six aggregate
A100s. Full clean-source CPU proof precedes admission, and afterany CPU analysis
checks missing producers as well as successful ones. Select only a measured
dominant operation for a subsequent exact optimization. All existing envelope
checks, patch decisions, model weights, memberships and K/V bits stay unchanged.
No fresh cohort, calibration, larger sweep, threshold relaxation or paper claim.
