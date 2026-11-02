# Consumed-input construction profiling

Second-level diagnostic: the parent is now the completed aggregated-transport
complete-request benchmark at source 461214f. Text-source optimize still has
approximately 0.44--0.45 seconds exclusive wall time. Add named scopes for
metadata validation, original materialization, overlay removal, source
encoding, projection metadata, candidate materialization, sparse patch build,
text-only patch checks, each of the three independent materializations and
bitwise comparisons, and final accounting. Only scopes and a temporary pair
name change; all operations, checks, packet bits and selection remain fixed.
The inactive scopes create no events or retained tensors. Require the same
frozen packet/logit/storage controls before using the profiling attribution.

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
