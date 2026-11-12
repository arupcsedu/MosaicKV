# Bounded CPU recovery after V36/V37

This retry changes audit infrastructure only. It does not alter model code,
scientific numerical/quality gates, byte budgets, or historical raw artifacts.

- Verification: the unchanged scanner converts nested scope arrays to tuples.
  The new validator accepts their identical JSON representation, while retaining
  every identity value, signature, owner, error and cursor check. Resume only
  saved verification epochs; never replay completed historical traversal.
- Coverage: eight 2-GiB workers cannot accommodate the previously nominated
  Parquet two-pass workload. Prioritize small JSON/YAML metadata and queue only
  work within the same limits. Preserve every resource-deferred file as
  unresolved. One proof wave is authorized by this finite controller.
- CPU suite: restore the known GCC/CXX setup, retain the complete suite, and
  add per-test names, duration reporting and 60-second stack diagnostics.
  The new infrastructure timeout is 1,200 seconds; no scientific tolerance,
  test assertion or test denominator is relaxed.

Run the `cpu` mode in Slurm, followed by an `afterany` `dispatch` callback.
The callback submits no worker if the CPU gate fails. Otherwise it submits
eight one-CPU verification tasks and one dependent CPU closeout. Each owner
has at most six bounded verification steps; valid partial checkpoints survive
later failures. The closeout independently validates checkpoint transitions,
runs the single resource-feasible proof wave, and merges reviewed exclusions.
All stages are write-once and require a clean source SHA.

No GPUs, model inference, benchmark-label access or continuous queue polling
are involved. Fresh COCO materialization remains disabled: missing historical
media/source-group evidence cannot be supplied by a bookkeeping repair.
