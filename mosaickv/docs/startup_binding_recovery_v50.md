# V50: recover actual analyzer revisions, reusing completed byte proofs

V49's successful CPU-only recovery saved two-pass hash witnesses for all 134
nominated Parquet files. It distinguished original runtime writer bindings,
committed snapshot bindings, artifact-only bindings, and unrecovered items.
Those original witnesses and their negative classifications remain immutable.

This continuation never reopens a Parquet body. It revalidates the saved
classifier output against the pinned V49 report, the preserved V48 occurrence
registration, and the same byte hash and geometry. V49's observed whole-recovery
elapsed time supplies a conservative timing bound; it is explicitly **not** a
new per-file timing measurement. Current-source checks are stat-only.

## Additional original-writer evidence

Three archived CPU analyzer executions have exact 55-byte clean-Git startup
records and retained Slurm accounting commands. Their registered hashes bind
the clean-worktree guard, actual Python analyzer invocation, stage and output
directory. The guard must precede the executed analyzer; an echoed command or
arbitrary wrapper cannot qualify. A leading shell `exec` is accepted only when
it executes the actual Python analyzer. These three exactly hash-registered,
reviewed public command records are retained for offline verification, after
rejecting credential-bearing records. Generic accounting output and token
assignments are never exported. The validator returns only sanitized scope facts.

The actual recorded analyzer source and standalone clean-worktree guard must
match their reviewed Git blobs. Their output-directory/report linkage then
joins to the independently recovered original artifact-hash map. This matters
for the LOOK-M recovery: its analyzer revision differs from its input plan's
revision, so the enclosing directory name and input SHA were insufficient.

The CIDEr request-validation writer is checked separately. Its recorded source
SHA is tied to the actual writer because its `common.inputs()` guard requires
the running clean source to equal the plan SHA. The writer, guard and original
clean-source helper implementations are all byte-verified. This is not a rule
that every input-plan SHA identifies its analyzer.

## Limits and claim boundary

The 210-second shared deadline, 2-GiB combined-read budget, 8-MiB/100,000-node
projected-metadata limits, 512-MiB checkpoint ceiling, 64-GiB artifact ceiling,
32-GiB CPU ceiling and eight-worker maximum remain unchanged. Credential bodies
stay unread, completed JSON identity proofs are not replayed, and the source
and outputs are separately immutable.

These are recovered producer/artifact bindings, not yet semantic closed-writer
identity proofs. They do not alter the historical unresolved denominator or
authorize fresh COCO confirmation, GPU runs, materialization, or paper claims.
Dirty/ad-hoc writer closures, missing metadata and unbound artifacts remain
explicitly unresolved rather than being replaced by current-code guesses.
