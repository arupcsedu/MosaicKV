# CPU historical-inventory parser correction

The first V7 streaming attempt exposed a real performance defect, not a
scheduler-allocation problem. The inherited lexical scalar matcher evaluated
`text[i:]` for every number inside skipped JSON values. Large numeric arrays
therefore repeatedly copied the remaining string, making a nominally linear
metadata projection quadratic. Its deadline was checked between files but
not inside this operation. Some workers produced no checkpoint after more
than twenty minutes; completed chunks also exceeded their 480-second budgets.

Array `22763193` was stopped because of this diagnosed defect. Preserve its
partial checkpoints, completed worker summary, scheduler records and derived
incomplete-coverage result. Do not discard the negative or call it a complete
exclusion scan. No GPU experiment was canceled or rerun for this correction.

The replacement uses offset-based lexical matching and checks deadlines
inside JSON projection. Benchmark values remain lexically removed before
JSON object construction. Malformed canonical inputs, unsupported structures,
and deadline-exhausted records remain explicit coverage errors or incomplete
work; they must never produce a freshness pass. Synthetic tests verify work
scaling by token/offset observations rather than fragile wall-time assertions.

Start from a new output root bound to the corrected source. Do not silently
resume an old checkpoint under a different parser/source SHA. The finite CPU
retry runs the complete CPU suite first, then eight identity-inventory shards
with at most four concurrent. Each worker makes at most three 180-second
invocations, under a 15-minute Slurm wall limit. An `afterany` reducer records
missing, failed and incomplete shards. There is no polling or GPU admission.

This corrects inventory execution only. Even complete declared identity-field
coverage does not certify unknown media aliases or global source groups.
No candidate sampling, materialization, configuration freeze, independent
confirmation, or superiority claim is authorized by the retry.
