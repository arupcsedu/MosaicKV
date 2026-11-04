# Private graph/arena verification: same consumed inputs only

The prior graph reader retained original prompt packets and all 128 parent
logits, but failed MM-Vet's decode-owned peak-storage gate. The fixed prompt
lease remained idle while history verification allocated its own reference
and output pairs. Capturing ordered lossless code replay was also not yet
tested. The attribution diagnostic showed negligible allocator-snapshot cost;
do not remove ownership checks.

This bounded successor tests ONLY the explicit policy `history_arena=true`,
`captured_code_replay=true`, `row_major_codes=false`. The separately tested
row-major source layout was slower and stays disabled. Paired K/V prefix
projection remains operator-only and is NOT part of this model test.

Before checkpoint loading, actual CUDA byte-graph controls must reproduce
every original code byte over three ordered layer passes, without doubled
in-place XOR, shared mutable state, or retained generator owners. The model
controls then check every original packet digest and all 128 parent logits,
no fallback, exact initial/final storage, charged workspace and graph owners,
and metadata retirement. History arena poisoning preserves three independent
verification fills. All methods also receive the common ordinary-output
lifetime fix. Thus storage comparisons use newly measured baseline peaks,
not stale pre-optimization values.

Every method retains its original direct reader AND original graph reader,
and receives the same new graph/arena option. PrefixKV also retains its fast
whole-text reader with all three options. A method cannot win by withholding
an optimization from its comparator. The finite afterany chain runs the full
CPU suite, two one-A100 controls, then (only if memory/correctness pass) paired
PrefixKV timing. Only a positive adjusted four-percent gate releases the
all-five-method timing extension. Five warmups and twenty repetitions use
seeded randomized order; paired full-request timings include construction,
capture, history, native attention, decode, and graph retirement. Raw trials
are preserved. Ten thousand bootstrap replicates use a conservative 64-test
engineering family. The upper ratio must be <=0.96 against EVERY baseline
reader, separately on BOTH consumed inputs. Equal or 5%-slower is not a pass.

At most two A100 workers here, within the aggregate six-A100 limit; one GPU,
12 CPUs, and less than two hours per worker. All failures receive terminal
analysis. No automatically submitted fresh cohort, quality tuning, or paper
campaign. This is a precision/storage variant with the main six mechanisms
disabled; independent benchmark-quality N=0. Decode-owned tensor-storage
peak is not whole-request GPU ownership peak. A positive two-input timing
diagnostic would not establish broad EviCastKV quality or superiority.

## Retirement-harness corrective retry

The first model-control attempt stopped before checkpoint loading in both
workers. A separately saved allocation diagnostic measured 31 live graph
objects before destruction, one afterward, unchanged tensor allocations,
and zero objects/allocations after the function returned. The harness called
`locals()` to test variable existence: its cached dictionary retained that
last strong graph reference even after `del graph`. Initialize the variable
explicitly and delete it unconditionally instead. A CPU weak-reference
regression covers this ownership trap; the real CUDA retirement guard remains
exactly 1,024 bytes with all original-code comparisons unchanged. Bind the
failed decision and allocation diagnostic in the corrective retry manifest.
