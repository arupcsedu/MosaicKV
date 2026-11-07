# Paired inline-repair engineering benchmark

The preceding two-input gate passed all 22 records, including bitwise
128-prediction natural/forced parity and unchanged measured initial, final and
decode-counter-peak storage. A single call does not establish a latency win.

Freeze the simpler prototype policy from the gate's `full` arm, using the
original validated source codec (not the optional zero-row map). Compare only
the old disposable-probe execution against the inline implementation. Nothing
else changes: model and dataset revisions, media/prompt hashes, BF16 eager,
selection, prototype values, CPU residuals, logical positions, entropy threshold
and all 128 predictions. Use the original CPU entropy reducer in both paths.
The optional zero-row source replacement remains tested and separately reported;
do not hide its small additional storage saving or extra runtime.

Use one exclusive A100 and 12 CPUs per consumed input, at most two workers.
Five warmups PER method precede twenty timed PAIRED repetitions, with seeded,
balanced randomized method order. Include the complete request, prefill,
construction and decode. CUDA events and synchronization bracket each trial.
Store every trial, error, allocator peak, realized KV boundary, entropy, prediction
parity flag, manifest and GPU/environment telemetry. Recheck unique paired IDs,
physical storage matching and parity from raw rows. Never silently drop a trial.

Report p5/p50/p95/mean/std and paired 10,000-replicate ratio/difference intervals.
The equal-or-faster engineering gate requires each input's Bonferroni-adjusted
one-sided upper ratio <=1.00; a separate stronger target is <=0.96. Neither a
5%-slower trade-off nor a point estimate alone passes. Bootstrap units are timing
trials on fixed consumed images, NOT independent benchmark examples.

FullKV is not a latency competitor. This is a runtime optimization comparison,
not evidence of superiority over PrefixKV, VL-Cache, LOOK-M or Prompt Top-K.
Captions must remain identical; no new quality claim, fresh cohort, calibration
or paper promotion follows automatically. Whole-request allocator peaks and
decode-owned KV counters remain separately labeled.
