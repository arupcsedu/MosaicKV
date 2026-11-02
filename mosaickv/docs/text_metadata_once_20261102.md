# Common exact-text host-metadata optimization

The frozen consumed-input phase study identifies text-source optimization as
one of the largest exclusive host phases. Its text_slots comprehension built
an entire head's position dictionary once for EACH text token. Build it once
per head instead, retaining all original input validation and -1 missing-slot
semantics. Compute one immutable text set for mask membership; do not change
mask bits, slot order, selected positions, packets, precision or GPU operations.
No cross-request memoization or tensor-state reuse. Every method receives both
changes, including merged baseline values and all existing fallback checks.

Before inference, run the CPU-only audit on the actual pinned parent ledgers
and processor routes, comparing all 32-layer slot/mask outputs with the pinned
legacy function. Keep its timing labeled CPU metadata work, not model speedup.
Do not use benchmark answers, labels or failure identities for construction.

The follow-up is the same two consumed inputs and five frozen precision/storage
adaptations, not new algorithm curves or paper evidence. Source SHA is clean
and immutable. Reuse the passed same-source CPU suite and constructor-memory
proof boundaries. First excluded warmups verify original packet hashes; EVERY
timed request must retain its own frozen 128-token vocabulary logits and exact
initial encoded bytes, including code-replay workspace. The source variant
continues charging actual lossless conversion and roundtrip checks. No changed
generation, no FullKV fallback and no removed numerical validation.

Five warmups and twenty randomized complete requests per method per input,
one A100/12 CPUs/48 GiB per input, at most six aggregate A100s, 75-minute limits.
Retain raw timing and automatic afterany analysis. Judge the unchanged <=1.05x
latency gate using the existing paired intervals and eight-comparison correction.
All quality metrics are inherited ONLY through exact output/state identity;
PPL and ROUGE remain separate, mixed, consumed evidence. No fresh cohort,
calibration, larger sweep or superiority claim follows automatically.
