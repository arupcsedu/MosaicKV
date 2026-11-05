# Once-only incoming source fan-out

The saved optimized joint-KV profile attributes 0.209 seconds to 4,032 steady
source-input hooks and 0.382 seconds to their corresponding joint-key wrappers.
Those attribution totals overlap downstream operations; they are not removable
latency estimates. Both paths copy identical normalized BF16 input: the source
hook saves history, and the key wrapper saves the joint projection input.

Replace the two independent copy launches with one integer-preserving fan-out,
writing the same source-history slot and same incoming projection buffer.
Use the same compiled binary after its first checked launch. Retain original
source validation, cache observers, byte counts and projection ledger. Add an
ordered, exactly-once CPU stamp (layer, history, exact hidden object/pointer/
shape/stride). Reject outstanding copies, other consumers or modified inputs.
Require the native query module without side-effect hooks: an arbitrary query
hook could mutate input between capture and consumption. Delete the temporary
cache callback and break owner references at retirement. No extra GPU tensor.

This is eligible only where the representation already makes the duplicate
copy. Offer the flag to baselines; report zero engagement when no joint source
exists. Do not fabricate a baseline speedup or add baseline work. Retain common
epilogue/rotary paths for all methods only when each improves every measured
own-control mean on both consumed inputs and passes numerical/storage checks.

Real GPU controls copy all BF16 bit-pattern families, including signed zero,
NaNs and infinities, through 127 rotating history slots, preserving unused
slots. Model requests must retain every frozen parent vocabulary word, K/V,
logical position and all three own-control storage boundaries, with 4,064
fan-outs/consumptions and no FullKV fallback. Two consumed inputs; four arms;
five warmups, three randomized timed repetitions; two one-A100 workers with 12
CPUs and 20-minute limits. Full CPU suite precedes GPU and afterany analysis.

This is a small execution hypothesis, not a promised 4% improvement. Independent
quality N=0; whole-request GPU peak remains uncertified; six-mechanism-disabled
storage variant labels remain unchanged; no fresh/larger campaign is launched.
