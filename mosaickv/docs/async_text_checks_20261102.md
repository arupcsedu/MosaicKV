# Async exact reconstruction verification, consumed inputs only

The completed 54438cf nested study attributes 0.214/0.220 seconds to the three
independent K/V bitwise comparisons, versus about 0.008 seconds to accounting.
Target the measured comparison path, not accounting or selection. This is a
common storage-transport optimization offered to every method.

Keep ALL THREE independent materializations and every original scalar check.
For each pair an integer-only Triton kernel compares the original BF16 bit
patterns, emits one boolean per 4096 scalars, and retains no K/V references.
All flags are reduced once after the third materialization. Live temporary
storage is bounded to small flag arrays, not a full boolean tensor or retained
K/V copies. Exact equality is unchanged, including signed zero and NaN payloads.
No tolerance, numerical patch, source precision, membership, model weight,
generation length, logical position, protected-token rule or output changes.
Failure may be detected after all three passes rather than after the first;
it is still rejected, never converted to a success or FullKV fallback.

Run the complete CPU suite, GPU bit-pattern/strided controls, and both consumed
five-method constructor workers. Every packet hash, physical byte count and
prefill logit must match the original parent. Audit constructor memory during
an excluded first warmup for every method; require lower observed constructor
CUDA peak than every baseline before timing promotion. This boundary excludes
weights, model activations outside the encoder and opaque GPU workspace.

Only after that passes run five warmups and twenty randomized complete requests
per method/input, retaining all 128 vocabulary-logit comparisons and the old
paired 1.05x engineering latency gate. Use two one-A100/12-CPU workers, no more
than six aggregate A100s, and afterany CPU decisions. Preserve negatives.
There are only two consumed examples and zero independent quality examples.
The source-precision variant disables the original six mechanisms. These are
different encoded-memory points, not byte-matched algorithm curves, official
baseline executions, new quality evidence or a paper-superiority claim.
