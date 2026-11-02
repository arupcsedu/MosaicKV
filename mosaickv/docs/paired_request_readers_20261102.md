# Paired full-request reader diagnosis and bounded follow-up

Only the same two consumed LLaVA inputs; independent quality N = 0.
No new cohorts, larger quality campaign, row/warp sweep, changed selection,
threshold relaxation, prototype/repair claim, or paper promotion.

Before model inference, one short GPU synthetic control verifies the real
Triton compiled-handle launch ABI, repeated-launch bit parity, all BF16 bit
patterns and request-cache cleanup. A failed/missing precheck blocks both
model workers. This control has no model/dataset and is not a quality row.

First run original and carried-literal readers in the SAME engine. On each
input use EviCastKV's source-precision/storage variant and the local PrefixKV
precision/storage adaptation. Each method receives both reader policies, five
warmups and twenty timed paired trials, randomized within trial. Preserve the
same original prefill/packet and all 128 vocabulary-logit vectors, plus exact
initial, final and decode-owned peak storage. These are not byte-matched
algorithm comparisons and the baseline is not official PrefixKV. Full-request
latency retains processor/prefill/construction/retirement/decoding boundaries;
model/disk-media loading, scoring and hashing are excluded. First excluded
warmups check packet hashes. Failures remain in producer records.

AFTER timing, run one separately instrumented request per method/reader. Trace
decode steps 0–15 (early), 16–95 (middle), 96–126 (late); record prompt-source,
text-overlay, generated-history projection/reconstruction, code replay, history
sealing and consolidation. Count the reader branch actually supplied to each
encoded kernel launch (raw, packed counts, original, carried). Require the
requested encoded reader in every window, 16/80/31 decode calls, identical
logits and byte boundaries. Timers are stream elapsed time including host gaps,
NOT isolated active-kernel time; nested scopes must not be summed. Instrumented
requests cannot enter latency intervals. The fully grown-cache helper is not
a substitute for these changing-history traces.

Use 10,000 paired bootstrap draws and Bonferroni eight engineering contrasts
(two policy/method contrasts and two EviCastKV/PrefixKV contrasts per input).
These intervals concern repeated requests, not independent quality examples.

Permit ONE conditional dispatch optimization attempt if both producers are
complete/numerically safe and each EviCastKV diagnostic has at least 0.10 s of
exclusive host projection work. Freeze the common reader to carried only if
its EviCastKV upper ratio bound is <=1.0 on BOTH inputs; otherwise use original.
The candidate caches only compiled projection handles inside ONE request,
using Triton 3.4.0 CompiledKernel.__getitem__. Validation still runs on every
projection. Keys include tensor pointers/dtypes/devices, exact constexpr and
launch options, and scalar types; only explicitly do_not_specialize scalars
may change values. No tensors/decoded caches are captured, no geometry or
arithmetic changes, no opaque global scratch, and all handles clear at exit.
Apply the same optimization to the comparator. Numerical controls and ALL FIVE
method adaptations must preserve logits and initial/final/peak storage before
paired timing. This is an opportunity test, not a promised improvement.

Promote only if the dispatch candidate's adjusted upper latency ratio is
<=0.99 versus its own original dispatcher on BOTH inputs. Otherwise stop.
If promoted, run the unchanged all-five-method five-warmup/twenty-trial request
benchmark, preserving the <=1.05 all-baseline engineering gate. The original
gate is not weakened, and even a pass is not benchmark-quality superiority.
Stop after the single dispatch attempt and conditional final benchmark.

Every producer uses current-process Slurm IDs, keeping inherited identities
under a parent field. Historical raw manifests remain untouched; a separate
hash-bound correction ledger cites sacct for array 22330023. Source, config,
inputs, model, environment and parent artifacts remain pinned. Finite afterany
analysis handles both successful and failed producers without continuous log
polling. Two workers, each one A100/12 CPUs/48 GiB, bi_fox_dgx; at most two
concurrent GPUs in this chain, never above six aggregate. Two-hour maximum per
worker (actual requested limit sixty minutes). All caches remain on scratch.
