# Complete-request K/V transaction timing, with excluded phase attribution

Proceed only from the passed K/V A-B-A artifact: all 128 vocabulary vectors
bitwise, 4032 folded warm keys, 32 original cold keys, 4064 actual value commits,
no fake raw V, no fallback and exact owner retirement on both consumed inputs.
No runtime module differs from that passed source. No thresholds are loosened.

Use the same five-reader paired screen: EviCastKV fused/native/joint-KV and
PrefixKV original/optimized graphs. All receive prepared patch dispatch and
event-sum accounting. Fresh actual packet/three-boundary storage audits precede
five warmups and three timed complete requests per arm with randomized order.
Native construction/copies/capture/retirement are included, not amortized away.
The 4% point screen is not the adjusted 20-repeat/every-baseline acceptance gate.

Before warmups give every arm exactly one additional excluded auxiliary
request. Separately cProfile the joint-KV and optimized PrefixKV requests with
the existing nonoverlapping constructor/steady-decode profiler; the other three
arms receive one uninstrumented auxiliary request. These profiles are not timing
trials or statistical evidence. They identify remaining native-construction and
decode host costs if the same full-request gate still fails. Store raw .prof
files and their hashes, separate from randomized trials.

Two one-A100 workers (12 CPUs each, 20-minute limit) follow a full zero-failure,
zero-skip CPU suite and automatic afterany analysis. Same consumed inputs,
independent benchmark N=0; no fresh/large campaign or whole-request GPU-peak
claim. Preserve both positive engineering checks and negative timing results.
