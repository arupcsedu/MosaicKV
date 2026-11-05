# Prepared graph replay and once-only joint-key consumption

The preceding K/V screen passed numerical/storage controls and was about 1.1–1.4%
faster than its fastest measured optimized PrefixKV control, but did not reach
the registered 4% threshold. No formal/quality claim follows from three trials.

Its saved cProfile records 12,096 reconstruction calls for 8,128 graph replays:
ordinary K consumption re-enters the complete format walker after the producer
has already reconstructed K. Cache only CPU scalar replay metadata at initial
capture. Preserve packet/attention/buffer/stride signatures, stream-capture and
lease guards, once-only ordered XOR replay, native dispatch counters and graph
retirement. Consume preproduced K through the same checked layer/epoch journal
before calling the generic reconstructor; still fill all actual history slots.
Keep original paths default-off. This does not change arithmetic, source bits,
positions, selection, dtype, mandatory evidence or GPU workspace.

Offer the prepared replay path to both PrefixKV graph readers as well. Compare
seven arms: EviCastKV fused, joint legacy/prepared; PrefixKV optimized and
original, each with legacy/prepared replay. Require identical actual initial,
final and decode-owned peak storage against each arm's own control, all 128
parent logit vectors bitwise and exact branch counters. Five warmups, three
randomized paired complete-request trials, two consumed inputs, two one-A100
workers, 12 CPUs each, 20-minute limits, full CPU suite before GPU release and
automatic afterany analysis. Three-repeat point estimates are not formal gates.

The predecessor retained a stale `only_immutable_prompt_captured=true` field in
joint graphs. Their current-input tail GEMM also produces incoming K/V. Correct
the new field and retain a hash-bound derived predecessor-scope review without
altering raw results or their timing/storage/token/logit measurements. Joint
keys reuse only K; joint-KV reuses both. Neither captures attention/softmax or
complete transformer layers. Default prompt-only controls remain labeled true.

The six-mechanism-disabled storage variant remains separately labeled;
independent benchmark N=0, no byte-matched algorithm superiority claim, and
whole-request GPU peak is uncertified. No fresh or larger campaign is launched.
