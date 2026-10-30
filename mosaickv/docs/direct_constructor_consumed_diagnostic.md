# Direct representation constructor

This optimization preserves selection and every encoded payload format. It
does not change quality policy, references, mandatory tokens, cache bytes or
decode. All inputs are the same two consumed diagnostics; no cohort advances.

The old diagnostic builds K/V, all-source and token-hybrid candidates,
materializes every candidate with full original-shape GEMMs, then constructs
compact candidates for two reduction policies and constructs dual candidates
again. The direct builder computes membership once, constructs the same
payloads directly, and validates the formats actually eligible for transport.
It retains three independent original-K/V bit checks per compact/dual format.
If the full-precision-reduction compact candidate needs zero correction bytes,
the other policy cannot have lower correction storage, so it is skipped using
a mathematical lower bound—not a timing or output-score heuristic. If any
corrections are needed, both policies remain candidates. Shared-source bodies
are reused when high-degree and complete source unions are identical.

CPU differential tests require identical format costs, original K/V bits,
logical memberships and shared-map ownership. GPU controls additionally
require identical original 32-token logits and initial/final/observed-decode
peak storage against the slow reference builder. Five warmups and twenty
randomized paired trials measure representation-construction wall and CUDA
time for Top-K, PrefixKV, VL-Cache and EviCastKV on each consumed input. LOOK-M
is not eligible for original-source regeneration and remains covered by the
separate all-baseline codec/history diagnostic.

Constructor timing excludes original prefill, selection, decoding, and trace
I/O. Do not add medians from different runs and call them measured end-to-end
latency. This is not a paper-native reproduction or a quality improvement.
