# Native readout handoff, not a selection change

The explicit-capture guard still computes the final normalization and vocabulary
readout for parent and active hidden states. Ordinary generation immediately
recomputes the selected pair. This variant returns the selected native BF16
normalization/readout from the same graph and borrows those real buffers until
ordinary generation consumes them exactly once. The GPU operator gate compares
the selected pair bitwise, including across graph retirement; CPU tests enforce
ordering, ownership and nonconsumption failures. All graph activation backing
is still reported separately from K/V. No weights or decisions change.

When no diagnostic capture is requested, the previous guard also copied every
complete parent vocabulary vector to CPU and immediately discarded that list.
The explicitly registered `export_parent_logits=false` option omits only these
unused diagnostic copies. It still computes every native parent readout, FP64
KL/max/token decision and per-step record. The observed-step counter remains
mandatory, and requesting an actual diagnostic capture with export disabled
fails closed. The reference behavior remains the default.

Prerequisites are a complete exact-byte/native-tail model run and its immutable
logit references. Use the same two consumed inputs with four arms: original
storage-only, common-codec storage-only, reference native-tail hybrid and handoff
hybrid. Both hybrids use identical fast source coding. Eight warmups and 24
timed repetitions use balanced randomized four-arm orders. Eight registered
input/comparison pairs use the existing adjusted equal-or-faster gate. K/V
initial/final/decode-owned peak bytes must match between the two hybrid arms,
and all 128 logit vectors must remain bitwise identical. These are engineering
diagnostics, not independent benchmark quality, whole-request peak certification
or a general benefit of all six mechanisms. No fresh cohort is submitted.
