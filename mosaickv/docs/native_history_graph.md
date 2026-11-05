# Private immutable-history replay diagnostic

The missing-counter fix preserved bits and reduced prior overhead, but the
complete request remained 0.27--0.29 seconds slower than original history.
Do not promote it alone. The saved profile also attributes substantial host
work to repeated history unpack/GEMM/copy/RoPE dispatch. The native operator
was already fast in CUDA graphs; the runtime replay path had not captured it.

This default-off variant captures only a sealed history packet's exact reader,
never attention, the whole model, eviction, repair policy, or changing tail.
Bind each graph to its request, packet, attention module, K/V component and
fixed target pointer/shape/stride/type/device. Retain owner references until
epoch retirement. Reject changed bindings and more than two live graphs per
layer. Construction and scratch verification still execute directly. Retire
old packet graphs after consolidation checks and before dropping source
owners, and retire all remaining graphs before the prompt pool. Count warmup,
capture, replay and destruction in the full request. Require zero additional
GPU tensor bytes; existing prompt-generator metadata remains charged.

For each real request expect 448 captures/retirements (32 layers x 7 epochs x
K/V), 7,104 real decoding replays, at most 64 live graphs and no retained graph
at completion. Preserve original three history verifications and all 128 parent
logit vectors bitwise. Native source corrections retain the 32-KiB epoch cap.
The complete 192-row owners remain charged when borrowing 128-row views.

First run only audited and one uninstrumented request per arm on the same two
consumed inputs. A complete correctness pass can authorize the registered
five-warmup/three-trial point screen on those same inputs, never fresh quality
data. Compare the retained Source-192 reader and all original/optimized
PrefixKV controls, offering the new common reader to PrefixKV too. A point
screen cannot establish formal four-percent latency or benchmark superiority;
independent quality N=0 and whole-request GPU peak remains uncertified.
