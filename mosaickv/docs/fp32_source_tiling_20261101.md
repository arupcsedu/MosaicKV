# Preserve FP32 headers; optimize only bit-identical execution

Preserve the completed 30-row compact-input study. Description source compact
headers degraded 128-token reference ROUGE-L although the 32-prefix guard
passed. The native/integrated-input alternatives did not establish a general
quality/memory/latency win. Do not use the compact-header CPU pass as evidence
of answer preservation, and do not tune on those answers or failure identities.

Use the already positive FP32-header conditional source + common lossless
text representation, without changing ANY encoded value, protected position,
cap or history policy. First repeat the registered all32-layer/two-input,
five-warmup/twenty-randomized-CUDA-trial operator study using FP32 headers.
Grouping is eligible ONLY when both components are bitwise identical in all
64 cases and summed median time improves >=10%; otherwise retain group one.
Record exposed compiler register, spill and shared-memory counts, or null if
unavailable. Do not invent a zero for an unavailable hardware counter.

Only after that gate, replay the selected tiling through 33-prefix and
128-greedy model controls. Require bitwise equality with the unchanged
FP32 source/text control, identical initial/final/peak tensor storage and
all previous FullKV/mandatory/position/retirement/isolation gates. Rerun
the four baseline reimplementations/Top-K with both header choices using
the same inputs and GPU controls. Source precision stays FP32, not adaptive
per-dataset header selection. Retain stronger unmodified baseline rows.

This is a same-consumed-input lossless engineering change, not a new quality
cohort. No new data, statistical superiority, same-byte comparison, complete
system/prototype/repair claim or whole-request peak certification. Latency
is single fixed-order full-model repetition (plus separate operator trials),
not paper timing. PPL remains a separate pending diagnostic.
