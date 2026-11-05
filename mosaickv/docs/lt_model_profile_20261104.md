# Conditional actual-model attribution

The full-request gate from `lt-reader-consumed-20261104-v1` remains untouched.
Run this diagnostic only after its two workers and analyzer finish, only when
its registered latency result is complete and negative. A positive gate
suppresses GPU attribution. CPU tests may run beforehand. No runtime changes
are made, and no new identities, quality evaluation or timing retry is released.

Use only the same two consumed model inputs, with the original graph/arena
EviCastKV reader, bounded native reader, and original optimized PrefixKV reader.
Each policy receives a full excluded warmup. Instrument one complete request,
requiring all 128 parent logit vectors bitwise and no fallback. Record host
conversion, initial graph capture, replay and retirement phases, with explicit
nested-phase warnings. These perturbed full-request times cannot enter a
latency table.

CUPTI runs only at actual model steps 1/64/126, after initial graph capture,
so profiler initialization cannot contaminate capture ownership controls.
Require positive-duration actual kernels in every exported trace and all 64
prompt-reader graph intervals in each window. Record raw kernel names,
counts and durations, plus per-reader stream intervals. Kernel-duration sums
are not wall time; stream intervals can include host gaps. Do not infer kernel
cost from annotations alone or conflate constructor, replay, attention and
generated-history costs. Instrumentation may affect host dispatch.

This study is not acceptance, promotion or paper superiority evidence.
Independent quality N=0, whole-request GPU peak remains uncertified. The finite
chain is CPU tests -> negative-parent conditional release -> two one-A100
workers -> afterany terminal analysis. Twelve CPUs per GPU, twenty-minute
worker limits, aggregate A100 cap six, no overlap with parent paired timing.
