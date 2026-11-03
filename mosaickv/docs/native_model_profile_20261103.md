# Actual-model CUDA attribution after negative operator screening

Preserve every failed gate, including prepared MM-Vet layer-0/padding-0 ratio
approximately 0.901 > 0.900. No rounding, threshold change, retry-for-significance,
or full-request benchmark promotion. The new stage is a DIAGNOSTIC, not a way
around that gate. Repeated isolated-layer results disagree with prior full-model
timings; diagnose actual GPU kernels before any further optimization attempt.

Use only the same two consumed identities, pinned model/precision and original
reader controls. One excluded 128-prediction request warmup, then one instrumented
128-prediction diagnostic per policy: original and prepared native. Record CPU
and actual CUDA/CUPTI activities only INSIDE decode calls 0, 64 and 126, with
explicit prompt reconstruction/native BMM/unpack/restore scopes. Keep all other
decode steps unchanged and verify all 128 parent vocabulary logits bitwise,
packet hashes and native plan activity. History sealing before decode entry is
outside these selected windows and must not be called measured by this profile.

Retain Chrome traces, input-shape-aware operator rows, actual CUDA device times,
CPU scope times, initial/final cache bytes and complete provenance. Remove
instrumented request duration fields rather than letting them enter latency
tables. Do not infer pure GPU kernel cost from earlier CUDA-event intervals
containing host gaps. Fail if CUPTI produces no actual CUDA activity. This is
neither quality validation nor audited whole-request cache/GPU peak evidence.

Finite afterany CPU preparation -> two one-A100/12-CPU/48-GiB workers -> CPU
kernel/host ranking. No automatic optimization, full-request timing benchmark,
quality sweep, new identities, or paper claim. At most two simultaneous GPUs,
six aggregate maximum, reservation bi_fox_dgx. Common scratch-cache environment;
clean immutable SHA; CPU preparation up to fifteen minutes, GPU workers up to
sixty minutes, CPU analysis ten minutes. No continuous polling.

Independent quality N=0; this remains a six-mechanism-disabled storage/precision
variant. Baselines are local adaptations, not official methods or byte-matched
algorithm evidence. The original reader remains default.
