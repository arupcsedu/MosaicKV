# Consumed-input numerical kernel compatibility (not a runtime promotion)

The preceding joint-row operator study failed exact incoming-row parity on both
consumed inputs. No combined projection is enabled. A saved CUPTI event for the
original `[1,4096] @ [4096,4096]` projection identifies the kernel name
`ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn`.
This is a geometry hint, not a documented mapping to a public algorithm ID or
proof that its `sliced1x2` means public split-K=2.

This separate diagnostic queries installed cuBLASLt IDs and checks at most 32
unique FP32-accumulating, zero-external-workspace configurations. The first six
vendor heuristic results are followed by configurations explicitly set to the
observed tile/stage/inner shape; only reported custom options 0 through min(4,
reported maximum) are considered. Every configuration must pass vendor checks
for both row counts before execution. API/header hashes, rejected statuses,
configurations and every changed word count are retained. Aligned disjoint
buffers and request-private descriptors are mandatory.

Only layer 0 K/V at decode updates 1,64,126 is observed. All original operators
and the complete 128-vector model trajectory remain unchanged. One incoming or
prompt word difference rejects a configuration. An exact result would justify
further all-layer operator checks, NOT integration, quality, or latency claims.
The prior negative results remain immutable. No parameters, numerical bounds,
benchmark labels, samples, baseline algorithms, or model weights are changed.
There is no fresh cohort or automatic promotion. Two one-A100 workers follow a
complete zero-failure/zero-skip CPU suite and automatic completeness analysis.
The diagnostic records its 2 MiB source/output workspace, but does not certify
temporary/vendor resources or whole-request GPU peak. All caches stay in scratch.
