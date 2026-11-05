# Common generated-history native projection: feasibility only

The saved actual-model CUPTI trace attributes 14.6 ms to 64 encoded-history
projection kernels at a middle decode step, above the native prompt GEMM
families. Instrumented kernel sums are not critical-path request timings.
This is a better-supported target than another geometry sweep.

Test the existing lossless 16-token-epoch source materializer, then a bounded
128-row BF16 input/output workspace and installed zero-workspace FP32 native
GEMMs. Preserve original RoPE operations and store real sparse differences from
the old projector. Verify BF16 output words and strided history sentinels, both
eager and captured, at every 16..112-row epoch. Five warmups and twenty paired,
randomized CUDA-event trials compare original and revised operator graphs.
Count both source/output buffers and every correction index/value. Native
algorithms are first eligible installed heuristics, not a timing search.

No model, dataset or new identity is accessed. No cache format or runtime is
modified by this experiment. A positive operator result is not latency or
quality superiority; real-model all-word checks, constructor costs, lifetime
and physical workspace accounting would still be required. Any implementation
must apply the history optimization identically to all compared readers.

The first default-heuristic run failed its unchanged 32-KiB epoch repair cap.
Preserve it as negative, not a timing result. A separate operator-only follow-up
tests the already all-layer-audited algorithm/configuration 31 (not a new timing
search), with actual feasibility checks at all seven history row counts. Keep
the same random inputs and repair cap. If rejected, save the epoch, per-component
changed-word counts/error and exact native algorithm attributes; do not hide
the failure behind an uninformative exception or expand runtime eligibility.

The measured fixed-kernel failure isolated an invalid synthetic RoPE fixture:
32,682 key words but only 91 value words changed at 16 rows. Its independently
random 128-wide phase halves do not obey Llama's duplicate-half rotary layout,
which the native reader requires. Correct the fixture to the actual shared
64-frequency sine/cosine construction and reject incompatible phases before
encoding. Preserve the invalid-fixture artifacts; they are not evidence that
native history on Llama is numerically or physically eligible. Rerun only this
checkpoint-free operator test, with the same repair cap and no runtime change.
