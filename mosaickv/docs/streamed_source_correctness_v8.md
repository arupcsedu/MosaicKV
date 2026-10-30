# Original-shape streamed projection: separate, fail-closed engineering variant

V6 compact projection failed original K/V bit checks at layer 3 on BOTH
consumed inputs (array 22187272). CPU analysis 22187273 preserved the partial
proofs. Finite controller 22187323 stopped and submitted no V7 timing. Do not
waive this failure, silently change the tolerance, or call the compact path
validated. Original-shape reconstruction remains the default.

V8 preserves the original GEMM shape and arithmetic. It changes only the
execution of independent projections and elementwise operations:

1. Decode source rows and scatter the identical full-shape canvas.
2. Run original full-shape K projection.
3. Fuse selected-position RoPE/gather, explicitly rounding EACH multiplication
   to BF16 before the BF16 sum, as in installed eager PyTorch. Disable FMA fusion.
4. Release full K before running the original full-shape V projection.
5. Gather unchanged V entries and release the canvas/full V promptly.

When workspace instrumentation is disabled for timing, do not build its unused
tensor-identity exclusion set. Instrumented memory runs retain the full audit.

No selected position, weight, phase, mandatory token, generated-history rule,
precision, model or attention computation is changed. Both selectors receive
the same path. A synthetic CUDA control first covers ordinary and stride-zero
maps, irregular original positions and signed zero. Then ALL original K/V
bits must match across 32 layers × two selectors × two consumed inputs before
installed decoding. Any failure stops the path. Retention-one and 32-token
own-greedy/logit parity remain mandatory. Count transient backing with the
same operator-lifetime audit, including any new kernel output.

Only a complete V8 gate authorizes V9 randomized five-warmup/twenty-repetition
timing from the SAME SHA and consumed inputs. No fresh cohort, calibration,
larger campaign, unchanged-guard waiver, whole-request peak certification,
independent benchmark-quality claim or superiority claim follows.

The CUDA kernel is experimental and validated only within this audited model
path; the CPU serial fallback is a correctness fixture, not performance evidence.
Two one-A100/12-CPU workers, ten-minute correctness limits, twenty for timing,
finite CPU afterany analysis, scratch-only common environment, six-A100 total cap.
