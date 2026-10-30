# Isolated EviCastKV storage candidate

This branch composes engineering components without changing any frozen
selection memberships or claiming a new benchmark result. It is an experimental
HF eager, BF16, full-MHA LLaVA-1.5 transport, not a production CLI/backend release.
No vLLM, SGLang, other model, prototype or residual-repair support is inferred.

Persistent prompt storage uses an explicitly lossy3-bit visual K/V representation
with per-channel time groups for keys and per-token feature groups for values.
All selected text, first four/last eight selected positions, and first two/last
two layers remain exact. Code payloads contain no time-group padding. Encode
the protected exact values losslessly; an integer-only scatter restores their
original BF16 bits, including all bit patterns, into the ordinary HF layer lease.
Quantized visual values are unchanged by this additional lossless coding.

Generated history retains every token losslessly through the validated direct
source constructor and consolidated history packets. Three original-bit checks
remain. Original logical positions and actual RoPE phases are preserved. Source
and exact-code headers, mappings, sparse bit corrections, tails, old/new overlap
and observed tensor workspaces are counted. Whole-request HBM peak remains
uncertified because retained diagnostic prefill/capture and model activations
are outside the explicit post-construction decode-owned backing boundary.

The pointer-table history shader failed actual128-token numerical controls and
is not enabled. Checks-only scheduling passed bit parity but is not selected
without its own latency verification. Neither an unsupported shader nor CUDA
graphs are silently substituted into the candidate.

Run the full CPU suite from the clean merge SHA. Two same-consumed-input canaries
must pass: ordinary retention-one identity, CPU/GPU integer-codec controls,
dequantize/repack32-token identity, exact protected encoding128-token bit identity
relative to the existing3-bit variant, the unchanged32-token KL/logit/token guard,
and reduced measured initial/final/observed-peak storage. Only then release the
finite12-worker common baseline experiment, at most6 aggregate A100s. Score
EOS-aware FullKV-reference ROUGE-L separately at32/64/128 and report all outcomes.
Numerical preservation of32 predictions is not independent answer-quality proof.
No fresh cohorts, calibration or paper-scale campaign are automatically enabled.
