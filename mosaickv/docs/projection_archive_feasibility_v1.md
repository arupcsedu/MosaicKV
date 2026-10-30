# Shared projection-source archive: consumed-input feasibility only

Repeated prompt/donor changes failed their unchanged numerical controls. The
shared BF16 codec preserves values and reduces storage, but benefits Top-K
almost equally. This follow-up asks a different, label-blind representation
question before implementing another decoder or running another cohort.

In the installed LLaVA/Llama MHA path, both original K and V linear projections
consume the same normalized BF16 input. For selected sets `S_h`, archive that
input at `U = union_h S_h`, rather than retaining separate K and V payloads.
Its unencoded source payload is `|U| * d_model * 2` bytes, whereas selected
BF16 K/V cost `2 * sum_h |S_h| * d_head * 2`. These are planning formulas, not
measured runtime reductions. Cross-head overlap changes their relative cost.
Do not infer savings from a nominal retention ratio alone.

Read-only CPU inspection of the two already consumed selections found greater
cross-head sharing for the mixed-prior EviCastKV selection. This is a reason to
test feasibility, not a claim about task quality. No answers, references,
generated answer identities or failure labels choose the archive.

## Exactness is empirical, not merely algebraic

Capture the actual common input using scoped prefill hooks. Verify that the
installed `LlamaAttention.k_proj` and `.v_proj` consume the same input pointer,
exactly once per layer. Pin Transformers 4.55.2 and its actual source hash.
Reconstruct a full original-shape zero canvas and scatter the archived union
rows into their original positions. Call the original unchanged projection
modules with the original GEMM shape, then the installed `apply_rotary_pos_emb`
with original cos/sin and finally gather the unchanged selected positions.
Zero filling is **only** for unselected rows of a token-wise linear projection;
it is not fed through a transformer layer or used to drop evidence implicitly.

Changing GEMM shape, alignment or kernels can change BF16 results despite
algebraic equality. Require bitwise K and V equality at every selected position
of every layer; any failure stops this prototype without relaxing tolerances.
No weights, selected positions, keys, values, logical positions, attention
backend, prototypes or residual repair are modified.

## Fairness, cost and limitations

Apply raw-source and losslessly encoded-source formats to BOTH Top-K and
EviCastKV. Charge normalized inputs, GPU union indices, head-selection maps
and retained rotary tensors, deduplicating only true shared storage. Compare
these **component** sizes with the same lossless K/V codec. Also report a
conservative per-layer smaller-format envelope, charging the complete rotary
metadata for every archive decision and deduplicating its true shared backing
in the final component sum. This envelope is not an installed hybrid decoder.

The study does not measure active-cache savings, construction-inclusive peak,
whole-request HBM peak, decode latency, PPL, ROUGE-L or CIDEr. Projection
recomputation can be substantially slower; report its synchronized one-shot
cost as a feasibility diagnostic, never as a warmed request benchmark.
Full original prefill and capture remain alive during this test.

Related work includes [Palu](https://arxiv.org/abs/2407.21118), which reconstructs
K/V using low-rank projections. This study instead retains original normalized
inputs and uses the unmodified full projections; it is neither a Palu
reimplementation nor an established novelty claim. The original source/latent
recomputation principle is not claimed as new.

Two one-A100/12-CPU workers, ten-minute limits, same consumed identities,
scratch-only common environment and finite `afterany` analysis. Fresh cohorts,
calibration and paper-scale execution stay disabled. A successful result
would justify an isolated model-level decoder control next, not superiority.
