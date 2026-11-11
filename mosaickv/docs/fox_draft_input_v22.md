# V22: decision-aware source-precision feasibility, not a paper campaign

V21's funded original-input rows reduced logit KL but still changed greedy
predictions. Thus prompt attention multiplied by input reconstruction error is
not a sufficient source-fidelity policy. Preserve all V20/V21 results.

This single successor keeps the original 49 MiB initial cap, frozen source
packet, same two consumed inputs, mandatory original K/V, native projections,
raw append-only history, and unchanged 32/128-prediction guards. It changes
only the prospective ranking of additional original-input rows. No grid, new
identities, benchmark annotations, or larger evaluation is authorized.

## Prospective signal

Reuse the original prefill for four private FullKV greedy draft steps. Run
the native-quantized parent over those same four generated draft tokens using
independent differentiable K/V cache clones. Read derivatives with respect to
the cache only; there is no optimizer, training, learned parameter, or model
weight update. Leave parameter flags, gradients, cache state, and logical
positions unchanged. Inference-tensor model weights are explicitly unsupported.

For each draft prefix, minimize KL divided by the registered 0.02 bound plus
the positive deficit of the parent's reference-token margin relative to FullKV.
The margin is the reference greedy logit minus its strongest competitor. Sum
four prefix losses and perform one cache backward pass. A proposed whole-row
restoration is scored by minus the cache gradient dotted with its original-
minus-parent K/V change, summed over both components and all heads.

This is a signed, downstream-model signal rather than unsigned input MSE.
Negative/zero estimated gains cannot receive optional capacity. Positive gains
do not guarantee actual nonlinear improvement: the unchanged model-level guard
still decides whether this consumed-input diagnostic passes.

One short rollout/backward scores all source rows. It does not run a separate
candidate forward for every row or access the saved 128-token oracle during
selection. New draft logits must match the corresponding first four native
inference logits bitwise. Saved 128-prefix tensors are loaded only after the
candidate has been constructed and are evaluation-only.

## Storage and limitations

The lossless text-only transaction remains a control. Only its measured net
unique-owner savings may fund the ranked original rows. All unused three-bit
codes, numerical patches, headers, row metadata, history, and observed decode
workspace remain charged. Forecast temporaries/overhead are reported separately;
their whole-request KV peak is not certified. The original six-mechanism
pipeline remains disabled; this is an explicitly distinct source-precision
variant with a short model-draft scoring component.

V20 genuinely matched initial/final/decode-owned peak bytes for the common raw
exact-selection transports. Full-prompt source versus 98-position Top-K is
still **unsupported for all-three-boundary equality**: source reconstruction
needs 600/604-position leases and additional native workspace. Lowering initial
bytes cannot repair equality at all boundaries. Unused baseline padding and
excluding already-counted workspace are forbidden. A common bounded-tile
attention/reader would require a separate implementation and numerical gate.

The finite launcher executes the full CPU suite with registered in-memory
capture, then two one-A100/12-CPU workers and automatic `afterany` analysis.
Nothing here establishes independent quality preservation or superiority.
