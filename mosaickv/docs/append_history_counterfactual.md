# Same-selected-prompt append-only teacher-forcing counterfactual

This is an evaluation-only intervention on the consumed MM-Vet input `v1_197`.
It is not a new candidate, fresh evaluation, prior-mixing sweep, or byte-matched
performance comparison. Preserve the existing negative development results.

The frozen FIFO captures are bound to plan
`c74c12fb58ecf38cad5d6a2cec7dfe70e13e16d4327c5662aca5a3516fe24c6a`.
Use only its Prompt Top-K and mixed-prior EviCastKV teacher-forcing arms. Keep
the selected prompt blocks, original K/V, per-layer prompt lengths, mandatory
policy, media, processor, model/revision, BF16 eager arithmetic, logical
positions and teacher targets unchanged. Saved answer-generation trajectories
are reused; only the 16-token FullKV parity control is rechecked before each
new teacher capture.

The registered target has 593 tokens, requiring 592 decode updates after the
original prefill prediction. Raise generated capacity from 256 to 592, so the
existing incremental transport appends every update without replacing history.
Raise the lifetime cap by

`(592 - 256) * (sum(layer_position_bytes) + max(layer_position_bytes))`.

This charges the additional history and the existing one-layer growth-copy
reserve. The pinned uniform LLaVA geometry gives a **planned**, not measured,
cap increase of 173.25 MiB, from 289 to 462.25 MiB. Actual initial, final and
post-compression transaction-peak backing storage must come from the trace.
This peak does not certify whole-request memory, inactive source storage,
allocator reservation or total GPU usage.

Before interpreting losses, require exact selected-block and prompt-capacity
equality, zero padding, identical target IDs and logical positions, no prompt
or generated eviction, and exactly identical NLL/teacher argmax before the
first observed FIFO eviction-affected prediction (258). Independently reconcile
the per-token NLL sum and runtime token count. Any violated invariant fails the
diagnostic and preserves partial evidence; do not loosen it retrospectively.

Report NLL totals and PPL separately for each role, plus NLL differences before
and after the observed eviction boundary. A difference after that boundary
isolates the effect of the history-retention policy conditional on this prompt
selection and teacher trajectory. It does not identify a unique harmful token
or demonstrate benchmark-answer preservation. In particular, EviCastKV's saved
free-generation divergence at token 23 predates history replacement and cannot
be explained or repaired by this intervention.

Use two one-A100 workers, each 12 CPUs/16 GiB host memory/20 minutes, with a
finite `afterany` CPU analyzer and write-once outputs. Remain below six aggregate
A100s. Cache paths are scratch-only. Capture timings include extra observer
synchronizations and are not performance evidence. Do not automatically promote
this diagnostic into calibration or a larger campaign.
