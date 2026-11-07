# Fixed-plan, consumed-input horizon feasibility audit

The actual FullKV draft target passed bitwise controls, but its one-prefix
joint selector did not improve mean FullKV KL over the simpler prototype
candidate. Restoration-only improved mean KL, while combining it with an exact
swap was less consistent. Do not expand its cohort or claim superiority.

This diagnostic changes NO candidate pool, byte cap, scoring weights or guard.
Freeze the ten existing plans from each consumed input's completed FullKV-draft
study. Execute each complete 128-prediction decode with isolated state and the
same storage reader. Explicitly label enumeration and restoration as controlled,
evaluation-only interventions, never natural uncertainty-triggered recovery.
Identity must reproduce the unchanged candidate bitwise. All proposal IDs,
mandatory positions, quantizer headers, K/V data and physical storage must obey
the existing controls. Any failed or missing plan remains in the record.

After every GPU producer finishes, a CPU analyzer may use the FullKV evaluation
logits to ask whether selecting among these SAME plans using the first 1, 2,
3 or 7 draft updates would predict lower error on later shared prefixes.
Exclude prediction zero: it is unchanged prefill, not forecasting evidence.
Keep the 32-token KL <=0.02, max-logit <=1 and token-agreement-one gate.
Never compare logit errors after generated input prefixes diverge.

This is an evaluation-only oracle-feasibility study. It is NOT an online longer
forecast implementation, not independent validation, not benchmark-label fitting,
and not evidence that a selected plan generalizes. The analyzer module must not
be used by the online selector. No reference captions or scores enter this
audit. Report all horizons and both inputs, including negative outcomes.

A useful result would justify implementing a corresponding bounded online
forecast on these consumed inputs, with its own parity and runtime controls.
If none consistently improves later-prefix fidelity, stop this substitution
direction rather than increase the beam or tune against sample answers.
Fresh-cohort evaluation and larger campaigns remain blocked. Temporary CPU
FullKV snapshots and readout workspaces retain their explicit cost accounting.
