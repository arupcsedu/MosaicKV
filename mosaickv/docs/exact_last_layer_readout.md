# Last-layer exact-readout engineering gate

The guarded prompt-output refiner passed local proxy tests but did not improve
mean model-logit KL over the simpler prototype-only candidate. Do not promote
it or expand to a fresh cohort. This gate tests a different, falsifiable
mechanism: a counterfactual through the actual frozen final-layer suffix.

Only LLaVA-1.5's last layer (31), batch one, BF16 eager, uniform full-prompt
storage and the winning reader are supported. Earlier layers are unchanged
by these one-head cache interventions. At the first shared greedy decode
prefix, capture the actual post-RoPE query, current K/V, pre-overlay routed
K/V, attention-head outputs, residual and projected attention. Snapshots are
CPU-owned and temporary. Read-only observers use the existing transport
interfaces; they do not replace arithmetic or retain GPU cache owners.

For a candidate head output, recompute the ENTIRE original output projection
(not an FP32 delta approximation), residual addition, post-attention RMSNorm,
MLP, final RMSNorm and vocabulary projection. Use the original batch-one
geometry. This is neither a JVP nor an assumption that nonlinear downstream
operators are unchanged. It applies only where there are no later transformer
layers. Model weights and training state cannot change.

## Registered controls

1. Recomputed actual attention-head outputs must match bitwise.
2. The original output projection must match bitwise.
3. No-change suffix logits must match the actual model bitwise.
4. Verify identity, original-prototype-member restoration and one deterministic
   equal-capacity exact substitution against three actual isolated one-update
   model probes. No second prefill is allowed.
5. Every vocabulary logit must match bitwise, with identical greedy tokens.
6. Restore all original overlay IDs, values, promotion flags and reservations;
   final 128-prediction output must match the uncontrolled reference bitwise.
7. Preserve original logical positions, exact capacity, mandatory rows, source
   retirement, storage accounting and the existing 32-token stability guard.

Tolerance is zero. On a failed control, save the first divergent tensors,
keep the method unsupported for ranking, and diagnose the operator. Do not
loosen the test or disguise the approximation as exact. Restoration in these
controls is not a natural repair event or benchmark-quality recovery.

For diagnosis only, compute a routed-head full-precision conditional target:
restore that head's original prompt K/V while keeping the actual draft query,
other heads and generated state unchanged. This is not global FullKV and is
not benchmark ground truth. It uses only original prefill evidence, never
answers, task scores, oracle future queries or full-reference decode logits.
The evaluation script may compare saved readouts afterward; those comparisons
must not feed the online planner.

## Scope and stopping

Use only the same two consumed images. Keep frozen revisions, prompts, media,
128 predictions and source-replacement controls. A successful gate would permit
a separately registered bounded readout-ranked proposer on these same inputs;
it would not establish superiority, validate every mechanism's benefit, or
authorize a larger quality campaign. Fresh/calibration and paper campaigns
remain blocked. Single-call timings include capture and verification and are
not latency-superiority evidence. Whole-request KV-peak certification remains
separate from the decode-owned counter.
