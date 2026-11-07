# Guarded precision transactions: bounded, consumed-input diagnostic

The simpler individual-query/residual-delta candidate remains the reference.
No paper superiority or universal six-component benefit is assumed. The prior
error/demand/opportunity combination and signed-restoration utility remain
negative or mixed diagnostics; they are not enabled by this variant.

## Why change the composition rule

Changing exact selection previously also changed the chosen prototype anchor
and members. This made selection effects inseparable from prototype effects.
The new precision transaction fixes the prototype membership and exact count
before trying either change. It begins from the existing six-stage core plan;
the refinement is explicitly additional, not a renamed submodular objective.

1. **Exact-only proposal:** restore one row at a time using its actual signed
   causal attention-output gain, including changed keys and softmax
   renormalization. Select four rows using only the early half of the prompt
   queries. Do not change the prototype cluster or mandatory rows.
2. **Prototype-only proposal:** use one closed-form early-query-demand-weighted
   FP32 residual prototype, preserving phase-neutral key construction. For
   member attention `a_qj` and cluster mass `s_q = sum_j a_qj`, weights are
   proportional to `sum_q s_q a_qj / sum_q s_q^2`. This is a training-free
   compression computation, not benchmark-label or model-parameter fitting.
3. **Individual vetoes:** each proposal must preserve early, late, and maximum
   error in BOTH causal prompt attention and the unmasked future-query proxy,
   with numerical tolerance 1e-12 and strict mean improvement. Prefix halves
   are not independent evaluation examples.
4. **Atomic composition:** apply only proposals passing their vetoes. Recheck
   the combined transaction. If composition fails, return the original plan
   and record rollback. No iterative threshold search or sample-ID rules.

The planner's strict input schema accepts only original prefill K/V, the
parent's reconstructed K/V, prompt queries, original rotary phases, and the
current exact/prototype plan. Answers, references, generated logits and task
scores are prohibited. The saved-capture evaluation wrapper is separate from
this online API. No oracle future queries enter compression.

## Storage and execution

Reuse the same 6,713 bytes of auxiliary GPU allocation; update their contents
before graph capture. Keep original K/V and logical positions intact except
the explicit existing prototype approximation. CPU scratch is released after
construction and must not be called residual-tier storage. All original
source-code owners must retire in the common lossless replacement transaction.
Count the common row map once as shared storage, including encoded and bound
representations. Initial accounting now includes shared phase/position/map
owners, consistent with final/peak accounting; historical private-only initial
records are not silently reinterpreted.

## Preregistered bounded execution

Only the same two consumed COCO inputs are used. Keep pinned model/data,
BF16 eager, 128 predictions, two-prediction provisional scope and the existing
32-token KL <= 0.02, max-logit <= 1.0 and greedy-token agreement 1.0 guard.
Compare exact-only, prototype-only, guarded composition and core ablations at
matching actual auxiliary/source storage. Natural repair thresholds remain
unchanged; forced restoration remains a mechanics control, not quality
recovery. FullKV is only a quality reference.
The no-forecast ablation also disables the query-based precision refinement;
the no-value ablation disables its value-driven exact reranker. Otherwise
those mechanisms would remain active under misleading ablation labels.

CPU proxy feasibility must show an accepted, meaningful change on BOTH
consumed inputs before GPU submission. A model-level failure stops promotion;
do not launch fresh cohorts or larger campaigns on a local proxy pass. Even
successful execution and lower KL do not establish benchmark superiority,
latency superiority, independent quality preservation, or a whole-request
peak advantage. Record every neutral/negative component ablation explicitly.
