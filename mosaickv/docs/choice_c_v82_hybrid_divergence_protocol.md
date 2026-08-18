# Choice C v82: hybrid draft forecast with fail-closed divergence

## Scientific boundary

V81 is sealed as a negative calibration. Its samples, generated outputs, references, and task scores cannot be used to tune v82. V82 uses a new disjoint DocVQA/TextVQA calibration split and retains the unchanged 30% memory and five-percentage-point fidelity gates.

V81 showed that replacing the small prototype tier with exact blocks did not improve answer fidelity. It also showed that prompt-window reranking was degenerate: Prompt Top-K and the reranker consumed the same prompt evidence, so all 11,904 candidate exchanges per sample were rejected. V82 changes only the forecast evidence and the fail-closed validation around reranking.

## Hybrid forecast

V82 reuses the original prefill. It clones the FullKV decode state, performs two or four deterministic greedy draft steps, captures their query/attention evidence, and discards the draft tokens and cloned cache. The final decode cache is never mutated by the draft. Hybrid evidence combines the fixed 16-position prompt window with these isolated draft observations.

The two-step hybrid forecast is shared by the prototype control, exact-only control, and conservative reranker. A four-step hybrid arm tests whether a slightly longer future horizon is useful. Draft overhead remains a separately measured forecasting cost.

## Fail-closed divergence

For stable layer/head route (h), prompt block scores (p_i) and hybrid forecast scores (q_i) are separately normalized. V82 records their total variation distance:

\[
\operatorname{TV}_h(p,q)=\frac{1}{2}\sum_i\left|\frac{p_i}{\sum_j p_j}-\frac{q_i}{\sum_j q_j}\right|.
\]

A route is divergence-eligible only when both distributions have positive mass and \(\operatorname{TV}_h\ge 0.01\). A reranking sample fails explicitly if no stable route is eligible. This prevents a repeated v81 outcome from being recorded as an effective future-query method.

Eligible routes still obey the v81 safety rules: equal-size optional blocks only, mandatory blocks retained, unchanged per-head capacity and active bytes, deterministic ordering, and a 99.5% or 99.0% selected prompt-mass floor.

## Calibration and gates

V82 materializes 64 new DocVQA and 64 new TextVQA samples using selection seed 20271219. Materialization excludes all repository ledgers, every v73--v81 input ledger, and prior observed result rows.

Every candidate must satisfy on every task:

- at least 30% lower active KV bytes than Prompt Top-K;
- at least 30% lower active-plus-residual bytes than Prompt Top-K;
- FullKV exact-answer agreement no more than five percentage points below Prompt Top-K;
- prototype logical bytes no greater than 5%;
- complete rows and a clean, identical committed source snapshot.

The prototype and exact controls must also have identical packed active bytes on every paired sample. No held-out campaign exists in the v82 submission chain. A future held-out protocol remains blocked unless the analyzer creates `analysis/frozen_candidate.yaml` after all gates pass.
