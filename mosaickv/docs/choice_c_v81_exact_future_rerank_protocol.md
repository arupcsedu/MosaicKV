# Choice C v81: same-byte exact decomposition and future-query reranking

## Status and scientific boundary

V80 is sealed as a negative label-blind calibration. It passed the unchanged memory gate but failed the unchanged fidelity gate, and it did not create a frozen candidate. Stable-head visual swaps are therefore retired. V80 samples, generated answers, and task labels cannot be used by v81.

V81 is a new, preregistered, disjoint calibration. Candidate construction uses no benchmark reference answer or official task score. Selection may compare each candidate's generated tokens with FullKV and Prompt Top-K outputs, which is model-output fidelity rather than benchmark-label tuning.

## Question 1: are the prototypes responsible for the fidelity gap?

The prototype control is the unchanged v80 no-swap cache. The exact-only control uses the same model, prompt, media, four-token blocks, mandatory-token policy, 27,500,000-byte cap, per-head physical capacities, forecast, graph, utility, and stable Prompt Top-K routing. It changes only the tier representation:

- prototype control: at most two RoPE-safe prototypes in each routed head;
- exact-only control: no prototype, residual, or repair tier; the available routed-head capacity is filled with exact source blocks.

The analyzer requires the prototype and exact-only controls to have identical packed active KV bytes for every paired sample. It reports answer fidelity, token fidelity, prototype bytes, and residual bytes separately. A mismatch in packed bytes is a hard protocol failure, not a result.

## Question 2: can forecast evidence improve stable exact blocks?

Prompt Top-K first constructs the stable-head exact cache. The future reranker may exchange an optional selected block with an optional unselected block only when both contain the same number of positions. Mandatory blocks are never donors. Therefore each replacement preserves:

- selected position count;
- active KV bytes;
- dense per-head capacity;
- original logical positions and next-token position.

For a stable layer/head route, let the selected Prompt Top-K anchor be (S_0), prompt attention be (p_i), and forecast attention be (q_i). For a donor (d \in S_0) and equal-size recipient (r \notin S_0), v81 computes

\[
g_q = \frac{q_r-q_d}{\sum_j q_j}, \qquad
\ell_p = \frac{\max(0,p_d-p_r)}{\sum_j p_j}.
\]

The replacement is eligible only if (g_q>0), (g_q>\gamma\ell_p), and the updated selected prompt mass remains at least a preregistered fraction of the original selected prompt mass. Candidates are ordered deterministically by surplus, gain, loss, and immutable node IDs.

Two fixed arms are tested:

- conservative: at most two replacements per stable head, 99.5% prompt-mass floor, \(\gamma=1.5\);
- wider: at most four replacements per stable head, 99.0% prompt-mass floor, \(\gamma=1.25\).

The trace records every accepted donor/recipient pair, prompt-mass preservation, normalized forecast gain, normalized prompt loss, candidate count, and rejected count.

## Data and gates

V81 materializes 64 new DocVQA and 64 new TextVQA validation samples with selection seed 20271211. It excludes every repository ledger, v73--v80 inputs, and prior observed result rows.

The gates are unchanged:

- at least 30% lower active KV bytes than Prompt Top-K on every task;
- at least 30% lower active-plus-residual KV bytes than Prompt Top-K on every task;
- FullKV exact-answer agreement no more than five percentage points below Prompt Top-K on any task;
- no more than 5% prototype logical bytes;
- all rows complete.

No held-out campaign may be submitted unless one v81 candidate passes every gate and `analysis/frozen_candidate.yaml` is created from a clean committed source snapshot.
