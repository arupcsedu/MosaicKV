# A representation change, not another exact-substitution successor

Status: experimental numerical core and consumed-capture CPU feasibility tool.
There is **no HF/vLLM integration or new benchmark-quality result**. Existing
`mosaickv_exact`, paper results, and running-job sources are not modified.
The local method label is `attention_matching_protected_reimpl`, never official
code and never a silent replacement for exact retention. If later incorporated
into MosaicKV, it must be an explicitly named representation ablation/extension.

## Why stop the current strategy?

The completed V9 diagnostic at
`/scratch/djy8hg/runs/mosaickv/bounded-downstream-v9-20261026-v1/analysis/`
accepted none of sixteen proposals on either consumed input. Real downstream
derivatives and exact replay worked; the prospective ranking screen and the
unchanged 32-prefix stability test did not. Its decision was
`stop_substitution_strategy_no_automatic_successor`. Preserve V7–V9 as negative
diagnostics. This is not proof that every exact subset is impossible, but it
does not justify another cohort with a similar substitution ranker.

Eviction discards numerator information **and** changes the softmax denominator.
An exact cache cannot store a missing value's information anywhere else. This
variant therefore changes the representation rather than optimizing the same
one-head swap again: keep representative original keys, retain protected values
exactly, and let explicitly synthetic unprotected values summarize discarded
information. A per-key additive log bias restores some lost attention mass.

Attention mass/output matching is **prior work**, not MosaicKV novelty:
Zweiger, Fu, Guo, and Kim, *Fast KV Compaction via Attention Matching*,
[arXiv:2602.16284v2](https://arxiv.org/abs/2602.16284v2), §§3.2–3.3 and Appendix A.2.
This is an independent local implementation with mandatory-entry protection,
explicit retained-history terms, bounded minimum-change value fitting, and
all-or-nothing rollback. No upstream code is copied or presented as official.

## What is constructed?

For one KV head and a fixed original-key subset, let:

- `A[q,j]` be FullKV-normalized attention probability on retained prompt key `j`;
- `h[q]` be the mass on the unmodified retained generated-history entries;
- `n[q,:]` be those history entries' attention-weighted value numerator;
- `Y[q,:]` be the complete FullKV attention output;
- `V0[j,:]` be the original retained values;
- `P` be the actual mandatory/protected positions.

Evidence may come only from online prompt queries or an isolated deterministic
draft that reuses the original prefill. No dataset reference, benchmark score,
oracle continuation, optimizer, parameter update, or learned adapter is an input.
Cache coefficients depend on each request, like a numerical cache-construction
operation; there is no persistent cross-request fitting or calibration training.

### 1. Reconstruct mass with positive multiplicities

For fixed original keys, solve a bounded projected least-squares problem:

\[
 \min_w \|Aw+h-\mathbf 1\|_2^2+\lambda_m\|w-\mathbf 1\|_2^2,
 \qquad w_j=1\;(j\in P),\quad 1\le w_j\le W_{\max}\;(j\notin P).
\]

Store `b_j = log(w_j)` in FP32 and add it to that key's attention score.
Ordinary generated-history tokens have bias zero. The implementation has a
fixed iteration bound, tensor-batched independent heads, deterministic ordering,
and rechecks the actual stored bias after casting. Bounding multipliers avoids
giving a nearly invisible key an unbounded influence.

### 2. Reconstruct values with a protected exact core

Let `d = Aw+h`, `X = A diag(w)/d`, and `U` be the unprotected columns. Solve:

\[
 \min_{\Delta V_U}
 \|X_U\Delta V_U-(Y-XV_0-n/d)\|_F^2+
 \lambda_v\|\Delta V_U\|_F^2.
\]

Protected corrections are exactly zero. A rank-truncated, regularized SVD
computes the minimum-change correction. Cast to the original cache dtype,
measure the resulting reconstruction error and value norms, and reject the
**entire bias/value transaction** if it is nonfinite, too large, or fails to
improve the reference-query error. Rejection returns the original retained K/V
and zero biases. It does not pretend that the compressed subset equals FullKV.

`compact_head` returns a `CompactedHead` carrying `requires_model_validation=True`.
An algebraic success is deliberately not an online admission or quality pass.
At retention 1.0, the caller must supply every original source position and
no fitting/transformation is performed. Disabled mode also performs no fit.

### 3. Preserve positions and keys without phase averaging

Selected keys are copied bit-for-bit, including their existing RoPE phase.
Original logical positions stay separate from physical compact slots. No
different-phase keys are averaged, no fake position is assigned, and protected
K/V values and their zero bias remain bitwise unchanged. Unprotected values
are **synthetic summaries**, not exact retained entries and not validated
MosaicKV prototype merges. Residual repair is not part of this experiment.

## Memory boundary and fairness

The capacity planner charges dense K/V backing allocations, the FP32 key-bias
array, and a preallocated FP32 per-query-head additive mask. It reserves generated
history before prompt capacity. All heads/layers currently have uniform dense
capacity; no claimed savings depend on ignoring ragged padding.

For LLaVA-1.5 (32 layers, 32 KV heads, dimension 128, FP16 K/V, 16 history slots),
**planning arithmetic, not measurements**, gives 39 prompt slots within a 28 MiB
budget: 27.5 MiB lifetime K/V, 0.15234375 MiB bias, and 0.21484375 MiB mask,
total 27.8671875 MiB. The next dense slot does not fit. Any additional persistent
runtime allocation must be charged before claiming a measured cap. CPU metadata,
compression workspace, the initial FullKV prefill, draft snapshots and peak total
GPU allocation must be reported separately. This arithmetic does not establish
lower global GPU peak memory or answer-quality preservation.

Every compared method must run on the same registered budget grid, with the same
history policy, inputs, model, precision, generation, and backend. At shared
budgets compare quality and latency; only curve-derived equal-quality points
can establish lower required memory. Never label unequal-budget points byte-matched.
Include an attention-matching control using a simple Top-K key dictionary as
well as a MosaicKV evidence-selected dictionary. Otherwise generic attention
matching gains could be falsely attributed to MosaicKV's selection policy.

## What the saved-evidence CPU audit can and cannot establish

`audit_attention_compaction_mass.py` reads only the two hash-bound, already
consumed V8 FullKV draft captures. It audits deterministic mean-draft-attention
key dictionaries, mass fitting, and numerical rank over all 2,048 layer/head
routes. It does not access benchmark annotations, run a model, or select new IDs.

The original full K/V and real mandatory-position map were **not saved**.
Consequently, first/last positions are explicitly only protected stand-ins.
The audit cannot measure value-correction safety, actual constructed-cache bytes,
logit stability, free-running quality, or the frozen MosaicKV selector. Rank and
mass-fit improvements are feasibility observations, not paper comparison results.

CPU invocation (a clean scoped source SHA is required):

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/audit_attention_compaction_mass.py \
  --output /scratch/djy8hg/runs/mosaickv/attention-compaction-v1-20261027-v1/cpu-audit
```

## Next engineering gate, not another cohort

Only if the numerical feasibility is worth pursuing, implement an **isolated**
HF bias-transport adapter. Do not bypass the current exact-only bounded-history
safety check or claim that synthetic values are exact. First prove zero-bias
and retention-1 parity, unchanged mandatory entries and logical positions, and
actual auxiliary/backing-storage accounting on synthetic and consumed inputs.

Then capture original source K/V for only the same consumed inputs and test
bounded construction plus the unchanged 32-token clone/repack and logit guard.
Keep reference/draft fitting, construction, transport and decode timing separate.
Fail closed on excessive value norms, poor conditioning, nonfinite output or
unsupported kernels. No validation threshold is relaxed in this implementation.
Even success there would justify only independent development validation, not
paper-scale execution or an automatic 30%/latency claim. Fresh cohorts,
calibration, GPU submission and paper campaigns remain disabled in this protocol.
