# A representation change, not another exact-substitution successor

Status: experimental numerical core, consumed-capture CPU feasibility tool, and
an isolated HF eager engineering transport. There is **no production runtime,
vLLM integration, or new benchmark-quality result**. Existing
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

## Isolated HF engineering gate, not another cohort

The separate `attention_compaction_hf_gate_v1.yaml` authorizes only two one-A100,
12-CPU, 30-minute workers on the SAME two consumed inputs. The original CPU
protocol's permissions are unchanged. No new cohort, calibration, paper run, or
unbounded follow-up is authorized.

`hf_attention_bias.install_attention_bias_cache` is a NEW installer, not a
bypass of the existing exact-only installer. It verifies every original
post-RoPE key and actual first/last mandatory value against the source. It
allocates its own bounded-history K/V, FP32 bias and per-query-head mask, and
counts unique backing storage including auxiliary decode metadata. Original
logical positions are recorded, and the next token retains its original position.
Only the audited Transformers 4.55.2 LLaVA-1.5 eager implementation is accepted.

Each worker performs ONE prefill, then a 32-token FullKV greedy draft together
with a zero-bias retention-1 transport. Bitwise logit and token identity is the
registered null gate. Failure prevents fitting/probing. Draft attention is
captured from real FullKV forwards; the generated-ring position schedule is
computed deterministically without pretending to execute another model.

If parity passes, recover real original K/V from that prefill and construct
39-slot dictionaries using separate heads/prefixes and the frozen mean-draft
ranking. The two actual mandatory boundary positions are protected—not the
CPU feasibility audit's unverified stand-ins. This is an attention-matching
control dictionary, **not** the frozen MosaicKV graph selector. CPU fitting
is explicitly measured with D2H/H2D costs. One fixed joint candidate is tested;
no hyperparameter search, alternative ranking, or extra model probing is allowed.

Compare the fitted cache to the identical-key, original-value, zero-bias anchor
with identical actual backing storage. Run independent clone and source-regather
controls, including an A-B-A repack replay after candidate execution. Every
accepted candidate must complete the unchanged 32-token, every-prefix guard:
`atol=1e-7`, `rtol=1e-6`, no worse KL/max-logit difference and no new greedy-token
disagreement. Rejections stop at their first violation; partial execution can
never pass. A numerical tie or all-head rollback is not meaningful improvement.
FP64 KL is diagnostic sideband, never a substitute for the registered GPU-FP32
decision. No tolerance is relaxed after a failure.

Phase events are flushed for partial failures. Save raw logits, real draft
evidence, fitted/anchor values, per-head decisions, storage observations and
manifests. An `afterany` CPU job independently rechecks hashes, null-control
logits, guard decisions and storage, then writes a fail-closed decision. A
single schedule-controller admission pass enforces the aggregate six-A100 cap;
this two-task array does not need a polling watcher. No automatic new GPU run
is permitted, regardless of outcome.

Synthetic CPU parity does not establish pretrained parity. A positive GPU
diagnostic would open a design review only, not paper-scale execution or a
30%/latency claim. First demonstrate useful model-level changes, then separately
authorize independent benchmark-quality evaluation.

From a clean committed snapshot, submit only this gate with:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/attention_compaction_hf_gate.py submit \
  --root /scratch/djy8hg/runs/mosaickv/attention-compaction-hf-gate-v1-20261027-v1
```

The GPU array uses `bii-gpu`, reservation `bi_fox_dgx`; the dependent five-minute
CPU analyzer uses `bii`. `submission.json` records job IDs. Completion produces
`analysis/report.json` and `analysis/decision.json`; negative or missing workers
cannot promote a new run. Pending resources are not a license to change samples,
thresholds, budgets, reservations, or other users' jobs.
