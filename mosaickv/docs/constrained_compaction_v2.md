# Constrained attention-compaction diagnostic V2

This is an exploratory **CPU-only saved-capture study**, not a quality result,
paper experiment, new cohort, or calibration. The effective method name is
`attention_matching_constrained_reimpl`. No model weights are trained or changed.
The negative GPU diagnostic remains immutable at
`/scratch/djy8hg/runs/mosaickv/attention-compaction-hf-gate-v1-20261027-v1`.
Do not rerun the rejected construction.

## Question and fixed controls

The old fit solved an unconstrained least-squares problem and then rejected a
head if any stored value exceeded four times its source value's norm. The V2
study asks whether enforcing that same bound *during* fitting yields useful,
stable coefficients. It does not widen the bound, relax the model guard, select
benchmark failures, or read benchmark answers/references.

The saved inputs are the same two consumed LLaVA-1.5 diagnostic requests. Their
32 layers, 32 heads, 32 draft prefixes, and 39 selected prompt positions per head
are fixed. Saved original keys/values, probabilities, history mass/numerators,
and FullKV attention outputs supply all evidence. Keys and logical positions
are never refitted. First/last mandatory values stay bitwise original and their
log biases stay zero. The stored FP16 value norm limit remains 4×; mass
multipliers remain in [1,64]. Ridge coefficients remain those of the parent.

## Four separate constructions

Let A contain original FullKV attention probabilities at the fixed dictionary,
V0 the selected source values, h the retained generated-history mass, n its
attention-weighted value numerator, and Y the FullKV attention output. For
nonnegative mass multipliers w, the reconstructed output is

\[
  \widehat Y_q =
  \frac{\sum_j A_{qj}w_jV_j+n_q}{\sum_j A_{qj}w_j+h_q}.
\]

- **Anchor:** w=1 and V=V0.
- **Mass-only:** fit the existing bounded mass objective with V unchanged.
- **Value-only:** w=1 and fit norm-constrained values.
- **Joint:** first fit the same bounded mass objective, then fit values under
  those fixed multipliers. This is a sequential construction, **not** a jointly
  globally optimal nonlinear solve.

For the two value-fitting arms, define d=Aw+h, X=A diag(w)/d and
R=Y-n/d-XV0. The projected solve is

\[
 \min_\Delta \tfrac12\|X\Delta-R\|_F^2+
                 \tfrac\lambda2\|\Delta\|_F^2,
 \qquad \|V_{0j}+\Delta_j\|_2\le4\|V_{0j}\|_2,
 \quad \Delta_j=0\text{ on protected positions}.
\]

Here lambda is the parent's relative ridge times the squared largest singular
value of mutable X. Zero original values cannot gain nonzero norm. A fixed
256-step FP64 projected FISTA solve retains its best feasible regularized
iterate; each new candidate iterate is projected onto the value balls. The
accelerated extrapolation is an optimization workspace, not a stored cache.
No model gradient or optimizer is involved.

The radius includes a conservative dtype-rounding margin. After conversion to
the actual stored dtype, norms are checked again; rare over-bound rows are
projected back or restored to their original value. This is not unconstrained
fitting followed by head rejection: the iterations themselves are constrained.
Actual cast outputs determine reconstruction scores. A head is admitted only
if its **training-prefix** MSE improves by the unchanged parent minimum of 1e-8;
otherwise it returns the original same-dictionary zero-bias anchor. Both attempted
and returned construction errors are retained in observations.

## Prefix holdouts and their limitation

Fit coefficients separately on prefixes 1–8, 1–16, and 1–24; score the subsequent
8 prefixes (9–16, 17–24, and 25–32). Holdout targets never enter the coefficient
solver or per-head admission rule. These folds also expose behavior after the
16-position bounded generated-history ring begins evicting history.

**The dictionary was selected using all 32 parent draft prefixes.** Therefore,
these are conditional coefficient-fit holdouts with a frozen dictionary, not
independent key-selection tests, independent model decodes, or evidence of
benchmark-quality preservation. Attention reconstruction and autoregressive
logits are different quantities. Even a large CPU improvement may fail the
unchanged downstream model guard.

## Registered descriptive screen and stopping rule

Before reading this study's real outcomes, require for an eligible component:

1. all stored values obey the original 4× norm bound and mandatory protection;
2. each input/fold has at least one admitted head;
3. every held-out fold has aggregate returned MSE ratio at most 1.000001
   relative to its same-dictionary anchor;
4. the mean of its three fold ratios is at most 0.95 on **each** consumed input.

This is a descriptive surrogate screen, not a statistical quality gate. Among
eligible arms choose the lowest worst-fold ratio, then mean ratio, then name.
If none qualifies, preserve the negative study and submit **no GPU follow-up**.
Do not change these thresholds after observing results.

Only a useful bounded fit permits a separately recorded diagnostic using the
same consumed inputs, 16 coefficient-fitting prefixes, the same 28 MiB total
persistent cache cap, zero-bias/retention-1 and clone/repack controls, and the
unchanged **32-token** guard at checkpoints 1/4/16/32. Every prefix must remain
no worse than the anchor in registered KL and maximum logit error, with no new
greedy-token disagreement (atol 1e-7, rtol 1e-6). CPU eligibility does not submit
that GPU diagnostic automatically. A GPU execution needs a clean immutable
source. Fresh cohorts, calibration, and paper-scale work remain blocked.

## Execution and provenance

Run only in the common `/scratch/djy8hg/env/mosaickv` environment after sourcing
`mosaickv/scripts/cache_env.sh`. No installations, weights, media, or dataset
access occur. All caches and outputs stay under `/scratch/djy8hg`.

To avoid frequent commits, this CPU engineering study can use a staged,
hash-bound dirty overlay on the real parent Git SHA. Its complete binary patch,
per-file hashes, parent negative seals, capture hashes, config, and source dirty
status are preserved. Such a run is explicitly **exploratory, not canonical**.
Do not change these bound files while workers run. Canonical GPU and paper runs
still require a clean immutable source snapshot.

Stage only the eight new study files and submit to a new immutable root:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/study_constrained_compaction_v2.py submit \
  --root /scratch/djy8hg/runs/mosaickv/constrained-compaction-cpu-v2-20261027-v1
```

This submits two one-CPU/no-GPU workers (4 GiB, 20 minutes each) and a finite
`afterany` CPU analyzer (5 minutes), not a polling controller. The analyzer
hash-checks artifacts and recomputes component summaries from per-head JSONL.
Failures or missing workers cannot authorize follow-up. Outputs include
per-fold component errors, per-head returned/attempted errors, stored norm
checks, timings, saved 16-prefix fits, producer model/dataset provenance, and
an immutable decision. No benchmark task score is inferred from these tensors.
