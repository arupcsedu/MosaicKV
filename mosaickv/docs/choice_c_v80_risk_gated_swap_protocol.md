# V80 future-query risk-gated swap protocol

## Why v79 was rejected

V79 reduced active and active-plus-residual KV bytes by at least 34.80%, but
no candidate passed the unchanged fidelity gate on both tasks. The best
DocVQA candidate was 3.125 percentage points below Prompt Top-K and passed;
all candidates were 9.375 points below Prompt Top-K on TextVQA and failed.
No frozen candidate was created and no held-out campaign was submitted.

The derived label-blind audit accessed generated token IDs, FullKV generated
outputs, and cache telemetry only. It did not access benchmark references or
official task scores. Broad swapping changed 14--25 of 32 sequences per task.
It improved token fidelity on some examples and worsened it on at least as
many. Mean swap count had nearly zero correlation with whether a sequence
changed, so swap count alone is not a useful safety signal.

V79 compact traces stored only aggregate swap counts. Exact per-head causal
attribution is therefore unavailable and is not claimed. V80 corrects this
instrumentation gap: each layer/head trace records the exact displaced text
block and admitted visual block IDs, candidate count, risk rejections,
prompt-mass loss, and future-query forecast gain.

## Risk-gated rule

V80 still starts from deterministic Prompt Top-K selection within each stable
layer and KV head. A candidate exchanges one selected, non-mandatory text
block for one unselected visual block with the same position count. Equal-size
exchange keeps active bytes and cache capacity unchanged.

Let prompt attention mass be $a_i$ and future-query forecast attention be
$f_i$. Within one head, normalized prompt loss and forecast gain are

$$
\ell_p(t)=\frac{a_t}{\sum_{j\in\mathrm{text}}a_j}, \qquad
g_f(v)=\frac{f_v}{\sum_{j\in\mathrm{visual}}f_j}.
$$

The existing prompt-side evidence rule remains:

$$
\frac{a_v}{\sum_{j\in\mathrm{visual}}a_j}
\ge \rho_p\ell_p(t).
$$

The new risk gate additionally requires

$$
\ell_p(t) \le \epsilon, \qquad
g_f(v) \ge \rho_f
\frac{f_t}{\sum_{j\in\mathrm{text}}f_j}.
$$

If forecast mass is missing, non-finite, negative, or zero for either
modality, the gate rejects the swap. It never guesses. Mandatory blocks cannot
be displaced, the configured text-mass floor remains enforced, and at most
one exchange is allowed per head.

## Preregistered candidates

All candidates retain the v79 27,500,000-byte cap, four-token blocks, 1/32
routed heads, at most two RoPE-safe prototypes per routed head, and no
residual storage or repair.

- Control: no stable-head swaps.
- Strict: $\epsilon=0.0005$, $\rho_f=2.0$, 99.95% selected-text floor.
- Balanced: $\epsilon=0.001$, $\rho_f=1.5$, 99.9% selected-text floor.

V80 uses 48 new DocVQA and 48 new TextVQA examples selected with seed
`20271123`. All declared v73--v79 ledgers and prior observed result rows are
excluded. The unchanged gates require, on every task, at least 30% lower
active and active-plus-residual bytes, no more than a five-percentage-point
FullKV-agreement deficit relative to Prompt Top-K, and at most 5% prototype
logical bytes.

The prepared submission chain materializes the two ledgers, runs two
one-A100 calibration workers, and invokes the dependent CPU freeze. It does
not submit held-out evaluation.
