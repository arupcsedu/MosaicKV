# V79 adaptive stable-head swap protocol

## Sealed v78 result

V78 passed the unchanged memory gate with at least 34.20% lower active and
active-plus-residual bytes than Prompt Top-K, but failed the unchanged DocVQA
fidelity gate. Its best candidate was 6.25 percentage points below Prompt
Top-K, exceeding the allowed five-point margin. Freeze job `18583439` therefore
exited with status 2 and did not create `frozen_candidate.yaml`.

The fixed 50% and 60% visual targets were enforced successfully. They increased
selected image blocks but did not improve FullKV answer agreement and reduced
token agreement. The immutable seal
`configs/ledgers/choice_c_v78_negative_calibration_seal.yaml` therefore retires
fixed visual quotas and binds the v78 protocol, ledgers, and results by SHA-256.

## Adaptive swap rule

V79 begins with the same deterministic, capacity-filling Prompt Top-K exact
selection in every stable layer/head. It then considers only equal-size swaps
between a selected, non-mandatory text block $t$ and an unselected visual block
$v$. Equal block sizes ensure a swap cannot change active bytes, cache length,
or padding.

Let $a_i$ be prompt-window attention mass. Within one layer/head, define

$$
g(v)=\frac{a_v}{\sum_{j\in\mathrm{visual}}a_j},\qquad
\ell(t)=\frac{a_t}{\sum_{j\in\mathrm{text}}a_j}.
$$

The swap is eligible only when

$$
g(v) \ge \rho\,\ell(t),
$$

where $\rho$ is the configured visual-gain ratio. After every accepted swap,
retained selected-text mass must also satisfy

$$
\sum_{i\in S'\cap\mathrm{text}}a_i
\ge \tau
\sum_{i\in S_0\cap\mathrm{text}}a_i,
$$

where $S_0$ is the pre-swap Prompt Top-K selection, $S'$ is the proposed
selection, and $\tau$ is the text-mass preservation threshold. Mandatory
tokens cannot be swapped. Candidate pairs are ranked by

$$
g(v)-\rho\ell(t),
$$

then visual gain, text loss, and immutable node IDs. At most two swaps are
permitted per stable head. This upper bound limits perturbation; it is not a
required visual quota, and a head performs zero swaps unless all evidence and
text-preservation conditions hold.

## Frozen candidate family

V79 preregisters three arms at the unchanged 27,500,000-byte cap:

- a no-swap v78 control;
- conservative swaps with $\tau=0.9975$ and $\rho=1.5$; and
- balanced swaps with $\tau=0.995$ and $\rho=1.25$.

All arms retain four-token blocks, the last-token mandatory policy, 1/32 routed
heads, at most two RoPE-safe prototypes per routed head, and no residual
storage or repair.

## Larger disjoint calibration

V79 uses 32 new DocVQA and 32 new TextVQA examples selected with seed
`20270907`. It excludes every declared v73-v78 ledger and prior observed result
row. Candidate selection remains label-blind and uses FullKV generated-output
agreement, generated-token agreement, active/residual bytes, and cache
telemetry only.

The unchanged gates require on every task:

- at least 30% lower active bytes than Prompt Top-K;
- at least 30% lower active-plus-residual bytes;
- FullKV answer agreement no more than five percentage points below Prompt
  Top-K; and
- at most 5% prototype logical bytes.

The prepared submission command is:

```bash
bash scripts/submit_choice_c_v79.sh
```

It submits only ledger materialization, two one-A100 calibration workers, and
the dependent CPU freeze. It requires a clean committed source snapshot and
does not submit held-out validation.
