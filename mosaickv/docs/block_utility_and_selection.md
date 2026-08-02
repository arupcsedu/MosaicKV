# Block utility and budgeted selection

The backend-independent implementation lives in `mosaickv.selection`. It
scores nodes from a `SparseEvidenceGraph`, selects source blocks without
changing them, and can materialize the result as an `ExactTier`.

## Attention input boundary

`compute_block_utilities` accepts either per-node forecast attention or
per-layer/per-KV-head token attention. Token probabilities are summed over each
block's physical positions and normalized independently for every layer and KV
head. Inputs must be finite, nonnegative, carry a non-empty provenance label,
and be explicitly marked RoPE-aware.

This boundary is deliberate: current adapters capture pre-RoPE `q_proj`
vectors while their caches store post-RoPE keys. The utility implementation
does not silently dot incompatible query/key representations. A backend must
apply the correct model-specific positional transform or directly supply
RoPE-aware forecast attention.

## Per-block signals

Every `BlockUtility` records:

- forecast attention probability;
- value novelty, defined as one minus the maximum positive value-vector cosine
  similarity to an incoming or outgoing graph neighbor;
- expected attention-output contribution, defined as forecast probability
  times the pooled value-vector norm;
- normalized weighted graph centrality and singleton facility coverage;
- modality rarity from graph-wide modality frequency;
- redundancy, the complement of value novelty; and
- mandatory priority from the immutable cache block descriptor.

The configured local equation is implemented literally:

```text
u_i = lambda_q * forecast_attention_i
    - lambda_v * value_contribution_i
    - lambda_o * uniqueness_i
```

Here `value_contribution_i` is the recorded expected attention-output
contribution and `uniqueness_i` is value novelty. All raw terms remain in the
utility table so their signs and scale are auditable. The two coefficients are
signed: the default `lambda_v=-0.25` and `lambda_o=-0.25` turn the literal
minus terms into preservation rewards. Positive values intentionally penalize
the corresponding signal and remain available for explicit ablations. This
sign convention is important; a positive default would preferentially evict
the blocks with the largest expected attention-output contribution.

Attention, value contribution, and novelty are each normalized to a
nonnegative unit-sum signal within every layer/KV-head stratum before applying
the equation. This avoids making the registered coefficients depend on prompt
length or raw value-vector magnitude.

## Set objective and sign convention

Facility-location coverage is the average, over graph targets, of the maximum
sparse edge similarity supplied by a selected representative. Each block has
self-similarity one. Modality coverage is the fraction of modalities present
in the graph represented by the selected set. The objective is also literal:

```text
F(S) = sum(i in S) u_i
     - lambda_g * facility_location_coverage(S)
     - lambda_m * modality_coverage(S)
```

Ordinary facility-location and modality coverage are monotone submodular
rewards. Under the specified minus signs, the objective is submodular when
`lambda_g <= 0` and `lambda_m <= 0`; the defaults are therefore `-0.25` for
both. Positive coefficients intentionally penalize coverage and are supported
for exact evaluation and ablations, but the lazy-greedy selector rejects them
because cached marginal gains would not have the required diminishing-returns
upper-bound property.

`ObjectiveBreakdown` records both coverage rewards and their complementary
deficits, plus the local sum and exact total. The tiny-graph property auditor
exhaustively checks monotonicity and diminishing returns rather than inferring
monotonicity from coefficient signs; negative local utilities can make an
otherwise submodular objective non-monotone.

## Hard budgets and mandatory blocks

`SelectionBudget` supports:

- `blocks`: cardinality cost one per fixed-size block;
- `retained_slots`: the exact number of source positions in each block; and
- `bytes`: the descriptor's exact K/V payload bytes, for variable-sized
  blocks.

Mandatory blocks are inserted in node-ID order before optional selection. If
they alone exceed the hard budget, selection fails closed. Byte-mode results
require selected active bytes to equal budget cost and never exceed the byte
limit. The result can gather its selected source blocks through
`to_exact_tier(full_state)`.

## Lazy greedy and oracle comparison

The optional phase uses a heap of cached marginal-gain-per-cost bounds. Stale
entries are recomputed after the selected set changes. Deterministic ordering
is: larger gain density, larger raw gain, lower cost, then lower node ID.
Selection stops at a nonpositive gain by default.

Every graph block receives a `SelectionDecision`, including selected marginal
gain, gain per cost, rank, cost, and one of: mandatory exact, lazy greedy,
budget exhausted, nonpositive gain, or lower gain.

`exhaustive_select` enumerates feasible subsets only when node count does not
exceed `exhaustive_max_nodes`. `compare_greedy_to_exhaustive` reports the
absolute and relative objective gap and the exact number of evaluated feasible
subsets. This is a correctness diagnostic for tiny graphs, not an experimental
quality result.

The bounded v17 value ablation used identical packed active bytes. Removing
value contribution increased mean attention-output error by 0.0023, so the
registered value-aware term improved that requested fidelity diagnostic. It
reduced mean logit KL by 0.0085, however, so the downstream logit result is
mixed and no universal fidelity claim is made.
