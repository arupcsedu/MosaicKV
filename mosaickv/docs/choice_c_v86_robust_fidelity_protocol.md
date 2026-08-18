# Choice C v86: fresh disjoint robust-fidelity calibration

## Status and scientific boundary

V85 is sealed as a negative held-out result. It remains part of the scientific
record and is not a calibration source. V86 may use the v85 input directory only
as an exclusion set. V86 design, candidate ranking, and freezing must not use v85
answers, references, task scores, generated tokens, per-sample telemetry, or the
identities of examples on which v85 succeeded or failed.

The v86 candidate family was fixed using only aggregate, label-blind calibration
telemetry that existed before v85. In particular, the earlier calibration showed
unused margin above the required 30% memory reduction and showed that aggregate
answer agreement alone was not a sufficiently conservative freeze criterion. No
v85 measurement selects the v86 byte cap or risk thresholds.

## Design

All candidates use the same LLaVA-1.5 weights, processor, prompts, media,
deterministic decode, four-token blocks, mandatory-token policy, backend, and
28,500,000-byte configured active-cache cap. The cap spends some memory margin on
additional exact blocks, but it is not assumed to pass: measured active and
active-plus-residual bytes must each remain at least 30% below Prompt Top-K on
every task.

The three preregistered arms are:

1. an exact-only no-rerank control;
2. an ultra-conservative hybrid-forecast arm allowing at most one stable-head
   replacement; and
3. a conservative hybrid-forecast arm allowing at most two replacements.

Prototype merging, residual storage, and repair are disabled. Therefore a frozen
candidate, if one exists, supports only the `mosaickv_exact` effective-method
claim. Draft rollout reuses the original prefill, operates on an isolated cache,
and is deterministic.

## Fresh sample boundary

The ledger materializer must run from a clean commit and select 128 DocVQA and
128 TextVQA calibration examples with seed 20280203. It excludes all repository
ledgers, all earlier Choice C input directories, prior observed result rows, and
the complete v85 input directory. The materializer fails if the sealed v85 input
directory is absent, preventing an accidental calibration run without the
required exclusion boundary.

## Label-blind robust fidelity gate

Calibration telemetry removes benchmark references and task scores. Candidate
selection compares only generated outputs and token IDs with the paired FullKV
and Prompt Top-K outputs.

For each task, sample identifiers are ordered by a salted SHA-256 digest and
assigned round-robin to four folds. This produces exactly 32 examples per fold
without consulting labels or model outputs. Every candidate must first satisfy
all existing gates:

- at least 30% measured active-byte reduction versus Prompt Top-K;
- at least 30% measured active-plus-residual reduction versus Prompt Top-K;
- FullKV exact-answer agreement no more than five percentage points below Prompt
  Top-K on each task;
- zero prototype logical fraction and zero residual bytes;
- valid hybrid forecast divergence; and
- the preregistered risk-activity and no-rerank-control checks.

V86 then adds, without weakening those gates, two checks on every task and every
fold:

- exact-answer agreement with FullKV may be at most five percentage points below
  Prompt Top-K's agreement with FullKV; and
- token agreement with FullKV may be at most five percentage points below Prompt
  Top-K's agreement with FullKV.

Among candidates passing every gate, selection maximizes worst-fold exact-answer
agreement, then worst-fold token agreement, then macro token agreement, followed
by a lexical candidate-ID tie break. If no candidate passes, no
`frozen_candidate.yaml` is created and held-out evaluation remains blocked.

## Execution rule

The designed chain is ledger materialization, two one-A100 calibration workers,
and a CPU robust-fidelity freeze analysis. It submits no held-out job. The current
worktree contains intentionally uncommitted scientific records, so the chain must
not be submitted until the user explicitly authorizes a clean commit.
