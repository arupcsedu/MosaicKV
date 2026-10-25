# Bounded-history V6: single-exchange draft guard

## Why V5 is negative

The immutable V5 report and its two ledgers are hash-bound in
`bounded_history_memory_v5_negative_seal.json`. All 16 generation proposals
rolled back. Their first guard violation occurred at prefix 1 (9 proposals),
2 (5), or 4 (2), but all were decoded for 32 prefixes. V5 therefore measured
additional overhead without changing the same-byte compressed anchor's answer.
Its 16 identities are consumed and excluded from V6. Original raw artifacts are
not edited. The successor uses this aggregate failure mechanism, not sample
answers, references, or failure identities, to motivate the change.

## V6 transaction and decision

V6 is a separately labeled development variant, not a change to the frozen V5
policy. Its effective method includes `__draft_single_exchange_layers_v6`.

1. Start from the ordinary MosaicKV exact anchor, with the same prompt/history
   partition and hard lifetime byte cap as Prompt Top-K. Keep exact original
   K/V and logical positions. No prototypes or residual storage are introduced.
2. Construct at most four independent single exchanges from the current
   anchor. Each removes one nonmandatory token **from every KV head of one
   donor layer** and adds one omitted token to every KV head of a receiver.
   Equal layer-slot costs conserve physical bytes. Per-head lengths are equal
   within a layer; different layers may have different lengths. No padding is
   used. Prompt reconstruction shortlists candidates; it cannot authorize one.
3. Clone the existing FullKV prefill and compressed states. Advance FullKV and
   the original anchor lazily, once per needed prefix. Use only FullKV's own
   greedy draft tokens as inputs, never benchmark targets or references.
   Reuse these logits/tokens for every candidate. No second prefill is allowed.
4. Probe each candidate on private cache buffers. At **every prefix**, require
   FullKV-relative KL and maximum absolute logit error to be no worse than the
   original anchor, within the unchanged `atol=1e-7`, `rtol=1e-6`. Also forbid
   new greedy-token disagreements where the anchor agrees with FullKV.
   A composition must additionally satisfy the same checks against its
   previously accepted parent. This prevents chained tolerance drift.
5. Stop a candidate on its **first violation**. Rejection can be established
   from a partial horizon; acceptance cannot. Passing plans must complete all
   32 prefixes, including checkpoints 1/4/16/32. Rank passing candidates by
   actual draft disagreement count, summed KL, summed maximum logit error,
   then logical-position SHA for deterministic ties.
6. Compose another single exchange only after selecting a complete-horizon
   passing parent. At most two rounds/four proposals each are permitted.
   If no proposal passes, retain the current compressed anchor, not FullKV.

The hard upper bound is eight proposals and 320 model decode calls per phase
(32 FullKV + 32 original-anchor + 8×32 candidate updates). Rejections reduce
that bound. For example, one proposal rejected at its first prefix requires
three calls rather than V5's 96; this is an **algorithmic count**, not a measured
GPU speedup. All planning, packing, cloning, checks and probe time are charged
to compression/end-to-end time. Temporary probe caches and verification copies
are recorded separately from published live decode KV and contribute to peak
GPU memory. A faster rejection is not evidence of better answer quality.

## Trace and CPU verification

Each attempt records its parent/candidate logical-position hashes, single
physical transaction, layer capacities, prefix metrics, first rejection,
complete-horizon status, and root/parent decisions. The replay validator checks
the decision, first-violation stopping, shared anchor evidence, byte continuity,
actual-logit winner, composition order, probe counts, timings, and rollback.
Source tensors and original logical counters are checked unchanged.

CPU tests cover single/ composed exact-byte exchanges, mandatory preservation,
deterministic ordering, real tiny-model layer-local cache transport, source
isolation, partial-pass rejection, violations at tokens 1/2/4/16/32, parent-only
regression, 100% retention bypass, and zero-update teacher forcing. Synthetic
and random-weight tests are correctness evidence, not pretrained quality data.
V1–V5 policy behavior and disabled-field config hashes remain unchanged.

## Development execution only

`configs/experiments/bounded_history_memory_v6.yaml` registers the next eight
parent IDs per task after the 30 explicitly consumed development IDs:
LLaVA-Description and paper-style MM-Vet, 16 examples total. The historical
parent cohorts were previously evaluated; this is not globally untouched or
paper-level held-out evidence. The selection uses IDs only. Prepare checks all
five prior ledgers per task by SHA and rejects any overlap.

Four arms are FullKV (quality reference only), Prompt Top-K, the original
MosaicKV exact anchor, and V6. All compressed arms use exactly 40 MiB at initial,
final, and peak live-storage boundaries, with 1,024 output tokens, five warmups,
FP16/eager and identical inputs. Both generation and teacher-forced phases are
audited. Corpus PPL and FullKV-reference ROUGE-L remain separate metrics.

After an explicitly authorized scoped commit makes the source clean, the
canonical small submission is:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/submit_bounded_history_memory_v3.py --version 6 \
  --root /scratch/djy8hg/runs/mosaickv/bounded-history-memory-v6-20261025-v1
```

The command refuses a dirty worktree or existing output root. It submits two
reference and six compressed workers with the tested aggregate-six-A100
controller, one A100/12 CPUs per worker, a two-hour limit, reservation
`bi_fox_dgx`, followed by CPU analysis. It does not launch calibration, powered
validation or paper curves and cannot create a frozen candidate.

Execution success alone is insufficient. The preliminary screen must observe
actual accepted changes in each task and no point-estimate PPL/ROUGE regression
against either Prompt Top-K or the original anchor. Report paired intervals,
fallback/acceptance frequency, phase timing, exact storage and failures even if
the screen fails. A positive small screen still requires separate disjoint
calibration, freezing and powered validation before any paper claim. This
equal-byte test cannot itself establish smaller memory at equal quality.
