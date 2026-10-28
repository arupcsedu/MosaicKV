# Numerical recovery and the separate freshness constraint

This is a bounded, label-blind diagnosis on **consumed inputs**, not a new
benchmark cohort. Existing failed gates and completed runs remain unchanged.
No attention precision option is enabled in the online compressor.

## What is actually known

Independent compact-source construction matched production packing exactly.
The MileBench failures were in the additional no-eviction padding control, not
evidence of changed retained entries. All 32 COCO FullKV boundary-A/B controls
exceeded the registered 1e-4 absolute bound (0.0078125--0.03125), despite identical
first-token decisions. Neither observation justifies waiving a numerical gate.

Installed Transformers 4.55.2's `models/llama/modeling_llama.py` and
`models/mistral/modeling_mistral.py`, `eager_attention_forward`, multiply FP16
Q/K, use FP32 softmax, then cast probabilities back to the query dtype before
the probability/value product. Layout-dependent arithmetic is a hypothesis,
not a proven explanation of every error.

## Registered arithmetic experiment

Compare the native function with an evaluation-only, model-scoped version using
FP32 QK, softmax and probability/value multiplication. Cast the attention output
and returned probabilities back to the original query dtype. Model weights,
stored K/V dtype, logical positions and masks do not change. Process-global TF32
is explicitly disabled for both serial conditions and restored on exit. Native
function-source hashes are recorded; no installed package file is edited.

The experiment uses the same two consumed MileBench padding-failure inputs and
the first two already consumed COCO validation inputs. It tests ordinary-generate
and cache-reinjection parity for at least 16 tokens, 16 forced padding-control
steps, and the original A/B check. Every numerical threshold is unchanged:
1e-4 for retention-one and A/B, eight FP16 epsilons for normwise padding error.
An independent same-chunk B/reinjection check is an additional diagnosis and
**cannot replace a failed A/B check**. No benchmark references enter the model.

Even a positive result permits only a separately registered common-arithmetic
regate: every method/reference would need the same policy and new correctness,
quality and timing evidence. Historical FP16-arithmetic measurements are not
pooled with the new variant. A diagnostic is not proof of superiority.

## Freshness is not an arithmetic or configuration switch

The original pinned official pools contain 200 IDs per subtask. Known historical
exclusions/ledgers cover all 200 IDs for ActionSequence, CLEVR-Change and
Spot-the-Diff. They cannot supply 64 untouched IDs each. Adversarial annotations
over the same media are not a fresh replacement. A bounded scanner is not proof
of exhaustive media disjointness.

The scanner now resolves ledger links relative to their containing manifest,
not the shell working directory; missing/outside-scratch links still fail
closed. A new inventory and feasibility report are generated without altering
the old inventory, selecting new samples, or claiming scan completeness.

If these pools remain exhausted, a **new, explicitly reviewed protocol** must
use additional official MileBench subtasks/unused source data, or scope fresh
confirmation to the supported COCO lane. Previously used inputs may remain
development evidence but must not be called untouched confirmation. No such
replacement or confirmation is automatically submitted by this diagnostic.

## Execution

Use a clean numerical-recovery worktree, source `cache_env.sh`, set the common
environment/PYTHONPATH as in `command_examples.md`, then:

```bash
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_numerical_recovery.py submit \
  --root NEW_IMMUTABLE_RUN_ROOT \
  --mile-parent /scratch/djy8hg/runs/evicastkv/fox-review-20261028-v4 \
  --coco-parent /scratch/djy8hg/runs/evicastkv/fox-review-20261028-v6-coco-scoped \
  --dependency EXISTING_OPTIMIZATION_ARRAY
```

One A100/12 CPUs/64 GiB/20 minutes, then one finite `afterany` CPU analysis.
The dependency preserves the six-GPU aggregate ceiling without changing any
running experiment. No repeated polling, package installation or quality
successor is used. Outputs: `protocol.json`, per-case records, `results.json`,
`analysis.json`, `report.md`, `history-inventory.json`, `freshness_resolution.json`.
