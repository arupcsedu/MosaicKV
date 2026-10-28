# Opportunity/demand V1: bounded quality study

The two-input correctness canary passed at `1667824`; it did not measure a
quality advantage. This separately registered stage uses **all 64 original,
already consumed COCO development inputs**, never an untouched confirmation
claim or another sample selection based on answers. No new images are selected
or materialized. Original negative results remain immutable.

The production policy, coefficients, precision, backend and KV construction are
bitwise source-identical to the passed canary. Compare the frozen complete arm,
opportunity routing only, demand coverage only, and both. Preserve Prompt Top-K
and `vl_cache_reimpl` at **both 20/40 MiB**, plus FullKV once as a quality
reference. This is 13 arms/input, **832 requests**. Every request recomputes its
own complete prefill and generation. Each input's 13 arms run on one GPU in
counterbalanced order. No representation clones or boundary-B experiments
contaminate these quality runs. They are not repeated headline latency trials.

Require per-input matching of actual initial, final and peak backing storage,
including VL-Cache padding; original prompts/media/references, 64 emitted
tokens/63 appends, model/processor/scorer hashes and configuration are bound.
Every loaded worker checks retention-one parity. The passed experimental
identity and independent representation canary is hash-bound, not rerun for
every scored request. Errors and incomplete rows block full-cohort statistics.

Before submission, register **18 exploratory development contrasts**: each
of three new variants minus each of Top-K, VL-Cache and frozen complete, at
each budget. Use 10,000 paired reusable-media-cluster bootstrap replicates,
seed 20271014, ordinary 95% and Bonferroni family-18 intervals. Recompute corpus
CIDEr and reference document frequencies in every replicate; verify the sparse
reducer against pinned official reduction on this actual prediction cohort.
Do not average per-shard CIDEr. The original Fox six-contrast confirmation
family is not silently changed into this exploratory family.

The decision reports whether a variant has a **strictly positive** adjusted
lower bound against both compressed baselines at both budgets. Zero or a
zero-crossing interval is inconclusive, not equality or noninferiority.
No interim outcome-based stopping, seed choice, cohort enlargement, new byte
cap, floor weakening or automatic confirmation is allowed. Failure to pass is
a negative/inconclusive development result, not an implementation exception.

Six one-A100 workers maximum in aggregate, 12 CPUs and 45 minutes each; 16
four-input shards. A finite CPU `afterany` callback analyzes success **and**
failure. Common environment and all caches are scratch-only. No polling daemon
or implicit wake-up of an interactive agent is claimed.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_opportunity_development.py prepare --root /scratch/djy8hg/runs/evicastkv/NEW_DEVELOPMENT_ROOT
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_opportunity_development.py submit --root /scratch/djy8hg/runs/evicastkv/NEW_DEVELOPMENT_ROOT
```

Independent COCO-only confirmation remains a **different** stage requiring a
frozen candidate, input/media history audit, prospective test family, fixed
sample size/power and stop rule. MileBench remains development-only. Prototype
and residual-repair performance are not evaluated or claimed here.
