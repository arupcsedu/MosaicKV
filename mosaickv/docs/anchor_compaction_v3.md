# V3: anchor-preserving, CPU-only compaction study

V2 is a **preserved negative result**, not a failed job to retry. Its sealed
report, decision, source overlay, and two consumed GPU captures remain untouched.
V3 uses a separate worktree, immutable output root, config, and method label:
`attention_matching_anchor_preserving_reimpl`. This is an exploratory
saved-tensor reconstruction study, **not benchmark-quality or paper evidence**.
No model weights, datasets, media, task labels, or new identities are accessed.

## Why this differs from V2

The constraint `||V'|| <= 4 ||V0||` alone permits a correction as large as
`||V'-V0|| <= 5 ||V0||`. V2 respected its final norm bound, yet some later-prefix
errors worsened substantially despite almost zero fitting error. This motivates
a small trust region around the original values rather than relaxing that bound.
It does not establish that the new construction will improve model quality.

Use the same normalized attention matrix X, FullKV output residual R, original
values V0, and unchanged keys/positions. Solve

\[
 \min_\Delta \frac12\|X\Delta-R\|_F^2+
       \frac{\lambda}{2}\|\Delta\|_F^2,
 \quad \|\Delta_j\|_2\le0.1\|V_{0j}\|_2,
 \quad \|V_{0j}+\Delta_j\|_2\le4\|V_{0j}\|_2.
\]

Protected corrections remain exactly zero. The fixed candidate increases the
relative ridge from 1e-6 to 1e-2 (10,000×); lambda is that coefficient times
the squared largest singular value of mutable X, as in V2's scaling. Values
are request-local cache coefficients: no model training or weight updates occur.

The candidate therefore cannot use a large counteracting value to fit a small
set of prefixes. The zero correction remains feasible. Its 10% delta ball lies
inside the original 4× value-norm ball by the triangle inequality. Both actual
stored-dtype constraints are still checked, including zero-value and cast-rounding
cases. The fixed 256-iteration projected solver has a conservative cast cushion
and restores any row that cannot be represented safely. It does not enlarge
the trust region after rounding.

## Fixed diagnostic cases

Before executing real fits, register three cases:

| Case | Relative ridge | Delta limit | Purpose |
| --- | --- | --- | --- |
| `ridge_only` | 1e-2 | No extra delta limit; 4× final bound retained | Mechanism ablation |
| `delta_only` | 1e-6 | 10% of source value norm | Mechanism ablation |
| `anchor_preserving` | 1e-2 | 10% of source value norm | Only promotable candidate |

Each case separately reports exact same-dictionary anchor, mass-only,
value-only, and sequential mass-plus-value corrections. Mass multipliers stay
in [1,64]; mandatory biases stay zero. The two ablations cannot be promoted if
the fixed candidate fails. No adaptive per-head holdout selection is allowed.

Fit prefixes 1–8, 1–16, and 1–24 independently and score subsequent prefixes
9–16, 17–24, and 25–32. Fitting and per-head admission use only the fitting
prefixes; later outputs never enter the solve. As in V2, the fixed 39-key
dictionary was selected using all 32 draft prefixes. These are **coefficient-only
holdouts on already consumed inputs**, not independent dictionary or answer tests.
These studied holdouts cannot later be relabeled as untouched validation.

## Unchanged decision and stopping rule

Every input/fold must obey all stored norm, delta, mandatory-value, and bias
constraints. Every fold needs a meaningful admitted change and aggregate held-out
MSE no worse than its anchor (ratio <=1.000001). Each input's mean of its three
fold ratios must be <=0.95. These are exactly V2's descriptive usefulness
thresholds, with an additional delta constraint; they are not statistical
benchmark gates. Missing or failed cases block promotion.

If the fixed candidate fails, preserve the negative result: **no GPU rerun,
calibration, or larger campaign**. If useful, the analyzer may recommend—but
does not submit—a separately authorized same-consumed-input GPU diagnostic.
That diagnostic still needs a clean immutable source, the unchanged 28 MiB
persistent-cache cap, clone/repack and zero-bias/retention-1 controls, and all
32 prefixes of the registered no-worse KL/logit/token guard (atol 1e-7,
rtol 1e-6, checkpoints 1/4/16/32). CPU reconstruction gains do not establish
model-logit or answer-quality preservation.

## Scheduler and typing repairs

The `30:00` parser correction already exists at base commit `7acfd01` and is
included in V3's base. The timed-out V9 controller used older frozen source.
Do not modify that historical run. Bind the current controller's file hash and
run a CPU preflight covering `30:00`, `59:59`, `2:00:01`, and day-prefixed
durations before submission; no Slurm state is changed by that preflight. The
existing controller regression also replays V9's two 30-minute workers.

The deferred `extra: dict[str, Any]` annotation is corrected only in V3's
separate solver source. V2's hash-bound source remains unchanged and verifiable.

## Bounded execution

Use only `/scratch/djy8hg/env/mosaickv`, after sourcing
`mosaickv/scripts/cache_env.sh`; every cache stays under scratch. Six CPU workers
(one consumed input × one case each) request 1 CPU, 4 GiB and 20 minutes, with
at most four running concurrently. A finite `afterany` analyzer requests 1 CPU
and 5 minutes. **No GPU is requested or visible.** No polling controller runs.
Every fold saves per-head observations immediately, including failed-worker
context; completed outputs and decisions are write-once.

As with V2, avoid frequent commits by preserving a complete staged binary patch
on the real Git base and hashing all source/input files. Such execution remains
explicitly **dirty-source exploratory**, never canonical. Keep the bound source
unchanged while workers and their analyzer run. The new plan also binds V2's
negative seals and immutable source/input hashes.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/study_anchor_compaction_v3.py submit \
  --root /scratch/djy8hg/runs/mosaickv/anchor-compaction-cpu-v3-20261027-v1
```

Stage only the scoped V2/V3 implementation files first. All fresh-cohort,
calibration, GPU-execution, and paper-campaign authorization flags stay false.
