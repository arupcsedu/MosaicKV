# V8.2: CPU-only readout-aware audit and event-driven completion

This diagnostic extends the frozen V8.1 captures without another GPU run.
It uses the same two consumed inputs and their 32 rejected proposals. It
does not access benchmark answers, change proposals, or create a new cohort.
The V8.1 source/plan, all captures, provenance, and negative result are bound
by hashes. The original horizon and FP32 tolerances remain unchanged.

## Readout trace

Only the cached final RMSNorm vector and vocabulary projection matrix are
loaded from the original pinned LLaVA-1.5 checkpoint. No complete model,
vision tower, dataset, or weight download is needed. Their tensor hashes,
shapes, dtypes, configuration/index hashes, revision, and epsilon are recorded.

For a saved final residual vector \(h\), the CPU sideband computes

\[
 n(h)=w\odot h\big/\sqrt{\operatorname{mean}(h^2)+\epsilon},
 \qquad z(h)=W_{\mathrm{LM}}n(h).
\]

It compares residual-space error, normalized error, and vocabulary KL.
An analytic directional derivative measures
\(-\nabla_h\mathrm{KL}(p_{\mathrm{FullKV}}\Vert p_h)^T\Delta h\).
A Fisher/logit-space quadratic approximation adds
\(-\tfrac12\operatorname{Var}_{p_h}(\Delta z)\). This omits RMSNorm's second
derivative and is **not** a KL bound. No gradients update model weights.

CPU FP64 arithmetic differs from the original GPU FP16 forward and FP32
guard. Reconstruction error and greedy agreement against the saved GPU
logits are reported explicitly. FP64 results never replace registered decisions.

## Information available before probing versus after probing

| Score | Availability and limitation |
| --- | --- |
| Original head reconstruction gain | Before probing; original FullKV-query surrogate |
| Projected residual gain | Before probing; uses predicted local projected change and FullKV/anchor residual error |
| Identity-tail readout gains | Before probing; assume the remaining decoder Jacobian is identity, explicitly unvalidated |
| Actual final-hidden/normalized-hidden gain | After probing; explanatory only |
| Actual readout directional/quadratic gain | After probing; needs the candidate's final residual, not a cheap proposer |
| Actual CPU readout KL change | After probing; explanatory only |

The projected residual score is
\(-\operatorname{mean}(2e\odot\widehat{\Delta r}+\widehat{\Delta r}^{\,2})\),
where \(e=r_{\mathrm{anchor}}-r_{\mathrm{FullKV}}\). It accounts for direction
and cancellation rather than only the norm of the changed head's error.

All prospective scores average the same 32 available FullKV/anchor draft
prefixes, matching the original surrogate's evidence horizon. Their ranking
is compared against each proposal's **first-prefix** measured KL change, so
selection of a failure prefix cannot leak into the prospective screen.
Observed later prefixes are explanatory only. No unexecuted candidate
prefixes or successful 32-token trajectories are invented.

## Descriptive screen, not a paper gate

Three alternatives are screened against the original reconstruction ranking.
On **both** consumed inputs, a signal requires rank correlation at least 0.3,
an increase of at least 0.1 over reconstruction, positive-prediction precision
at least 0.75, and at least four positive predictions. Positive outcomes use
the unchanged registered KL tolerance. These fixed criteria are recorded
before running this audit; they are not confidence intervals or generalization
evidence. Post-forward scores cannot qualify.

A positive screen authorizes only bounded CPU candidate-design work.
It does not establish successful substitutions, benchmark quality, or a memory
advantage. A negative screen automatically seals the diagnostic and stops
new inference. Fresh cohorts, calibration, paper campaigns, and GPU submission
remain disabled in either case.

## Completion without continuous polling

`submit_readout_v8_2.py` submits this finite dependency graph:

1. Optional parent job/array IDs finish (including failed jobs).
2. The CPU audit runs once through `afterany:<all-parent-IDs>`.
3. Completion runs once through `afterany:<audit-ID>`, validates hashes and
   completeness, writes `decision.json`, and takes the authorized next action:
   a negative seal or a bounded CPU-design request.
4. One independent, ephemeral `codex exec` reviewer reads only validated
   aggregate context and records a constrained structured interpretation.

There are no sleep loops, repeated `squeue` calls, log-tail loops, automatic
GPU retries, or recursive agents in this chain. Failed/missing artifacts
produce an integrity-blocked decision rather than an orphaned `afterok` job.
Callbacks are write-once; duplicate execution is rejected.

The Codex reviewer is **not** a wake-up of the current interactive chat.
It is a separate bounded CLI invocation, using existing authentication without
copying credentials. It cannot override the deterministic decision, modify
source, or submit jobs. Shell tools and web search are disabled. A rootless
Bubblewrap mount makes the filesystem, including home, read-only except for
the scratch callback directory and scratch cache root. Codex logs/SQLite
are explicitly redirected to scratch; history/session persistence and shell
snapshots are disabled. Missing Bubblewrap, credentials, connectivity, or a
timeout are recorded once, without retrying or weakening the scientific gates.

The CLI integration follows [official non-interactive documentation](https://developers.openai.com/codex/noninteractive/)
and [the configuration reference](https://developers.openai.com/codex/config-reference/).
It is intentionally distinct from unrestricted agent-driven source mutation.

## Commands

After CPU tests pass and a scoped immutable commit exists:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/submit_readout_v8_2.py \
  --root /scratch/djy8hg/runs/mosaickv/bounded-readout-v8-2-20261026-v1
```

For required parent jobs, append `--after JOB_A JOB_B` to delay the audit until
all of them finish, without polling. Artifact hashes still bind this audit to
V8.1: new captures require a separately versioned, reviewed binding, not merely
different dependency IDs. This does not authorize new capture runs.
CPU allocation is four cores/12 GiB/ten minutes for the audit and one
core/2 GiB/five minutes for completion; neither job requests a GPU.
