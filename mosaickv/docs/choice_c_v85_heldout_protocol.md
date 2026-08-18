# Choice C v85: v84-frozen untouched held-out validation

## Purpose

V85 tests the immutable v84 candidate on new DocVQA and TextVQA examples. It is a held-out quality--memory validation, not another tuning stage. The candidate ID, byte cap, block size, hybrid forecast, per-swap risk thresholds, model revision, decoding parameters, and comparison methods are fixed by the read-only v84 freeze.

The candidate is exact-only. Results must be labeled `mosaickv_exact` at the effective-method level even though `mosaickv_full` remains the common protocol slot used by the comparison harness. V85 cannot support prototype-merging, residual-repair, or complete-MosaicKV claims.

## Immutable source and candidate

The v84 calibration used clean commit `9e300dda54fb5377aefae06d1f28614e426d42fd`. V85 GPU execution must use an isolated detached worktree at that exact commit so the runtime snapshot remains identical to calibration. The preregistration and analyzer use a second clean worktree pinned to the later authorized v85 protocol commit. This separation permits an archived protocol without silently changing the evaluated runtime.

The frozen candidate SHA-256 is `940b124a2c2ab09d8192ab03d6cb2efda85339523e613980d567495a7280cf64`. Every materialized ledger binds this hash, and every GPU worker verifies it before model loading.

## Untouched samples

V85 deterministically selects 128 DocVQA validation examples and 128 TextVQA validation examples using seed 20280113. Materialization fails unless both the complete v83 held-out input directory and v84 calibration input directory are available as exclusion sources. It also excludes all discoverable earlier Choice-C ledgers and prior observed-result rows.

The sample-selection operation may read dataset identifiers and media needed to construct the ledger, but it may not inspect answer labels, references, task scores, generated answers, or failure identities. Official scoring occurs only after all three frozen methods have generated outputs.

## Compared methods

Each task uses identical model weights, precision, processor, chat template, media preprocessing, prompt, output length, deterministic decoding, attention backend, block granularity, mandatory-token policy, warmup count, and measurement protocol for:

- `full_kv`;
- `prompt_attention_topk`;
- protocol arm `mosaickv_full`, whose verified effective method is `mosaickv_exact` with the frozen v84 risk-gated hybrid selector.

Each method receives one warmup and two counterbalanced measured repetitions. Failed or missing rows invalidate the task.

## Preregistered gates

For each task separately, let (M) denote the frozen candidate and (P) Prompt Top-K. V85 requires:

\[
1-\frac{\operatorname{bytes}_{\mathrm{active}}(M)}{\operatorname{bytes}_{\mathrm{active}}(P)}\ge 0.30,
\]

and

\[
1-\frac{\operatorname{bytes}_{\mathrm{active}}(M)+\operatorname{bytes}_{\mathrm{residual}}(M)}{\operatorname{bytes}_{\mathrm{active}}(P)}\ge 0.30.
\]

Quality is evaluated by paired per-example official task scores against both FullKV and Prompt Top-K. For each comparator, the candidate must have a nonnegative paired mean difference and the lower endpoint of the 10,000-resample paired-bootstrap 95% confidence interval must be at least (-0.05). Both comparisons and both memory boundaries must pass on both tasks. FullKV must remain transformation-free.

Latency is reported but cannot reject the run. DocVQA and TextVQA scores are reported separately and are never averaged as if they were the same metric.

## Execution boundary

This protocol is preregistered but unsubmitted. Its submission script must refuse a dirty repository, a changed frozen-candidate hash, a changed runtime commit, a reused output directory, or missing v83/v84 exclusion ledgers. No candidate parameter can be changed after held-out materialization. A failed v85 result must be sealed as negative rather than used to revise v84.
