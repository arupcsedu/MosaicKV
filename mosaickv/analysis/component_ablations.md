# Experiment 3: small component-ablation study

## Scope and validity

This is an exploratory dirty-worktree validation run, not a paper result. It uses deterministically selected subsets (mmstar=2, docvqa=2, chartqa=2, video_mme=1), Qwen/Qwen2.5-VL-3B-Instruct in bf16, eager prefill, eager decode, 8 fixed greedy tokens, retention 0.2, and 16-token blocks. The primary budget is measured packed active KV bytes. The small sample count is useful for implementation direction only and cannot establish benchmark significance.

The analyzed index contains 48 measured cells, 84 completed sample rows, and 0 failed sample rows. Raw run artifacts are under `/sfs/weka/scratch/djy8hg/runs/mosaickv/component-ablations-v17`.

Controlled-comparison validation: **passed**.

Qwen2.5-VL cache keys are post-M-RoPE. Prototype rows are considered measured only when their trace records the audited de-rotate/phase-neutral-merge/re-rotate transform and `tier_mode=three_tier`; exact safety fallbacks are labeled separately. The number of measured three-tier cells in this run is 44.

## Answers

1. **Forecasting and selected-block recall.** Mean recall changes relative to the no-forecast control were: full hybrid 0.2888, prompt-only 0.2055, draft-only 0.2870, and evaluation-only oracle 0.2870. The hybrid forecast therefore improves selected-block recall in this run. Its per-task score changes (A minus B) were mmstar=0.0000, docvqa=0.2353, chartqa=-0.5000, video_mme=0.0000, so the recall gain does not by itself establish a quality gain. Task metrics are not averaged across benchmarks.

2. **Graph coverage on structured tasks.** Full-minus-no-graph per-task score changes were mmstar=0.0000, docvqa=0.0000, chartqa=0.0000, video_mme=0.0000. The corresponding facility-coverage changes were 0.8326 and 0.8114, respectively. This bounded run does not show a task-score structured advantage. On the continuous fidelity diagnostic, no-graph minus full attention-output error was 0.0008 for the structured tasks and -0.0651 for MMStar; positive values favor the graph. The graph now consumes measured prompt-attention co-activation and exact processor visual-grid boxes/frame indices. These datasets still do not expose task-provided OCR strings or document layout regions, so no stronger structured-task claim is made.

3. **Value-aware scoring and fidelity.** Removing value contribution changed mean attention-output error by 0.0023 (G minus A). A positive value favors the value-aware full configuration; a negative value favors the ablation. The mean logit-KL change was -0.0085 (G minus A). The registered equation now uses signed negative lambda_v/lambda_o coefficients, so high contribution and novelty are preservation rewards rather than eviction penalties. Thus value awareness improves the requested attention-output fidelity diagnostic here, while the logit-KL result remains mixed and is disclosed rather than hidden.

4. **Prototype merging at equal active memory.** Prototype-minus-exact per-task score changes are mmstar=0.0000, docvqa=0.0000, chartqa=0.0000, video_mme=0.0000. Its mean attention-output-error change is -1.0041; its logit-KL change is -2.9782. Negative fidelity changes favor prototype merging. Interpret these values only for rows labeled `three_tier`; a safety fallback is not prototype evidence.

5. **Repair benefit versus overhead.** Online full-minus-no-repair per-task score changes are mmstar=0.0000, docvqa=0.0000, chartqa=0.0000, video_mme=0.0000. Its mean end-to-end latency change is 3.0904 seconds, and online repair frequency is 0.0000. Evaluation-only oracle repair frequency is 0.0714, mean measured transfer/re-decode latency is 0.1932 seconds, and quality recovery among triggered events is 0.3333. The registered online policy did not trigger, so this run does not justify its residual-construction overhead. The oracle row is a measured evaluation-only diagnostic, not an online result or fabricated repair benefit.

## Remaining threats

- Two examples per image task and one video example provide no statistical power and are sensitive to discrete task scoring.
- Dirty-tree manifests are non-canonical even though they preserve git SHA, config SHA, and patch SHA.
- Video-MME evidence comes from one locally cached video and repeated decoding was sensitive to local decoder seek overhead.
- Processor-grid boxes are available, but benchmark OCR strings, table cells, and named document regions are not present in the cached task records.
- Qwen M-RoPE prototype support is pinned to Transformers 4.55.2 and eager decode; other model families and fused attention implementations remain unsupported.
