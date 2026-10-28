# Frozen V3 same-consumed-input HF verification

This protocol freezes the CPU-selected **anchor-preserving joint correction**,
not the ridge-only or delta-only ablations. The effective method is
`attention_matching_anchor_preserving_reimpl`; it is not `mosaickv_exact` or a
paper-quality result. V1/V2 negative artifacts and the V3 CPU study remain
untouched in their original worktrees/output roots.

## Frozen construction

- Pinned LLaVA-1.5-7B HF checkpoint/revision, FP16 and eager attention, identical
  processor, prompts, media hashes, and two consumed sample IDs.
- The same 39 original key positions per layer/head from the parent GPU capture.
  This dictionary was selected with all 32 draft prefixes. It is deliberately
  fixed here: this is not an independent dictionary-selection or benchmark test.
- One original prefill; a deterministic 32-prefix FullKV draft and zero-bias
  retention-1 control. No second prefill or benchmark-reference access.
- Joint mass/value correction with coefficient fitting restricted to prefixes
  **1–16**. Original keys, logical positions, mandatory first/last values and
  their zero biases remain unchanged.
- Fixed 256-step projected value fit, relative ridge **1e-2**, change bound
  `||delta V|| <= 0.1 ||V0||`, and unchanged final norm ceiling `||V|| <= 4||V0||`.
  Original bounded mass fit, multiplier ceiling 64, and training-only admission
  threshold 1e-8 remain unchanged. No model weights are trained or updated.

The CPU decision is hash-bound to its plan, report, and decision. Preparing the
GPU protocol checks the selected case/component and every inherited source/input
hash. The new plan stores a separate frozen dictionary JSON and media hashes for
each worker. Only the same two consumed inputs can be submitted.

## Unchanged model-level acceptance

The shared isolated HF transport is unchanged. It counts unique persistent K/V,
FP32 biases, masks, and decode metadata against the **28 MiB** cap. Each head has
39 prompt positions plus a 16-position generated-history ring with four sinks.
The FullKV reference/control states are private diagnostic workspaces; their
memory and global GPU peak remain separately reported, not hidden as live KV.

Require bitwise zero-bias/retention-1 logits and tokens over all 32 prefixes.
Run same-byte anchor, independent clone, candidate, and source-repacked anchor
in A–B–A order. Clones/repacked states must match the anchor bitwise, with source
K/V, logical positions, weights and attention-mask scope unchanged.

The candidate must pass the **original 32-prefix** KL/logit/token guard relative
to the same-byte exact zero-bias anchor: atol 1e-7, rtol 1e-6, checkpoints
1/4/16/32, no new greedy disagreement. Early rejection stops on the first
violation; acceptance and meaningful improvement require the complete horizon.
GPU-FP32 telemetry remains authoritative; FP64 KL stays sideband only.

An automatic CPU analyzer replays every stored coefficient from the captured
probabilities/outputs on **only the first 16 prefixes**, requiring bitwise equality
with stored values and biases. It independently verifies actual stored-dtype
10% delta and 4x value bounds, mandatory entries, original dictionaries, complete
provenance, storage, timings, and the existing model guard. Missing/failed workers
are retained as failures; they cannot authorize promotion.

## Execution

Create one scoped immutable source commit before preparation. The GPU protocol
requires a clean tracked and untracked worktree and records that SHA. Do not
modify it while workers or their dependent analyzer run.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
bash mosaickv/scripts/assert_clean_worktree.sh
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/anchor_compaction_hf_v3.py submit \
  --root /scratch/djy8hg/runs/mosaickv/anchor-compaction-hf-v3-20261027-v1
```

Two array workers request one A100, 12 CPUs, 96 GiB CPU memory and 30 minutes
each on `bii-gpu` / `bi_fox_dgx`. The finite `afterany` analyzer uses one CPU,
4 GiB and 10 minutes. A single admission pass uses the corrected, hash-preserved
Slurm-time parser and aggregate six-A100 policy. There is no watch/log polling
controller. Every cache is rooted under `/scratch/djy8hg`; model/processor loading
uses the existing common environment and local files only.

Outputs are write-once: plan, redacted inputs, dictionaries, per-prefix logits
and tokens, captured evidence, fitted/original K/V and biases, phase events,
storage records, manifests and analysis/decision artifacts. The manifest labels
the real V3 method and 28 MiB total-persistent accounting, not the parent
40 MiB control-config sentinel or the old V1 fitting algorithm.

## Stopping rule

On failure, preserve the diagnostic and stop: no retry of this version, fresh
cohort, calibration, or larger campaign. A pass permits only a design review
toward independently authorized benchmark-quality validation; it establishes
neither a compression advantage over a competitor nor paper readiness. All
fresh-cohort, calibration and paper-campaign flags remain false.
