# Choice C algorithm and system improvement report

## Status and scientific scope

This report records an exploratory engineering campaign for MosaicKV. The
worktree was dirty, so none of the numbers below is paper-eligible. The result
does not establish general superiority across models or tasks. It establishes
that one frozen MosaicKV configuration passed a preregistered, equal-active-byte
fidelity gate on LLaVA-1.5-7B and DocVQA, first on internal-validation examples
and then on a once-only held-out set.

The primary gate was frozen before held-out evaluation:

\[
E=(1-\text{token agreement})+\log(1+\text{logit KL})+
  \text{attention-output error}.
\]

MosaicKV had to reduce this error by at least 30% relative to the best eligible
baseline, use the same measured active KV bytes within 2%, remain within a 0.05
task-score noninferiority margin, and exercise genuine exact, prototype, and
residual tiers. The baselines were Prompt Attention Top-K, `lookm_reimpl`, and
`vl_cache_reimpl`. FullKV was the fidelity reference, not an equal-memory
competitor.

## Algorithm changes that survived validation

The frozen profile is `draft8_prompt16_draft_trust` with SHA-256
`52309b83f0b5ba115977168ce9f540788b77102f60418acff32512206b19cc75`.
It combines the following mechanisms.

1. **Prompt-anchored selection.** Fifteen percent of selectable capacity is
   reserved for prompt anchors before the remaining budget is optimized. This
   protects instruction and question evidence that a vision-heavy utility can
   otherwise displace.
2. **Label-free fidelity calibration.** Candidate profiles were selected using
   FullKV token, logit, attention-output, and tail fidelity only. Dataset labels
   and task scores did not enter calibration. The chosen candidate won 60.6% of
   1,000 bootstrap resamples, above the frozen 55% stability threshold.
3. **Per-head prompt/draft routing.** Stable heads use the last 16 prompt query
   positions. Shifted heads trust eight temporary greedy draft queries and give
   zero weight to stale prompt queries. The draft reuses the original prefill
   and is discarded before final decoding.
4. **Fine evidence preservation.** Fifteen percent of capacity is reserved for
   evidence coverage, with a 10% minimum stratum fraction. This prevents a
   globally strong modality or evidence group from consuming every slot.
5. **Sparse evidence graph.** Each node keeps at most four compatible
   neighbors. The smaller degree was retained because it avoided diffuse
   coverage observed with eight neighbors in design-only tests.
6. **RoPE-safe prototypes.** Each prototype covers at most 15 blocks and 255
   logical positions. Keys are neutralized to a common rotary frame, pooled in
   FP32 with forecast-attention weights, and rephased at a
   forecast-weighted-member-medoid anchor. Incompatible assignments fail closed
   to exact selection.
7. **Residual preservation.** Prototype source blocks remain in lossless CPU
   residual storage, indexed by their original logical positions. The online
   repair policy is disabled in this frozen profile because calibration did not
   establish a sufficiently precise trigger. Therefore this campaign validates
   three-tier representation, but it does **not** validate a quality or latency
   benefit from online repair.

Several attempted changes were rejected rather than silently retained:

- shorter prototype spans harmed fidelity;
- an eight-neighbor graph was worse than four neighbors on design data;
- draft horizons were unstable until the larger label-blind calibration set
  selected the eight-step profile;
- restoring one, two, or four residual blocks changed bytes but did not change
  the tested token decisions;
- unordered segmented GPU reductions changed two near-tied greedy decisions,
  so prototype reductions preserve the original member order.

## System changes

The implementation now:

- batches rotary-phase neutralization and rephasing by compatible tensor shape;
- batches member gathering while retaining deterministic, ordered FP32
  prototype reductions;
- caches invariant source/tier membership sets used by repeated state checks;
- uses a vectorized, stable top-k path for unrestricted sparse graph neighbors;
- retains the original per-head prompt-attention reduction order because a
  more aggressive batched reduction weakened numerical fidelity.

On the 16-sample internal parity set, selected blocks, generated tokens, logit
KL, and attention-output error matched the pre-optimization implementation for
all 16 samples. Mean tier-construction time fell from 31.0880 s to 13.3570 s,
graph time from 7.7182 s to 6.7596 s, and total MosaicKV evaluation-path time
from 55.8292 s to 37.5283 s. The total reduction was 32.78%.

This is not yet a low-latency compressor. On the held-out set, mean online
component times were 1.3459 s for forecasting, 6.7814 s for graph construction,
1.8824 s for utility and selection, 13.3481 s for tier construction, and
2.9641 s for cache packing. Mean compression time was 26.3218 s. Prompt
Attention Top-K compressed in 2.0502 s, so the present Python/HF prototype path
trades substantial latency for fidelity. Prototype construction and graph
construction are the next kernelization targets.

## Frozen validation results

### Internal validation

The 16-example internal gate passed. MosaicKV's fidelity error was 0.872525;
the nearest equal-byte baseline was `lookm_reimpl` at 1.583122. The relative
improvement was 44.89%, with a paired-bootstrap 95% confidence interval of
18.66% to 73.90%. MosaicKV and FullKV both had mean task score 0.125000.

### Once-only held-out evaluation

Slurm job `18027457` used five A100 GPUs and completed successfully in 35:29.
All five methods completed the same 32 sample IDs with no failed rows.

| Method | Samples | Mean task score | Mean fidelity error | Mean active KV bytes | Mean residual bytes | Mean compression (s) | Mean end-to-end (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FullKV | 32 | 0.210646 | 0.000000 | 319,766,528 | 0 | 0.0000 | 0.8268 |
| Prompt Attention Top-K | 32 | 0.164062 | 2.510175 | 59,719,680 | 0 | 2.0502 | 15.8229 |
| `lookm_reimpl` | 32 | 0.125000 | 2.011917 | 59,719,680 | 0 | 32.5113 | 61.5460 |
| `vl_cache_reimpl` | 32 | 0.128906 | 3.572143 | 59,719,680 | 0 | 1.7554 | 28.5183 |
| MosaicKV Full | 32 | 0.211566 | 1.002385 | 59,719,680 | 265,814,016 | 26.3218 | 37.4703 |

Against the nearest fidelity competitor, `lookm_reimpl`, MosaicKV reduced the
frozen error by **50.18%**. Its paired-bootstrap 95% interval was **23.28% to
70.87%**. All three compressed baselines used exactly the same measured active
KV bytes as MosaicKV. MosaicKV's mean task score was 0.211566 versus 0.164062
for the best baseline task score; the paired-bootstrap difference interval was
-0.031250 to 0.137454, so this small sample does not establish a statistically
significant task-score improvement.

## What is and is not established

Established for this exploratory configuration:

- the preregistered 30% equal-active-byte fidelity criterion passed on the
  once-only held-out set;
- the outcome was not obtained by inspecting held-out examples and retuning;
- exact/prototype/residual tiers were active and residual bytes were measured;
- the deterministic system optimization retained 16/16 parity while reducing
  the implementation's total evaluation-path time by 32.78%;
- all five GPU workers and the overall Slurm allocation exited successfully.

Not established:

- a universal 30% advantage across models, tasks, retention ratios, or output
  lengths;
- a statistically significant task-score advantage;
- a latency or throughput advantage over simple baselines;
- a benefit from uncertainty-guided residual repair, because the frozen policy
  correctly remained disabled;
- a paper-ready result, because the source worktree was dirty.

The held-out IDs have now been opened and must never become design data. The
next scientific step is to freeze and commit this implementation, create new
disjoint ledgers, and replicate across at least one general VQA task, one chart
task, another model family, and multiple retention ratios. The next systems
step is to replace Python prototype grouping and graph construction with
fused/batched device kernels while retaining the exact deterministic parity
tests.

## Artifact locations

- Frozen profile:
  `/scratch/djy8hg/runs/mosaickv/choice-c-forecast-stability-design-v15/frozen_fidelity_profile.json`
- Internal gate:
  `/scratch/djy8hg/runs/mosaickv/choice-c-ordered-quality-system-v22/internal_validation_gate.json`
- Held-out gate:
  `/scratch/djy8hg/runs/mosaickv/choice-c-ordered-quality-system-v22/heldout_gate.json`
- Held-out rows:
  `/scratch/djy8hg/runs/mosaickv/choice-c-ordered-quality-system-v22/heldout_rows.parquet`
- Held-out summary:
  `/scratch/djy8hg/runs/mosaickv/choice-c-ordered-quality-system-v22/heldout_summary.parquet`
- Held-out traces:
  `/scratch/djy8hg/runs/mosaickv/choice-c-ordered-quality-system-v22/traces/choice-c-heldout-mosaickv_full-dfcb0a7f9a/`
