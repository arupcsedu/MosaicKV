# Zero-step forecast: remove the unused full-cache clone

This is an implementation correction, not a new quality-selection algorithm.
The zero-token rollout previously cloned the complete prefill cache only to
obtain each layer's KV-head count. No draft execution used that clone. The same
module already has a metadata-only layout reader used by conditional forecasting;
use it for every zero-step rollout. Nonzero isolated rollouts remain unchanged.

Report zero cache-clone time and account metadata-read time in query preparation
and total forecast time. Do not disguise metadata work as free or alter generation,
selection, prompt/history capacities, stored K/V, original positions or thresholds.

CPU regression forbids cache extraction/clone and decode on this path, and checks
that all source tensors are unchanged. The complete CPU suite is also required.

After the V5 frontier finishes completely, use its four EviCastKV endpoint arms
(128 and 512 history positions on the same two consumed inputs) as frozen numerical
controls. Re-execute only those four arms on the corrected source. Their config
hashes must remain identical; require exact generation/teacher-forced token lists,
selected blocks, answer, NLL, token count, PPL and ROUGE-L equivalence. Initial,
final and common active decode-transaction peaks must also match.

Require reduced global registered-owner peak. This covers the removed temporary
source snapshot and must not be mislabeled as a further reduction in active-cache
capacity. Record CUDA-allocator peak separately. Generic whole-request/internal
KV-transient certification remains false; no numerical or statistical quality
improvement follows from an equivalent implementation optimization.

Four one-A100/12-CPU workers, maximum six A100s aggregate, 30-minute limits, one
finite afterany CPU analyzer. No fresh cohort or paper campaign is authorized.
Single-trial timing differences are descriptive, not a repeated latency win.
