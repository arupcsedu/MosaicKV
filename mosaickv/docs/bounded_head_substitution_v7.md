# One-head V7 mechanistic diagnostic

## Completed recovery and V6 disposition

The V6 analyzer failed because its gate could contain a NumPy boolean, which
standard JSON cannot serialize. The fix converts gate decisions to native
booleans and supports NumPy scalar serialization explicitly. Serialization now
finishes before a write-once file is opened, so type errors cannot leave a
partial output. Non-finite numbers and unsupported objects still fail closed.

The completed CPU-only recovery is at
`/scratch/djy8hg/runs/mosaickv/bounded-history-v6-derived-analysis-20261025-v1/`.
It contains 64 rows, complete corpus-PPL and FullKV-reference ROUGE-L summaries,
10,000-replicate paired intervals, and analysis source/patch provenance. All
producer artifact hashes were checked. Original predictions and the incomplete
`analysis/report.json` remain unchanged. No GPU inference was rerun.

`bounded_history_memory_v6_negative_seal.json` binds the original and derived
reports plus both eight-example identity ledgers. All 16 identities are
consumed. Both V6 task screens failed; no calibration, frozen-candidate or
paper campaign is authorized. Aggregate failures motivate this diagnostic;
sample answers and failure identities must not guide its choices.

## Smaller, physically safe transaction

V6 moved one token in every KV head of a donor/receiver layer. Its LLaVA FP16
transaction changed 16 KiB of membership. V7 instead replaces one retained
nonmandatory token with one omitted token **inside a single layer/KV head**.
For that same checkpoint this changes 512 bytes of membership (a layout-derived
quantity, not a measured memory reduction). Every head keeps exactly the same
capacity. The hard lifetime cap remains 40 MiB, with zero padding and unchanged
generated-history policy. Original source K/V are gathered verbatim, retaining
post-RoPE keys and original logical positions. No prototype or residual tier
is enabled. This is not yet a new online method or a smaller-memory claim.

Packing copies the changed layer and aliases untouched immutable layers. Each
model probe then owns private cache buffers. Source tensors, next decode
position, and original logical length must remain unchanged after all probes.

## Actual-logit shortlist, not prompt-best plus veto

1. Build a bounded pool of at most 16 route-diverse one-head substitutions.
   Prompt statistics generate proposals without requiring positive prompt
   reconstruction gain; these statistics do not decide the final shortlist.
2. Reuse the original FullKV prefill. Evaluate each proposal for up to four
   FullKV-generated greedy draft prefixes; stop on its first guard violation.
3. Construct the two-plan shortlist from the **actual** draft disagreement
   count, summed KL, and summed maximum logit error, with deterministic ID ties.
   Discarded proposals are never extended. Only the best two private states are
   retained, plus the candidate currently being screened.
4. Extend the shortlisted states through all 32 prefixes, without replaying the
   short prefix or performing a second prefill. Original-anchor/FullKV draft
   evidence is shared and advanced lazily once per prefix.
5. Keep V6's `atol=1e-7`, `rtol=1e-6`, every-prefix KL/logit non-regression,
   greedy-token protection, and checkpoints 1/4/16/32 unchanged. An early pass
   is not acceptance. A late violation rejects the proposal immediately.
6. A meaningful modification requires a full-horizon pass and at least one
   numerical improvement beyond the unchanged tolerance, or recovery of a
   draft greedy-token disagreement. A numerically identical swap does not
   satisfy this diagnostic's readiness condition.

The maximum is 16×4 + 2×28 candidate updates plus 32 FullKV and 32 anchor
updates: 184 decode calls, before fail-fast savings. This bound is analytical,
not a latency measurement. Record pool, packing, state construction, screening,
extension, wall time, actual probe calls, and temporary KV/GPU memory. The
diagnostic does not generate benchmark answers or compute benchmark scores.
Passing finite draft checks does not establish answer-quality preservation.

## Scope and launch boundary

The versioned protocol selects only the first already-consumed V6 identity per
task by ledger order: one LLaVA-Description and one paper-style MM-Vet input.
Preparation creates prompt/media-only files. References, target text, scores,
and extra annotation fields are stripped before worker inputs are written.
No fresh identity is selected or materialized.

After a separately authorized scoped commit makes the source clean:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/head_substitution_v7.py \
  submit --root /scratch/djy8hg/runs/mosaickv/bounded-head-substitution-v7-20261025-v1
```

The launcher submits only two one-A100/12-CPU workers with two-hour limits under
`bi_fox_dgx`. Both start held and join the existing aggregate-six-A100 admission
controller. Workers enforce clean source/config/input hashes, deterministic
CUDA, FullKV/reinjection parity, private-state isolation, and full manifests.
Failed diagnostics retain their error record; no failure becomes a passing
quality row. Existing results and paper files are untouched.

Do not materialize a fresh cohort unless the consumed-input diagnostic first
shows accepted, meaningful changes with correct bytes and complete traces.
Even then, a fresh small disjoint development test needs separate execution
scope; calibration and all larger campaigns remain blocked. There is no claim
that this change will necessarily pass on the real checkpoint.
