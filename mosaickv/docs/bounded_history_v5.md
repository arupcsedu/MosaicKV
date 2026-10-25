# V5 draft-logit allocation guard

V4 remains a negative development result. Its immutable recovery report is
hash-bound in `configs/experiments/bounded_history_memory_v5.yaml`. No V4 sweep
is extended and none of its answers or individual failure identities select V5
samples, capacities, thresholds, or exchanges.

## Change and rationale

V4 improves a prompt attention-output reconstruction surrogate, but its eight-
example-per-task test regressed on corpus PPL. Two consumed-identity diagnostics
also showed that four-token greedy agreement cannot certify a 1,024-token answer.
V5 therefore retains the original exact-byte proposal builder and its search
limits, then tests the **entire joint allocation** with actual model logits.

1. Reuse the original prefill; never prefill again for a probe.
2. Clone independent FullKV, compressed-anchor, and proposed cache states.
3. Run a deterministic FullKV greedy draft for 32 updates. Feed that same
   model-generated prefix to both compressed states, using their normal
   generated-history transport, including ring wrap/eviction.
4. At **every** prefix, require candidate KL and maximum logit difference from
   FullKV to be no worse than the anchor, within frozen numerical tolerances
   `atol=1e-7`, `rtol=1e-6`. Forbid a new greedy disagreement where the anchor
   agrees with FullKV. Explicitly report checkpoints 1, 4, 16, and 32.
5. Publish the whole proposal only if all checks pass; otherwise publish the
   unchanged same-byte compressed anchor. Never substitute FullKV.
6. Discard private probes and verify the original cache tensors/logical counters
   are unchanged. One-token PPL requests perform no draft or decode preallocation.

Neither targets, benchmark references, scores, nor evaluation-only future-query
oracles are accepted by the guard API. Generation and teacher-forced evaluation
with the same prompt budget must make identical selection decisions. Original
K/V values and logical positions are retained; no prototype or residual is added.
The variant is explicitly named `mosaickv_draft_logit_guard_v5`, not silently
substituted for historical `mosaickv_exact` results.

This is a conservative finite-horizon hypothesis, **not a guarantee of answer
quality**. Passing on a forced FullKV prefix need not preserve a free-running
long answer. All-rollbacks may preserve the anchor but establish no new benefit.

## Accounting and trace contract

The live output cache has exactly the same physical initial/final/peak budget
as every compressed comparator, separately in generation and PPL. Proposals
are identified as proposals; final position hashes use original **logical**
positions and report the actual post-guard selection and layer capacities.
Guard time is charged to compression, TTFT, and end-to-end latency, and receives
its own logit-stability phase. FullKV is excluded from latency comparisons.

Private FullKV/anchor/proposal caches and verification copies are additional
temporary GPU allocations. Their byte counts are reported separately; they
are included in peak GPU memory, not hidden as reductions in total HBM use.
An eventual smaller *live cache* claim is not a smaller process-peak-memory claim.

## Gated execution, in order

1. CPU configuration, later-prefix rejection, atomic rollback, exact-byte,
   mandatory-token/position, no-label-access, and state-isolation regressions.
2. From an authorized clean commit, run the V5 development preflight: eight new
   examples each from LLaVA-Description and paper-style MM-Vet; FullKV reference,
   Prompt Top-K, original MosaicKV exact anchor, and V5; 40 MiB compressed cap;
   identical FP16/eager model revision, 1,024 outputs, five warmups, and seed.
   This is eight workers: two references followed by six compressed workers,
   at most six aggregate A100s, one GPU/12 CPUs/two hours per worker.
3. IDs exclude every explicit V1--V4 development ledger (22 consumed IDs/task).
   The parent cohorts were used historically: these are disjoint development
   examples, **not globally untouched held-out evidence**.
4. Require complete metrics/provenance, parity, exact storage in both phases,
   valid guard traces, and task-level preliminary ROUGE-L/PPL nonregression.
   Eight examples cannot create `frozen_candidate.yaml`. Stop if quality fails.
5. Only after positive development evidence: preregister a larger, separately
   disjoint calibration, quality target, strongest-eligible-competitor rule,
   multiplicity correction, power calculation, and stopping rules. Freeze the
   candidate only after consistent quality preservation; then run a genuinely
   independent powered confirmation. Sample growth cannot remove a real loss.
6. Only after confirmation: measured-byte curves with every compressed method
   at every budget. Same-byte curves establish quality comparisons. Separately
   estimate storage needed for the same frozen quality target; never call
   unequal-byte points byte-matched. Report negative points and latency honestly.
7. Validate the frozen candidate separately in the PrefixKV PPL/ROUGE-L,
   VL-Cache CIDEr/ANLS, and representative LOOK-M/MileBench Accuracy/ROUGE-L
   lanes. Do not mix metrics, call local variants official, or assume the
   LLaVA-1.5 bounded-history transport works on VL-Cache's Mistral checkpoint.
   Adapter support and controls must be established before that separate lane.

No automatic larger campaign, powered validation, or paper promotion is enabled
by this preflight. Prototype and repair claims remain excluded.
