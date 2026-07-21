# Decode-time residual repair

`mosaickv.repair` implements backend-independent, uncertainty-guided repair of
a constructed exact/prototype/residual cache. The controller owns trigger
logic, residual ranking, tier mutation, budget enforcement, timing, and event
records. A backend supplies provisional logits, globally normalized prototype
attention masses, and a callback that recomputes the same decode step from the
promoted cache state.

Current Hugging Face adapters still declare `supports_residual_repair=false`
and cannot decode packed active/logical layouts. Consequently this module is
validated with synthetic cache tensors and callback-driven re-decode; it does
not claim end-to-end repair support for LLaVA, Qwen2.5-VL, OneVision, or
InternVL checkpoints.

## Persistent state

`RepairCacheState.from_construction(full, construction)` starts from a checked
`ThreeTierCacheConstruction`. Initialization rejects a prototype-bearing cache
unless the adapter declared residual repair support. The state records:

- initial and currently exact source node IDs;
- all promoted block IDs, which remain exact on later steps;
- the immutable prototype catalog and currently active prototype IDs;
- prototypes superseded by a promotion versus prototypes evicted for budget;
- remaining residual payloads and their original-position index;
- the construction's block, retained-slot, or byte budget; and
- a strictly step-ordered repair event history.

Logical sequence length and next decode position remain in the unchanged
`LogicalPositionMap`; physical active cost is computed separately from the
current exact and prototype tiers.

## Provisional signals

For provisional vocabulary probabilities `p`, normalized next-token entropy is

```text
H_normalized = -sum(p * log(p)) / log(vocabulary_size).
```

It lies in `[0, 1]`. Callers provide one globally normalized attention mass for
every active prototype; the masses must be nonnegative and sum to at most one.
For prototype `j`, dispersion is the mean of its recorded K and V mean-squared
dispersions, and risk is

```text
risk_j = attention_mass_j * dispersion_j.
```

The step record preserves per-prototype masses and risks, total prototype mass,
maximum risk, and entropy. If a cheap draft probability vector is supplied, it
also records `KL(provisional || draft)`. Draft KL is diagnostic and does not
silently alter a named trigger policy.

## Policies

`RepairConfig.policy` accepts exactly:

- `none`: never repair;
- `entropy`: compare normalized entropy with `entropy_threshold`;
- `prototype_risk`: compare maximum cluster risk with
  `prototype_risk_threshold`;
- `entropy_or_prototype_risk`: trigger when either threshold is met; and
- `oracle`: an evaluation-only policy.

Threshold comparisons are inclusive. `max_blocks_per_step` is the hard `R`
limit. Oracle repair requires `evaluation_only=true`, cannot be invoked through
the online `repair_decode_step` API, and is isolated in
`mosaickv.repair.oracle.evaluate_repair_decode_step`. Only that API accepts an
oracle decision or full-cache reference token. Reference tokens label whether
a wrong provisional token changed to the reference token; they never influence
an online trigger.

## Promotion and budget enforcement

On a trigger, active prototype clusters are ordered by descending current risk
with prototype ID as the deterministic tie-break. Within a cluster, residual
members are ordered by descending construction weight and then source node ID.
At most `R` source blocks are selected.

The parent prototype of a promoted block is superseded and removed from the
active tier, avoiding duplicate exact/prototype membership and a stale partial
average. Unrestored members remain in residual storage but inactive. Promoted
payloads are cast back to source cache dtype/device, added to the exact tier,
removed from residual storage, and retained on subsequent decode steps.

The completed plan is checked against the original construction budget before
transfer. If promotion would exceed it, active prototypes not superseded by
the repair are evicted in ascending `eviction_utility` order. That utility is
the selected anchor's recorded selection marginal gain, and its provenance is
stored in every `PrototypeRecord`. Ties use prototype ID. If no feasible plan
exists even after prototype eviction, no transfer or re-decode occurs and the
event is recorded as `active_budget_infeasible`.

## Transfer and one-pass re-decode

Pinned Torch residual payloads are copied with `non_blocking=true` on a
dedicated CUDA stream. A current-stream event dependency makes the restored
tensors safe for re-decode, while CUDA events measure the transfer interval.
INT8 payloads are transferred encoded and dequantized on device. CPU/mock paths
use a synchronous measured copy and explicitly record that the transfer was
not asynchronous.

After promotion the caller's `re_decode(repaired_state)` callback is invoked
exactly once. CUDA logits use synchronized CUDA-event timing; CPU logits use a
monotonic host timer. A second repair call for the same step index is rejected.
The callback is responsible for reinjecting the supplied typed tiers into its
backend and recomputing the same current token, not advancing decoding.

Each `RepairEvent` records the policy, reason, step, restored IDs and bytes,
transfer mode/time, re-decode time/count, superseded and budget-evicted
prototypes, budget before/after, maximum logit change, token change, and
evaluation-only quality recovery label.

## Validation

The CPU suite covers all policies, deterministic entropy/KL, a deliberately
removed critical block, quality recovery labeling, persistent promotions,
same-step replay rejection, the `R` limit, and lowest-utility budget eviction.
The no-download CUDA smoke is run with:

```bash
sbatch mosaickv/slurm/three_tier_smoke.sbatch
```

Its output is `validation_smoke` evidence only, never a checkpoint-quality or
paper result. Slurm job `17112166` completed this no-download check on an
NVIDIA A100-SXM4-80GB with PyTorch `2.11.0+cu130`: pinned residual transfer
reported the asynchronous CUDA path, restored source block `2`, invoked one
re-decode, and kept the active cache within its construction budget.
