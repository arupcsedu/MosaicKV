# Decode-time residual repair

`mosaickv.repair` implements backend-independent, uncertainty-guided repair of
a constructed exact/prototype/residual cache. The controller owns trigger
logic, residual ranking, tier mutation, budget enforcement, timing, and event
records. A backend supplies provisional logits, globally normalized prototype
attention masses, and a callback that recomputes the same decode step from the
promoted cache state.

Qwen2.5-VL, LLaVA-1.5, and LLaVA-OneVision declare residual repair support for
the audited Transformers 4.55.2 eager path and can decode packed
active/logical layouts through the owned HF runtime. Each requires its exact
model-derived M-RoPE or standard-RoPE transform. InternVL, fused attention,
padding/non-monotonic position maps, and other Transformers versions remain
fail-closed. Synthetic callback tests and real checkpoint evidence are
separate validation levels.

## Persistent state

`RepairCacheState.from_construction(full, construction)` starts from a checked
`ThreeTierCacheConstruction`. Initialization rejects a prototype-bearing cache
unless the adapter declared residual repair support. The state records:

- initial and currently exact source node IDs;
- all promoted block IDs, which remain exact on later steps;
- the immutable prototype catalog, current prototype records, and active IDs;
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

`prototype_risk_min_entropy` optionally gates this risk trigger on a minimum
normalized entropy. This prevents a high-attention prototype from changing an
otherwise high-confidence token. The entropy policy itself is unaffected.

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
with prototype ID as the deterministic tie-break. The default member priority
is its normalized prototype weight times its recorded K/V reconstruction error;
the legacy construction-weight ordering remains an explicit experimental
control. Candidate allocation proceeds in rounds across risk-ranked clusters,
so increasing `R` first broadens evidence coverage instead of repeatedly
decomposing a single cluster. At most `R` source blocks are selected.

Promoted payloads are cast back to source cache dtype/device, added to the exact
tier, removed from residual storage, and retained on subsequent decode steps.
When only part of a prototype is promoted, the default path keeps the remaining
members represented. It subtracts the promoted members from the FP32 weighted
mean, renormalizes the remainder, and updates K/V dispersion algebraically. For
post-RoPE Qwen keys, subtraction occurs in the audited phase-neutral space and
the refreshed key is rotated back to the original anchor position. No
unpromoted residual block is transferred to perform this downdate. A prototype
is superseded only when all its members have been promoted. This removes the
former quality bug where repairing two members of a ten-member cluster also
discarded the other eight members.

The completed plan is checked against the original construction budget before
transfer. A partially refreshed parent is protected from eviction during its
own repair event. If promotion would exceed the budget, other active prototypes
are evicted in ascending `eviction_utility` order. That utility is
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

`ResidualConfig.require_pinned_memory=false` now selects an explicitly pageable
CPU allocation; it does not attempt a best-effort pin. This makes pinned versus
pageable placement experiments auditable. Every HF trace records the requested
placement, observed payload placement, storage dtype, initial residual bytes,
and whether all payloads were pinned. A requested/observed mismatch is an
Experiment-4 validation failure.

After promotion the caller's `re_decode(repaired_state)` callback is invoked
exactly once. CUDA logits use synchronized CUDA-event timing; CPU logits use a
monotonic host timer. A second repair call for the same step index is rejected.
The callback is responsible for reinjecting the supplied typed tiers into its
backend and recomputing the same current token, not advancing decoding.

Each `RepairEvent` records the policy, reason, step, restored IDs and bytes,
transfer mode/time, re-decode time/count, superseded and budget-evicted
prototypes, budget before/after, maximum logit change, token change, and
evaluation-only quality recovery label.

The HF runtime enriches triggered event traces with the triggering prototype
clusters and restored source descriptors. Cluster records include layer/head,
anchor, current attention mass, dispersion, risk, modality composition, and
logical span. Restored block records include modality, original positions,
media/page/frame metadata, bytes, residual dtype, and actual placement.

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

The bounded Qwen2.5-VL-3B v17 ablation produced real pinned residual tiers for
all applicable full/prototype rows. The registered online
`entropy_or_prototype_risk` thresholds made zero promotions. Evaluation-only
oracle repair triggered on 7.14% of diagnostic decode steps on average,
measured 0.1932 seconds mean transfer plus re-decode latency, and labeled 33.33%
of triggered events as quality recovered. These are exploratory one-/two-
sample diagnostics: they prove the path is operational but do not justify the
online residual overhead or constitute a quality claim.

The follow-up Experiment-4 sweep completed 105/105 paired sample-policy rows
and 686 decode-step records on A100 GPUs. Its stable v2 matrix used one Git SHA
and one dirty-patch SHA across all 60 run manifests. Of seven fixed examples,
one no-repair answer differed from FullKV and was oracle-repairable. Entropy
repair and evaluation-only oracle repair each improved that ChartQA score;
the primary combined policy instead changed it to a different wrong number. A
conservative combined threshold also changed one DocVQA answer incorrectly,
cancelling its ChartQA improvement.
Mean synchronized repair-event latency ranged from 0.330 to 0.506 seconds for
the primary triggering policies, while paired end-to-end cost ranged from
5.306 to 13.220 seconds for the online policies. These negative and positive
cases are both retained in the
[qualitative appendix](../analysis/residual_repair_qualitative.md). The complete
[paired report](../analysis/residual_repair.md) remains exploratory dirty-tree
evidence, not a paper result.

A targeted quality-fix follow-up then evaluated the partial-prototype downdate,
reconstruction-error ranking, and entropy-gated risk trigger on the same seven
examples. All 44 configurations and 77 paired rows completed. The guarded
prototype-risk policy repaired the sole compressed benchmark failure, raised
the score sum from 4/7 to 5/7, introduced no incorrect answer changes, and used
16 repair events versus 38 for the earlier primary combined policy. Its paired
mean end-to-end cost was 3.983 seconds. The guarded combined policy also reached
5/7 without the earlier DocVQA regression, but cost 10.748 seconds.

FP8 and INT8 retained the quality recovery while halving mean residual storage
from about 53.60 MiB to 26.80 MiB; they did not improve latency because the
restored payload per event was only about 13.4 KiB and re-decode/dequantization,
not host-transfer volume, dominated. Likewise, `R=1`, `R=2`, and `R=4` all
repaired the one observed failure. Larger `R` therefore has no measured quality
justification in this sample; `R=1` is the bounded-study latency/byte default.
See the separate [quality-fix analysis](../analysis/residual_repair_quality_fix.md).
These remain exploratory dirty-tree measurements, not paper evidence.
