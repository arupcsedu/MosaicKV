# Forecast-ranked exact-head bypass — bounded consumed diagnostic

Preserve modality-normalizer v1 as negative. It reduced average modality-mass
error but failed the unchanged MM-Vet 32-prefix guard. Mass error had weak
association with logit error there; do not extend that scalar correction sweep.

## New representation and reason

The ordinary compressed cache is unchanged and uses the existing bit-exact
protected-text storage. Restore full original prompt K/V for only 1, 4, or 8
forecast-ranked heads. A separate head-local exact buffer, not a larger dense
32-head tensor, holds each such context. This avoids padding every unrelated
head when adding evidence to one head. Original keys already include RoPE;
copy them bitwise and use the actual current query's original rotary phase.
No donor, key/value prototype, residual offloading, or benchmark label is used.

Rank ALL 1,024 layer/head routes from four isolated FullKV draft queries:
compare original prompt attention outputs with retained, renormalized outputs,
then measure their squared difference after that head's frozen o_proj slice.
Keep per-query/head information before reduction. Break ties by layer then head.
This is a magnitude-preserving output-impact surrogate, not an entropy score or
a guarantee of downstream logit fidelity. Do not use the inspected wrong-token
position, benchmark answer, or saved 32-prefix reference to rank routes.

Each decode handle privately allocates the sidecar's full prompt plus 128 raw
history slots from the initial boundary. Append its actual current K/V from the
main cache, then recompute only its head attention before the original o_proj.
No full-size K/V concatenation is created. The unchanged other heads and original
weights remain intact. Immutable constructor seeds are not mutable request state;
forked handles must not share history owners. Charge the complete sidecar and
history reservation, not just used rows. Metadata, base storage, original RoPE,
and decode leases remain accounted. Whole-request KV peak is still uncertified.

## Installed interface

Use only Transformers 4.55.2 eager LlamaAttention, full MHA, head dimension 128,
evaluation mode, one query and an unpadded batch-one prompt. The inspected source
is the common environment's `transformers/models/llama/modeling_llama.py`:
`LlamaAttention.forward`, `apply_rotary_pos_emb`, `eager_attention_forward`,
`q_proj`, and `o_proj`. Scope hooks to one request and remove them on exceptions.
Other attention backends/models fail explicitly. Retention-1 ordinary execution
is untouched; no production CLI enables this diagnostic representation.

## Frozen experiment and controls

Use only the same consumed Description and MM-Vet inputs. Do not change weights,
processor, media, prompts, BF16/eager precision, source ratio, codec, selection,
mandatory policy, or 128-prediction/first-EOS scoring. The base is source-only,
not the delimiter variant that worsened MM-Vet. No fresh cohort is authorized.
The graph/submodular and prototype/repair mechanisms remain disabled and labeled.

Four variants: a one-head inactive shadow, and active 1/4/8-head exact bypasses.
The shadow and active one-head run must match initial/final/decode-peak bytes;
the 4/8-head points are larger causal interventions, not byte-matched baselines.
Each plan must remain within its original measured anchor initial byte cap.
All eight routed heads applied to a FullKV clone must reproduce its 32-prefix
logits within 0.001 and identical greedy tokens. Shadow execution must match the
saved source-only ALL-128-logit trajectory bitwise. Preserve failure tensors
before postconditions; don't rerun inference for a post-processing defect.

Require complete CPU suite before two one-A100, 12-CPU, 30-minute workers. Maximum
six A100s in aggregate, common environment, scratch-only caches, finite afterany
analysis. The original FullKV source, inactive candidate seeds, and construction
workspaces are excluded from decode-owned peak and must not be called total GPU
or whole-request KV superiority. Report this exclusion beside every memory claim.

The acceptance guard remains 32 prefixes, KL<=0.02, max logit difference<=1, and
token agreement=1. A useful common configuration must improve an actual model
metric without losing Description or MM-Vet FullKV-reference ROUGE-L relative to
the source-only anchor. ROUGE-L here is not official MM-Vet judging or PPL. Stop
and diagnose if no meaningful safe change passes. Do not add more heads or launch
a fresh/larger cohort simply because execution succeeded. This diagnostic cannot
establish superiority over LOOK-M, PrefixKV, or VL-Cache or authorize a paper run.

The first GPU launch failed the strict FullKV numerical control before quality
evaluation: independent one-head BF16 GEMMs differed from native batched MHA.
Keep v1 execution-negative. The corrective retry retains native head batch shape
using zero-stride K/V views, not duplicated K/V buffers. It performs more attention
compute and must report that cost. Save full control logits and head-output error
telemetry. The 0.001 control and all 32-prefix quality thresholds are unchanged.
