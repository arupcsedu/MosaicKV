# Early prompt-loss diagnostic and separated research tracks

Prof. Fox's original exact-selection/append-only evaluation remains separate
from the PrefixKV bounded-history adaptations. Neither result category can
replace or repair the other's evidence. This study reuses only consumed
Description `000000532947` and MM-Vet `v1_197`; it is not confirmation.

Use the pinned FullKV, Prompt Top-K and mixed-prior exact configurations from
token-eviction plan
`c74c12fb58ecf38cad5d6a2cec7dfe70e13e16d4327c5662aca5a3516fe24c6a`.
Replay the first 32 **saved FullKV greedy tokens**, never annotation targets.
Do not alter priors, selectors, weights, cache budgets, logical positions,
processor, backend or precision. The bounded transport appends during these
31 updates and cannot evict history. Description is a consumed negative control
for the early MM-Vet answer divergence; neither is statistically independent.

For each role compare an unobserved replay with a scoped read-only capture.
Require bitwise identical complete logits, selected prompt and initial/final/
post-compression transaction-peak storage. Recheck 16-token ordinary-generation
and extraction/reinjection parity. Hooks must be removed after success or error.
Capture actual attention-head outputs, projected attention, decoder residual
inputs, normalized attention inputs, post-attention residuals and layer outputs
at predictions 2/4/8/16/20/21/22/23/24/32. Record actual model logits across all
32 predictions. Captured times are not latency evidence.

At prediction 23, separately decompose omissions using FullKV probabilities,
FullKV values and each frozen compressed selection. Include all generated
positions in retained mass and explicitly retain softmax rounding. For exact
arithmetic, `full - retained = sum_omitted p_i * (v_i - retained)`; finite-dtype
softmax adds `retained * (sum_i p_i - 1)`. Report output-direction contributions,
not attention mass alone, without averaging query heads. This is frozen-query
algebra, not a prediction of changed queries or later-layer propagation, and
not causal evidence that restoring a block will improve an answer.

Only a subsequently verified same-byte restore/donor intervention could justify
changing selection. A diagnostic with nonzero layer error or high omitted-block
scores is insufficient. No automatic prior sweep, calibration, fresh cohort or
larger campaign is authorized here. Event-driven `afterany` CPU analysis must
also retain failed worker records. Maximum two one-A100 workers, each 12 CPUs,
16 GiB host memory and 20 minutes; aggregate MosaicKV occupancy stays below six.
