# Prompt-query sensitivity-weighted source fit: CPU feasibility only

Preserve the failed unweighted fit and its key/value counterfactual audit.
The prior fit reduced input SSE but increased attention-output error. Lower
unweighted input error is therefore not a sufficient objective here.

Freeze one alternative before execution. Use the unchanged three-bit, group128,
two-step affine refit, weighted by a diagonal attention-output sensitivity
**bound**, not a full Jacobian. For head h, prompt query q and source position i,
let a be the original causal attention probability and O the original attention
output. Sum each query's contribution before pooling:

    key_weight[i,c] = sum_h sum_q a[h,q,i]^2
        * ||V[h,i]-O[h,q]||^2 * ||Q[h,q]||^2 / d
        * rho[i]^2 * ||W_K[h,:,c]||^2
    value_weight[i,c] = sum_h sum_q a[h,q,i]^2 * ||W_V[h,:,c]||^2
    importance = key_weight + value_weight

Here rho squared is max over original rotary pairs of cos squared plus sin
squared. This follows a Cauchy upper bound on rotated key sensitivity; it does
not pretend cached keys are pre-RoPE. Exact K/V rows contribute zero. There is
no fitted mixture coefficient, benchmark label, averaged-query vector, task
reference, model-weight update or persistent importance tensor. This diagonal
approximation omits cross-channel/position interactions; usefulness is empirical.

Fit group scale/intercept with these nonnegative weights and retain the same
per-group BF16 weighted-error rollback. Input SSE may increase; report it rather
than mislabeling weighted error as unweighted MSE. Persistent codes/headers and
exact-row protections have identical byte sizes. Generic fitting remains
available for compatible baseline bodies; no baseline GPU integration is yet
claimed.

Use only the two consumed captures and the same CPU attention/projection oracle.
Require the unchanged >=5% attention-output MSE reduction and nonincreasing p95
on **both** inputs, plus identical source bytes/protections. Preserve failures.
No GPU run, new cohort or paper promotion is authorized by this numerical plan.
A positive result would still require baseline-compatible runtime integration,
the unchanged 32-token model guard and independent benchmark validation.
