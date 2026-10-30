# Storage-funded precision-risk exceptions: bounded development ablation

Do not tune against benchmark answers, inspected answer failures or new IDs.
The proposer receives only original selected K/V and prompt-derived quantities.
Fix two ablations before GPU execution: absolute reconstruction-error ranking;
and error weighted by last16 prompt queries (keys) / actual prompt attention
demand (values). Queries are invoked read-only from captured normalized input,
under inference mode; no model parameter changes or gradient steps occur.
Prompt-derived evidence is an approximation, not a guarantee about future text.

Each component may restore <=0.5% original scalar values. Every int32-index/BF16
exception costs6 bytes and must be funded entirely by that layer's lossless
protected-text/map savings. The corrected packet cannot exceed its uncorrected
raw-text3-bit representation. Deterministically break ties by source index.
Original mandatory values, K/V identities, RoPE phases and logical positions
are preserved; no prototype, offload, residual repair or extra history eviction.

Four workers: same two consumed inputs x mse/query. One A10012CPUs30minutes each,
<=6 aggregate GPUs, only after the common-control array finishes. The same
unchanged32-token KL/logit/token guard and128-token exact-overlay/repack control
remain. All transport-valid cases get5 warmups20 randomized paired repeats,
including quality failures. Separately derive EOS-aware FullKV-reference ROUGE-L
at32/64/128 against the hash-bound compatible FullKV predictions. These are
model-reference single-example diagnostics, not official task-score evidence or
independent confidence intervals. No fresh/calibration/paper advancement occurs.
If quality still deteriorates, stop this registered fraction rather than enlarge
it after seeing outcomes. Preserve every failure and timing/storage trade-off.
