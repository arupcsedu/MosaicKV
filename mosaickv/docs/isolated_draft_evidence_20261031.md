# Isolated four-step future-demand diagnostic

This is separately versioned from the frozen balanced-allocation validation.
Do not alter that candidate or use its remaining-112 outcomes to set these
parameters. Use only the same two consumed native-route inputs for this initial
diagnostic; no fresh benchmark cohort or paper campaign is authorized here.

Reuse the original FullKV prefill. The existing audited isolated rollout bridge
clones its cache and state, executes four deterministic greedy steps, captures
actual per-head attention, checks source-cache integrity, and discards draft
state. Do not prefill again, change weights, use benchmark answers, or use
saved oracle outputs as algorithm inputs. Compare the isolated draft's four
logit rows to the saved ordinary FullKV path only in the diagnostic controller.

Strip generated-history attention mass and normalize original-prompt demand
per head before averaging the four steps. Rank using a fixed 0.5 weight on the
old prompt/shared-prior mixture and 0.5 on draft demand. Original mandatory
last-token protection, raw slot cap, balanced layer allocation, original source
positions, and lossless generated history remain. The two candidate codecs are
the registered head3_boundary8 and head3k2v_mse. The no-draft balanced_boundary8
anchor is included. Forecasting is an explicit algorithm change, not a storage
optimization or a faithful published-baseline implementation.

Require unchanged 32-token KL<=0.02, maximum logit difference<=1.0 and token
agreement=1.0 controls; report both FullKV and same-byte anchor differences.
Record 128 greedy tokens and measured prompt/decode-owned storage. These are
not actual benchmark-quality or whole-request memory evidence. The diagnostic
retains its original source cache and therefore cannot support end-to-end or
whole-request peak claims. Construction time includes clone, integrity checks,
and temporary forecast decoding; do not call that overhead free. All common
codec controls must pass. Any task study would require a separate frozen plan
that offers formats identically to competitors and preserves their actual K/V.
