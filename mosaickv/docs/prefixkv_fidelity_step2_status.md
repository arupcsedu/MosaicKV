# PrefixKV fidelity and native-metric comparison: Step 2

## Decision

The large PrefixKV comparison is **not yet admissible**. The corrected OCR
adaptation has complete measurements, but it is not paper-faithful PrefixKV
and it failed its TextVQA development quality gate. Native-route pretrained
GPU differential validation remains required. No large campaign was launched
as part of this update, and no original nominal-ratio campaign was rerun.

## Completed four-worker validation

Immutable source: `968b5fe4d7169b8c5c84f40f69c4933821432d4e`.
Run root: `/scratch/djy8hg/runs/mosaickv/hf-prefixkv-matched-968b5fe-v1`.
Plan SHA-256: `146f451497d10bd1bd4f767cfd89591c7066dd51f19726f02e555c9dd7dab110`.

| Worker | Task/shard | Status | Elapsed |
|---|---|---|---|
| 21369990_0 | TextVQA / 00 | completed, exit 0 | 44:02 |
| 21369990_1 | TextVQA / 01 | completed, exit 0 | 43:56 |
| 21369990_2 | DocVQA / 00 | completed, exit 0 | 43:58 |
| 21369990_3 | DocVQA / 01 | completed, exit 0 | 44:54 |

Each worker requested one A100, 12 CPUs and a two-hour limit. The CPU analyzer
`21370058` exited 1 because the **quality gate failed**, not because inference
crashed or samples disappeared. Its completed report is
`analysis/prefixkv-validation.json` under the run root.

There are 64 examples per task, five methods, and 640 complete rows. An
independent read-only audit verified completion and trace hashes, paired input
controls, effective labels, completeness, and append growth. Its write-once
artifact is `analysis/step2-integrity-audit-20261014.json`.
SHA-256: `7beba2408062cdb72067b5b20478a865eb205aee68d8933d244cb92aadd17b4d`.
The audit implementation is `scripts/audit_prefixkv_matched_validation.py`.

Every compressed row has 41,943,040 initial payload bytes (40 MiB),
74,973,184 final payload bytes (71.5 MiB), 64 generated tokens and zero decode
evictions. The 63 cached decode appends add 524,288 bytes each. These are the
historical payload accounting fields, **not** independent observations of the
new initial/final/peak backing-storage contract. In particular, missing peak
storage must not be inferred or filled in as a measured value.

### Quality, separately by task

Scores below are percentages of the original per-task evaluator scores.
Differences/intervals are percentage points relative to Prompt Top-K, with
10,000 paired bootstrap replicates, seed 20271014. They are development
measurements, not a published-method reproduction.

| Task | Method/variant | Score | Difference [95% CI] | Answer agreement with Top-K |
|---|---|---:|---:|---:|
| TextVQA | FullKV, quality reference | 40.9375 | — | — |
| TextVQA | Prompt Top-K | 40.7813 | reference | — |
| TextVQA | `prefixkv_byte_calibrated_selection_only_reimpl` | 8.7500 | diagnostic adaptation | — |
| TextVQA | `prefixkv_question_scored_selection_only_reimpl` (C-only) | 38.7500 | −2.0313 [−7.6563, +3.2813] | 73.4375% |
| TextVQA | `mosaickv_exact` | 40.7813 | 0 [0, 0] | 98.4375% |
| DocVQA | FullKV, quality reference | 25.4691 | — | — |
| DocVQA | Prompt Top-K | 23.2369 | reference | — |
| DocVQA | `prefixkv_byte_calibrated_selection_only_reimpl` | 3.9063 | diagnostic adaptation | — |
| DocVQA | `prefixkv_question_scored_selection_only_reimpl` (C-only) | 23.5041 | +0.2672 [−3.9063, +4.8126] | 57.8125% |
| DocVQA | `mosaickv_exact` | 22.4557 | −0.7813 [−2.3438, 0] | 95.3125% |

The C-only DocVQA gate passed; TextVQA failed. C-only changes prompt scoring
and suppresses original decode-time eviction. Neither its successes nor the
control's collapse establish original PrefixKV performance. The old analyzer's
DocVQA pass refers to **C-only**, not to strict MosaicKV score parity.

Informational end-to-end ratios to Top-K are 0.5032 (C-only) and 0.9394
(MosaicKV) on TextVQA; 0.4994 and 0.9415 on DocVQA. FullKV is excluded from
compressed-method latency comparisons. These old payload-matched results
must not be relabeled as new peak-storage-validated measurements.

## Preserved official reproduction

The original immutable artifact remains at:
`/scratch/djy8hg/runs/mosaickv/prefixkv/official-profile-r050-2c7a266/official_profile_reproduction_gate.json`.
SHA-256: `bfa508516e9f8884e1cbc495e4af1379b06fca9644a833f242b49d8c2d4c0a35`.

- Pinned official PrefixKV: `597f1ab032704951550f93bcc8a23f1454b80aa4`.
- 1,000 LLaVA-Description examples; 63 completed shards.
- Corpus PPL: **3.4011690877**, compared with the registered published 3.41;
  absolute tolerance 0.25. The numerical PPL gate passed.
- NLL sum: 156550.1590423584; scored tokens: 127888.
- FullKV-reference ROUGE-L: **0.8283484825**, measured. There was **no
  preregistered published numeric ROUGE-L target**, so this is not an additional
  numerical reproduction pass.
- Model: `Zuyan/ElasticCache/llava-v1.5-7b`, revision
  `833edbdc7512240f2a3aa49feeb9468e2297bdbc`; BF16, legacy LLaVA,
  Transformers 4.31.0 / Torch 2.1.2+cu121.

That checkpoint/environment is different from the common HF checkpoint used
by the new comparison. Its PPL pass cannot silently certify cross-checkpoint
token/logit parity. Its upstream calibration identities were not published;
that remains a reproduction limitation.

## Implementation corrections and validation boundary

The older legacy parity callback globally apportioned prefill capacity and
then used its realized integer retention for decode eviction. Upstream uses
independent per-layer rounding and floating offline-profile retention. The
callback now follows the latter, and a synthetic regression executes the
actual pinned upstream class for prefill and 64 decode transitions.

`evaluation/prefixkv_differential.py` now compares selected positions, per-layer
lengths, K/V values, logical-position trajectories, eviction events, generated
tokens and logits, and source-cache isolation. It additionally supports:

- complete target-only teacher forcing, including the first target's prefill
  prediction; no prompt or image token contributes to NLL;
- NLL sums and exact scored-token counts, with absolute tolerance
  `(2 * logit_atol + 1e-6) * target_count`;
- upstream/local ROUGE-L against the same frozen FullKV answer/hash.

The pretrained GPU worker is `scripts/verify_prefixkv_native_fidelity.py`.
It first checks 16-token ordinary-generation/retention-one parity, then checks
64 decode transitions and the complete human target for each materialized
sample. Logit tolerance is fixed at 1e-4, token agreement must be 1.0, and
selected positions/KV construction must agree. It runs actual pinned upstream
selection/eviction on the **same common HF checkpoint and environment** as the
local path. This establishes algorithmic transport fidelity if it passes;
it does not claim to have rerun the historical legacy checkpoint.

The CPU tiny-model tests are correctness fixtures, not pretrained quality
measurements. GPU success has not yet been asserted.

## Native unified PPL/ROUGE-L route

The existing materialized runner now supports `--unified-prefixkv` for
`llava_description` and `mmvet_prefixkv`, including Prompt Top-K. It preserves
raw JSONL plus separate long-form metric observation/aggregate Parquet files.
PPL is `exp(sum NLL / sum target tokens)`, never the mean of sample PPL values.
ROUGE-L and PPL stay separate.

FullKV references are write-once and bind model/tokenizer revisions, generation,
processor/image preprocessing, dataset/ledger, environment lock, and common
block/mandatory policy. Reference mismatches fail before generation. Existing
schema-v1 references without the optional control hash keep their old canonical
hashes but do not satisfy a new unified reference contract.

The native unified PrefixKV variant has original scoring, a deterministic
byte-rescaled offline layer profile, and append-only decode growth. It is
reported as `prefixkv_selection_only_reimpl`; it is **not** the C-only
`prefixkv_question_scored_selection_only_reimpl` adaptation and is not an
unmodified official-method operating point. Faithful native eviction is tested
separately. Suppressing eviction is a disclosed requirement of the common
append-only budget comparison, not hidden as upstream behavior.

The MM-Vet protocol revision is now reconciled with the reviewed
`PrefixKV/MM-Vet`, `official_218` materialized parent:
`8ce15108c0b55c67200eb96601ba2bb98183ff65`. The frozen comparison contract
contains both parent-ledger hashes. No original ledger or result was rewritten.

## Registered small gate and large-run blockers

`scripts/prepare_prefixkv_native_validation.py` prepares (does not submit):

1. ID-only deterministic cohorts of 32 examples per native route from the
   reviewed parent ledgers. These are development subsets, not untouched data.
2. Two-example shards, one method per byte-validation job, common caps
   **40/80/120/160/240 MiB**, and unchanged frozen MosaicKV selection policy.
3. One reusable FullKV reference per shard, not one new reference per cap.
4. Explicitly disabled byte-validation and large-campaign authorization.
   An optional flag enables only the preceding native fidelity stage.

The worker template requests one A100, 12 CPUs, 96 GiB CPU memory, two hours,
`bii-gpu`/`bi_fox_dgx`. Submission must use the existing global controller with
at most four A100s in aggregate. No script here automatically submits jobs.

Before the 32-example byte gate, the pretrained native differential must pass.
Before the 1,000/218-example comparison, every retained cap must be physically
feasible with equal measured initial/final/peak backing bytes, independent
largest-feasible-allocation certificates, complete paired metrics/provenance,
and a frozen-reference hash. Unsupported caps must be labeled, never silently
underfilled or retried at another ratio. Compatible completed work is reused;
only missing/failed shards of the same immutable plan may be retried.

The current empirical blockers are therefore **pretrained native fidelity**,
the **32-example physically matched native-metric gate**, and then the large
comparison. Successful OCR adaptation alone does not discharge any of them.

## Verification performed for this update

The common environment passed 211 targeted unit/integration tests spanning
the comparison contract, native preparation, PrefixKV parity and decode
policies, official metric fixtures, and tiny HF runtime paths. The final
fingerprint/configuration subset passed again (27 tests). Ruff passed for all
Step 2 changed Python files, MyPy passed for the five modified evaluation/CLI
modules, and `bash -n` passed for the worker template. These are CPU tests;
CUDA unavailability on the test host was reported, not hidden.

No commit or new GPU submission was made by this update. Canonical submission
needs a scoped immutable source commit; unrelated paper, requirements, vLLM
and scheduler changes must remain untouched. The comparison rule content
fingerprint was refreshed for the evaluation/fidelity corrections, while the
MosaicKV **selection-policy fingerprint stayed unchanged**.
