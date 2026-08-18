# MosaicKV implementation and artifact readiness after Choice C v86

Date: 2026-08-17

Reviewed source commit: `2004c1e71e7d8f0a4c5a7f123172f856bc2a4e0a`

Calibration run: `/scratch/djy8hg/runs/mosaickv/choice-c-v86-disjoint-robust-fidelity-calibration-v1`

## Decision

The repository is **engineering-ready for continued controlled development of
the Hugging Face exact-selection path**, but it is **not scientifically ready
for a paper claim that the frozen MosaicKV method meets the quality--memory
gate**. V86 is a valid negative label-blind calibration result. It produced no
frozen candidate, and held-out evaluation remains blocked.

This distinction is essential:

- implementation and CPU artifact checks pass;
- the v86 measurements pass the memory gate;
- the v86 measurements fail the preregistered aggregate and fold-robust
  fidelity gates; and
- v86 exercised `mosaickv_exact`, not prototype merging, residual storage, or
  repair.

No v87, replacement calibration, or held-out campaign was launched during this
review.

## V86 execution integrity

The two one-A100 workers completed in at most 1 hour, 23 minutes, and 16
seconds. The run contains 1,280 measured rows: 128 examples for each of five
arms on DocVQA and TextVQA. All rows completed, no row contains an error or a
missing required measurement, and calibration telemetry contains neither
benchmark references nor official task scores.

The source was clean at commit `2004c1e`, model and dataset revisions were
pinned, the common environment lock was recorded, and manifests contain the
required model, precision, backend, attention implementation, CUDA, driver,
PyTorch, Transformers, vLLM, SGLang, GPU, seed, and measurement-type fields.

The v86 ledgers contain 128 unique samples per task and have zero overlap with
the v85 held-out ledgers. The DocVQA ledger records 1,140 prior unique samples
excluded; the TextVQA ledger records 568.

## Scientific gate outcome

Every candidate used exact-only replacement with four-token blocks, a
28,500,000-byte configured cap, no prototypes, no residual storage, and no
repair.

| Task | Prompt Top-K active bytes | Candidate active bytes | Total reduction | Aggregate fidelity gate |
|---|---:|---:|---:|:---:|
| DocVQA | 40,693,760 | 27,488,256 | 32.45% | Fail |
| TextVQA | 40,546,304 | 27,701,248 | 31.68% | Pass |

The conservative risk-gated arm was closest on aggregate DocVQA fidelity, but
its FullKV answer-agreement difference from Prompt Top-K was -5.47 percentage
points against the preregistered -5-point limit. No threshold was changed after
observing this result.

The robust fold gate also failed. The best two risk-gated arms each passed 26
of 32 task/fold performance checks, but all 32 were mandatory. Every candidate
had a worst answer-fold difference of -12.5 points. Worst token-fold
differences ranged from -18.75 to -20.02 points.

Job `18641059` therefore exited with code 2 by design, indicating that no
eligible candidate could be frozen. This must not be described as a runtime
crash or as a successful candidate freeze.

## Implementation readiness

### Ready within the validated scope

- The editable package and `mosaickv` CLI load from the common Python 3.11.15
  environment.
- `mosaickv doctor` runs without model downloads and correctly reports a
  CPU-only login host while detecting installed HF, vLLM, and SGLang packages.
- The complete CPU/unit and non-GPU integration suite passes: 586 tests.
- Ruff passes across `src`, `tests`, and `scripts`.
- Strict MyPy passes after making the NumPy scalar boundary in incremental
  objective evaluation explicitly return a Python `float`.
- `pip check` reports a consistent installed environment.
- The v86 HF eager-prefill/SDPA-decode run is canonical, deterministic, paired,
  and fully manifested.
- Exact active-byte and residual-byte accounting is present for every row.

### Implemented but not validated by v86

The repository contains forecasting, evidence-graph, value-aware utility,
budgeted selection, prototype, residual, and repair modules. V86 validates only
the exact-selection configuration with hybrid forecasting and graph/utility
selection. Because prototypes, residuals, and repair were disabled, v86 cannot
support a `mosaickv_proto` or `mosaickv_full` quality claim.

HF model adapters and earlier bounded GPU evidence exist for additional model
and tier paths, but they are not interchangeable with this clean v86
calibration. A paper table must cite the exact run and effective method rather
than inferring support from code presence.

### Explicitly unsupported for paper rows

- Native vLLM and SGLang whole-block transports have version-pinned
  experimental evidence, but the common MosaicKV selector-plan bridge is not
  paper-authorized.
- vLLM/SGLang prototype injection and residual repair remain unsupported.
- The v86 result is LLaVA-1.5-only and must not be generalized to Qwen or
  OneVision.
- No v86 candidate passed the freeze gate, so no new held-out campaign is
  authorized.

## Artifact preservation

The repository seal is
`configs/ledgers/choice_c_v86_negative_calibration_seal.yaml`. It records hashes
for the protocol, parent v85 seal, input ledgers, telemetry, analyses, rankings,
fold table, and representative manifests.

It also seals the complete run tree using a deterministic inventory:

- files: 1,394;
- total size: 627,766,894 bytes;
- inventory SHA-256:
  `194331c8d650c61c29750d649933a22b7ad80b9822d02b657c11172ad0b8426c`.

The inventory hashes sorted records of the form
`<file-sha256><two spaces><relative POSIX path><newline>`. The validator
`scripts/validate_choice_c_v86_negative_seal.py` recomputes the complete
inventory, checks selected artifact hashes, verifies all 1,280 rows, confirms
the absence of label-bearing fields and a frozen candidate, and rechecks zero
overlap with v85.

The run tree remains under `/scratch`; it has not been copied into Git or a
public artifact archive. External artifact packaging will require a separate,
explicitly authorized archival operation that preserves this inventory hash.

## Environment and operational observations

The common environment contains PyTorch 2.8.0, Transformers 4.55.2, vLLM
0.10.2, SGLang 0.5.1.post3, FlashAttention 2.8.3, lmms-eval 0.7.2, and CUDA
12.8-built PyTorch. The A100 manifests record driver 595.71.05. All caches used
the `/scratch/djy8hg/cache/mosaickv` root.

The CPU doctor correctly cannot access a GPU or NVIDIA Management Library
(NVML) on the login host. Test output also warns that `pynvml` is deprecated.
These are environment warnings, not v86 measurement failures. GPU logs warn
that the cluster's 4.18 kernel is older than Accelerate's recommendation; both
workers nevertheless completed, but the warning should remain disclosed.

## Readiness checklist

| Area | Status | Evidence or blocker |
|---|:---:|---|
| Clean canonical v86 execution | Pass | Commit `2004c1e`; both GPU workers completed |
| Complete provenance | Pass | Per-arm manifests and pinned revisions |
| Dataset disjointness | Pass | Zero overlap with v85 for both tasks |
| Row completeness | Pass | 1,280/1,280 completed; no errors or missing fields |
| Active and total memory gate | Pass | 32.45% DocVQA; 31.68% TextVQA |
| Aggregate fidelity gate | Fail | Every arm failed DocVQA |
| Four-fold robust fidelity | Fail | No arm passed every answer and token fold |
| Frozen v86 candidate | Blocked | Intentionally absent |
| Held-out evaluation | Blocked | No frozen candidate |
| CPU/unit/integration tests | Pass | 586 tests |
| Ruff | Pass | `src`, `tests`, and `scripts` |
| Strict MyPy | Pass | 95 source files after scalar-boundary fix |
| Installed dependency consistency | Pass | `pip check` |
| Full MosaicKV tier claim from v86 | Unsupported | Exact-only effective method |
| Native vLLM/SGLang paper rows | Unsupported | Selector-plan bridge not verified |
| Complete run-tree preservation | Pass | Sealed SHA-256 inventory |

## Required disposition

V86 must remain a negative calibration result. Do not create a frozen candidate
manually, reinterpret the exit code as success, weaken the gates, or submit a
held-out run. Within the present 24-hour completion window, the appropriate
endpoint is to commit the seal, validator, readiness report, test, and typing
fix only after explicit user authorization. Any future method design or GPU
campaign requires a new version and separate authorization.
