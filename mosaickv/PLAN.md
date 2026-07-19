# MosaicKV Milestone Plan

## Purpose and status vocabulary

This plan sequences the work needed to turn the MosaicKV design into a defensible research system and AAAI 2027 artifact. It does not claim that any algorithm, integration, experiment, or result has been implemented. All milestones below are **planned** as of the creation of this document.

Milestone status must be one of:

- `planned`: scope is defined but implementation has not begun;
- `in_progress`: work exists but at least one exit criterion is unmet;
- `blocked`: progress requires a documented dependency or decision; or
- `complete`: every exit criterion has inspectable evidence at a pinned git SHA.

## Milestone overview

| Phase | Milestone | Initial status | Depends on |
|---|---|---|---|
| A | Hugging Face full-cache reference | planned | governance documents |
| B | MosaicKV core | planned | A |
| C | correctness tests | planned | A, B |
| D | simple baselines | planned | A, C |
| E | published baselines | planned | A, C, D |
| F | quality evaluation | planned | C, D, E |
| G | systems evaluation | planned | C, F |
| H | vLLM integration | planned | C, G |
| I | SGLang integration | planned | C, G |
| J | artifact packaging | planned | A-I |

Statuses are governance metadata, not empirical results. Update them only with links to tests, manifests, or artifacts that support the change.

## A. Hugging Face full-cache reference

**Goal:** establish a minimal, auditable full-cache reference path against which every compressed and backend-specific path is compared.

**Deliverables**

- Pinned model and processor revisions for the first supported multimodal model.
- Canonical prompt, image/video preprocessing, tokenization, and generation configuration.
- Full-cache prefill and decode runner using Hugging Face Transformers.
- Per-example trace and run-manifest emission conforming to `REPRODUCIBILITY.md`.
- Determinism characterization and reference fixtures small enough for continuous testing.

**Exit criteria**

- The same resolved configuration can be rerun from a clean checkout.
- Repeated deterministic runs satisfy the documented token and numerical tolerances.
- Cache shapes, token positions, modality boundaries, output lengths, and memory accounting are tested and documented.
- No compression behavior is present in the reference path.

## B. MosaicKV core

**Goal:** implement the backend-independent MosaicKV algorithm without fusing it prematurely into an inference engine.

**Deliverables**

- Future-query forecasting with explicit inputs, horizon, and uncertainty outputs.
- Sparse cross-modal evidence graph construction with documented nodes, edges, sparsification, and modality mapping.
- Value-aware block utility with stable block definitions and inspectable utility terms.
- Budgeted submodular selection with deterministic tie-breaking and a hard budget check.
- Exact, prototype, and residual cache tiers with end-to-end byte and token accounting.
- Uncertainty-guided residual repair with logged trigger decisions and budget impact.
- Versioned configurations for the complete method and each named ablation.

**Exit criteria**

- Every one of the six MosaicKV mechanisms can be unit tested and ablated independently.
- Selected cache state never exceeds the configured budget under the documented accounting convention.
- Retention ratio `1.0` takes a non-lossy path suitable for the Phase C equivalence gate.
- The core API does not depend on benchmark labels or any training procedure.

## C. Correctness tests

**Goal:** establish correctness before performance optimization or broad evaluation.

**Deliverables**

- Unit tests for forecasting, graph construction, utility, selection, tier conversion, and repair.
- Property tests for budget feasibility, deterministic tie-breaking, indexing, modality boundaries, and sequence positions.
- Full-cache equivalence tests at retention ratio `1.0` for every supported precision, backend, and attention implementation.
- Adversarial fixtures covering empty modalities, long video sequences, small budgets, all-exact allocation, and repair saturation.
- Comparison checks that reject mismatched prompts, media, tokenization, generation, output length, budget, precision, or backend.

**Exit criteria**

- All required tests pass at a pinned clean git SHA.
- The `1.0` equivalence contract in `REPRODUCIBILITY.md` passes for each supported configuration.
- Known nondeterminism and unsupported cases are documented rather than suppressed.
- Failures produce actionable diagnostics and retain their run manifests.

## D. Simple baselines

**Goal:** provide transparent, locally implemented reference policies before adding published systems.

**Candidate baselines**

- Full cache.
- Uniform or fixed-stride retention.
- Recency-only retention.
- Random retention with recorded seeds.
- Attention-score or magnitude-based retention where model access permits.

**Exit criteria**

- All baselines share the same cache-budget accounting and controlled inputs.
- Baseline-specific behavior is covered by tests and versioned configuration.
- Random methods report all preregistered seeds, not only favorable runs.
- Result generation enforces complete row provenance.

## E. Published baselines

**Goal:** evaluate relevant published KV-cache compression approaches with verifiable provenance and fair adaptations to multimodal workloads.

**Deliverables**

- A baseline registry recording paper, repository URL, immutable commit SHA, license, citation, supported model/backend, and local patches.
- Official external code only under `third_party/`, with licenses and attribution preserved.
- Local paper-faithful implementations named `*_reimpl` and consistently described as reimplementations, never as official code.
- Compatibility notes identifying any unavoidable departures from the controlled-comparison contract.

**Exit criteria**

- Each baseline is either directly comparable under every required control or reported in a separate non-comparable section.
- Upstream commit SHAs and patches reproduce the evaluated executable state.
- Smoke tests and full-cache or no-compression sanity checks pass where the method supports them.
- No baseline result is reported without complete provenance.

## F. Quality evaluation

**Goal:** measure task quality and generation fidelity at matched cache budgets.

**Deliverables**

- Pinned benchmark and dataset revisions with documented licenses and preprocessing.
- Preregistered task metrics, retention ratios, seeds, aggregation rules, exclusions, and stopping criteria.
- Paired per-example outputs for full cache, MosaicKV, ablations, and comparable baselines.
- Confidence intervals or paired uncertainty estimates where statistically appropriate.
- Separate ablations for all six MosaicKV mechanisms.

**Exit criteria**

- Every reported row resolves to immutable run manifests and raw per-example outputs.
- Comparison-invariant validation passes before aggregation.
- Measured tables contain no synthetic, placeholder, interpolated, or estimated values.
- Failed runs, exclusions, and missing data are disclosed with reasons.

## G. Systems evaluation

**Goal:** characterize latency, throughput, memory, and cache behavior without conflating systems changes with quality changes.

**Deliverables**

- Preregistered warmup, synchronization, repetition, concurrency, batch, prompt-length, generation-length, and measurement procedures.
- Prefill latency, time to first token, decode latency or inter-token latency, end-to-end latency, throughput, peak device memory, and realized cache size as applicable.
- Breakdown of forecasting, graph, selection, tier construction, and repair overhead.
- Paired quality checks for every systems configuration used in claims.

**Exit criteria**

- Runs record hardware, software, clocks or power policy when controlled, backend, and attention implementation.
- Timings exclude or include preprocessing consistently and state the boundary.
- Sufficient repetitions and uncertainty summaries are reported.
- Cache-budget accounting includes tier payloads and required metadata.

## H. vLLM integration

**Goal:** integrate MosaicKV into a pinned vLLM revision while preserving the validated core semantics.

**Deliverables**

- A thin, documented adapter to the selected vLLM cache and scheduler interfaces.
- Pinned vLLM version or commit and all local integration patches.
- Backend-specific correctness, full-cache equivalence, continuous-batching, and memory-accounting tests.
- Quality and systems parity checks against the core reference on a shared test slice.

**Exit criteria**

- Retention ratio `1.0` satisfies the registered vLLM tolerance.
- Multi-request cache ownership, sequence lifecycle, and scheduling are tested.
- No backend optimization silently changes prompts, outputs, budgets, precision, or generation settings.
- Supported and unsupported vLLM modes are explicit.

## I. SGLang integration

**Goal:** integrate MosaicKV into a pinned SGLang revision while preserving the validated core semantics.

**Deliverables**

- A thin, documented adapter to the selected SGLang cache and runtime interfaces.
- Pinned SGLang version or commit and all local integration patches.
- Backend-specific correctness, full-cache equivalence, scheduling, and memory-accounting tests.
- Quality and systems parity checks against the core reference on a shared test slice.

**Exit criteria**

- Retention ratio `1.0` satisfies the registered SGLang tolerance.
- Request lifecycle, cache reuse or radix-cache interactions, and concurrency are tested where enabled.
- No backend optimization silently changes controlled comparison variables.
- Supported and unsupported SGLang modes are explicit.

## J. Artifact packaging

**Goal:** produce a reviewable, redistributable artifact that regenerates claims from pinned inputs without hidden state.

**Deliverables**

- Environment lockfiles or images with documented build instructions and immutable base identifiers.
- Data/model acquisition scripts that verify revisions and checksums without redistributing restricted assets.
- One-command smoke tests and documented commands for correctness, quality, systems, and table generation.
- Immutable run manifests, schemas, raw-output index, and scripted aggregation pipeline.
- Third-party license and attribution bundle, paper-to-artifact claim map, and artifact checklist.
- Resource and expected-runtime estimates for smoke, reduced, and full evaluations.

**Exit criteria**

- A clean-machine rehearsal passes using only documented inputs and credentials.
- Every paper table and figure maps to scripts and measured run IDs.
- The artifact exposes no secret, private data, or unlicensed material.
- Release tags, git SHA, config SHAs, dependency revisions, and archival identifiers agree.

## Cross-milestone gates

The following gates apply throughout the plan:

1. **Provenance gate:** no experimental row without the complete manifest fields required by `AGENTS.md`.
2. **Equivalence gate:** no compressed-path claim until retention ratio `1.0` reproduces full cache within the documented tolerance.
3. **Fairness gate:** no direct comparison when any required controlled input differs.
4. **Integrity gate:** no synthetic number in a measured-results table.
5. **Licensing gate:** no external code outside `third_party/`, without an immutable SHA, or without preserved license and attribution.
