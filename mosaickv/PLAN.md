# MosaicKV Milestone Plan

## Purpose and status vocabulary

This plan sequences the work needed to turn the MosaicKV design into a defensible research system and AAAI 2027 artifact. The Hugging Face reference, backend-independent mechanisms, and eager exact-selection runtime are in progress; they do not establish real-checkpoint support or paper results. No milestone status implies an experimental result.

Milestone status must be one of:

- `planned`: scope is defined but implementation has not begun;
- `in_progress`: work exists but at least one exit criterion is unmet;
- `blocked`: progress requires a documented dependency or decision; or
- `complete`: every exit criterion has inspectable evidence at a pinned git SHA.

## Milestone overview

| Phase | Milestone | Current status | Depends on |
|---|---|---|---|
| A | Hugging Face full-cache reference | in_progress | governance documents |
| B | MosaicKV core | in_progress | A |
| C | correctness tests | in_progress | A, B |
| D | simple baselines | in_progress | A, C |
| E | published baselines | in_progress | A, C, D |
| F | quality evaluation | planned | C, D, E |
| G | systems evaluation | planned | C, F |
| H | vLLM integration | blocked | C, G |
| I | SGLang integration | blocked | C, G |
| J | artifact packaging | planned | A-I |

Statuses are governance metadata, not empirical results. Update them only with links to tests, manifests, or artifacts that support the change.

## A. Hugging Face full-cache reference

**Goal:** establish a minimal, auditable full-cache reference path against which every compressed and backend-specific path is compared.

**Current evidence:** eager explicit prefill/token-decode adapters, typed cache
extraction/reinjection, q-projection capture, the transformation-free FullKV
runner, synchronized CUDA phase measurement, raw per-trial storage, and
no-download architecture tests are implemented in
[the adapter package](src/mosaickv/adapters/huggingface),
[the FullKV implementation](src/mosaickv/fullkv.py), and
[the measurement package](src/mosaickv/measurements). The protocol is documented
in [the adapter guide](docs/huggingface_adapters.md) and
[FullKV reference guide](docs/fullkv_reference.md). This phase remains
`in_progress`. Pinned Qwen2.5-VL-3B and LLaVA-1.5-7B one-image integration
and unified 16-token retention-1 numerical parity gates passed on an A100.
The development records remain non-canonical, and the other registered
checkpoint/precision combinations have not passed. See
[the unified runtime validation record](docs/huggingface_runtime.md).

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

**Current evidence:** typed full/exact/prototype/residual state containers,
fixed-size per-layer/per-KV-head blockization, logical-position tracking,
storage accounting, membership invariants, lossless 100%-retention
reinjection, prompt-window/draft/hybrid future-query forecasting, sparse
cross-modal evidence graph construction, value-aware block utility, and
mandatory-first budgeted selection, conservative RoPE-gated prototype
construction, indexed CPU residual storage, and a backend-independent
uncertainty-guided repair controller are implemented in
[the cache-state module](src/mosaickv/cache_state.py) and documented in
[the cache-state guide](docs/cache_state.md), with forecasting in
[the forecasting package](src/mosaickv/forecasting) and
[forecasting guide](docs/future_query_forecasting.md), with graph construction in
[the graph package](src/mosaickv/graph) and
[evidence-graph guide](docs/evidence_graph.md), with utility and selection in
[the selection package](src/mosaickv/selection) and
[utility/selection guide](docs/block_utility_and_selection.md), with tier
construction in [the prototype and residual packages](src/mosaickv/prototypes)
and [three-tier guide](docs/three_tier_cache.md), with repair in
[the repair package](src/mosaickv/repair) and
[decode-time repair guide](docs/decode_time_repair.md). This phase remains
`in_progress`: the unified eager runtime performs compact exact-selection
decoding while preserving original logical positions. Current post-RoPE
adapters still fail closed from prototype merging to exact selection, so
real-model prototype and repair support remains unavailable until adapter
mutation and parity gates pass. See
[the unified HF runtime guide](docs/huggingface_runtime.md).

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

**Current evidence:** CPU planner tests cover all method labels, budgets,
retention-1 reconstruction, safety fallbacks, and monotonic active storage.
No-download tests use real randomly initialized LLaVA-1.5, Qwen2.5-VL, and
LLaVA-OneVision Transformers architectures to exercise compact eager decoding,
trace completeness, and FullKV token parity. Pinned Qwen2.5-VL-3B and
LLaVA-1.5-7B single-example gates plus a pinned 20-example MMStar development
run validate the integrated execution, scoring, and artifact paths. This phase
also has passing unified retention-1 numerical-tolerance records for the pinned
Qwen2.5-VL-3B and LLaVA-1.5-7B eager configurations. It remains `in_progress`
until every configuration claimed as supported has a canonical registered
fixture; OneVision/InternVL checkpoints and non-eager attention are not covered.

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

**Implemented scope**

- `full_kv`: transformation-free full cache.
- `random_kv`: seeded random exact-block retention.
- `uniform_kv`: fair retained-cost allocation by layer, KV head, and modality.
- `prompt_attention_topk`: prompt-window eager-attention block ranking.
- `value_topk`: within-layer/head value-novelty block ranking.

All compressed policies retain exact blocks only and share FullKV blockization, mandatory-token
handling, cache packing, explicit decoding, timing, and byte accounting. Their contract and
configuration are documented in [the simple baseline guide](docs/simple_baselines.md), with unit
coverage in [the baseline tests](tests/unit/test_baselines.py) and no-download runtime coverage in
[the tiny HF integration tests](tests/integration/test_hf_tiny_models.py). This phase remains
`in_progress` pending a pinned clean-SHA real-checkpoint parity and budget sweep.

**Exit criteria**

- All baselines share the same cache-budget accounting and controlled inputs.
- Baseline-specific behavior is covered by tests and versioned configuration.
- Random methods report all preregistered seeds, not only favorable runs.
- Result generation enforces complete row provenance.

## E. Published baselines

**Goal:** evaluate relevant published KV-cache compression approaches with verifiable provenance and fair adaptations to multimodal workloads.

**Current evidence:** official LOOK-M is pinned as an unmodified submodule at
`ecf0f51a9c416c2d85e47faf2638502f01a6d748` with its MIT license. The
paper-equation `lookm_reimpl` implements text-prior scoring, recent/top-N
selection, cosine pivot assignment, and averaged/pivotal/weighted KV merging
through the shared eager HF cache and metrics runtime. The exact specification,
official-source differences, model assumptions, and labeling rules are in
[the LOOK-M specification](docs/baselines/lookm_spec.md). A strict artifact
comparator rejects unequal checkpoints, samples, tokenization, generation,
precision, and backend controls. The current
[parity report](docs/baselines/lookm_parity_report.md) contains no numerical
row because the official original-LLaVA checkpoint/runtime and the cached HF
conversion do not satisfy those controls. A standalone LLaVA-HF synthetic
`lookm_reimpl` smoke passed on the reserved A100 path with complete artifacts,
but its dirty-source manifest correctly makes it non-canonical and it is not an
official parity result. Official PrefixKV is also pinned, unmodified, at
`597f1ab032704951550f93bcc8a23f1454b80aa4` with its MIT license. The shared
runtime now contains `prefixkv_reimpl`: eager prompt-attention importance,
adaptive offline profiles, exact global per-layer budgets, protected boundary
tokens, fixed-distance decode eviction, strict calibration/evaluation
separation, LLaVA/generalized labeling, and a controlled parity artifact
comparator. See the [PrefixKV specification](docs/baselines/prefixkv_spec.md)
and [execution status](docs/baselines/prefixkv_parity_report.md). The official
legacy-checkpoint run now satisfies the identical model, tokenizer, prompt,
media, profile, generation, budget, precision, backend, hardware, and seed
controls. Its strict comparator reports 100% agreement for 16 generated tokens
and documents upstream's one-position global-budget undershoot. The run is a
dirty-worktree, single-trial development diagnostic rather than a paper result.
The ICLR `vl_cache_reimpl` now applies relative-threshold post-vision sparsity,
prompt-adaptive layer allocation, and accumulated post-vision Top-K through the
same exact-cache packer. Its [specification](docs/baselines/vl_cache_spec.md)
maps equations to code, isolates rounding/GQA/recency decisions, enforces
calibration/evaluation ID disjointness, and records a structural sensitivity
grid. Formula, budget, determinism, leakage, and tensor-level retention-one
tests pass; paper-model task and Triton latency trends remain unmeasured.
LOOK-M official parity, clean repeated PrefixKV measurements, and VL-Cache
paper-model trend runs keep this phase `in_progress`.

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

**Current evidence:** the pinned vLLM 0.11.2 Stage A FullKV wrapper streams
token outputs and records per-trial TTFT, ITL, throughput, latency, GPU process
memory, prefix-cache hits, and multimodal preprocessor-cache behavior through
the common evaluation/manifest path. Model registration covers Qwen2.5-VL,
LLaVA-1.5, and LLaVA-OneVision. Stage B is blocked: the installed scheduler and
runner expose no atomic sparse-logical-block commit that preserves original
positions. `--enable-mosaickv` therefore fails before loading weights and emits
no simulated result. The exact source boundary and proposed upstream API are
recorded in [the native blocker](docs/vllm_native_blocker.md). The Stage A GPU
acceptance command remains required before runtime support is claimed.

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

**Current evidence:** a version-pinned SGLang 0.5.10.post1 Stage A wrapper
launches Qwen2.5-VL through the common evaluation path with deterministic
Triton attention, one-token streaming, no overlap schedule, no CUDA graph, and
no server warmup. Raw trials preserve TTFT, token intervals, throughput,
process-tree GPU memory, Radix cached-token counters, Prometheus cache gauges,
exact server arguments, and logical active-KV byte accounting. The isolated
environment and both Qwen2.5-VL-3B/7B Stage A checkpoint GPU gates passed with
deterministic repeated tokens, exact byte accounting, Radix/GPU telemetry, and
an A-B-A request-isolation probe. Controlled HF eager token parity did not
pass, so optimized SGLang settings remain disabled. Stage B is blocked because
the installed public and internal APIs do
not offer an atomic request-scoped commit for KV allocator ownership,
`ReqToTokenPool`, Radix nodes, logical positions, and Qwen mRoPE positions.
`--enable-mosaickv` fails before server launch and never emits simulated native
rows. See [the SGLang native blocker](docs/sglang_native_blocker.md).

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
