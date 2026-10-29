# MosaicKV

**Paper name: EviCastKV.** EviCastKV and MosaicKV are the same system, not
separate methods. Historical `mosaickv_*` identifiers and immutable artifacts
are retained for reproducibility; experimental variants remain distinguished.
See the [CPU mechanism audit](docs/evicastkv_cpu_mechanism_audit_20261029.md)
for the latest routing, PrefixKV-adaptation, and compression-overhead findings.

MosaicKV is a training-free multimodal KV-cache compression research system for an AAAI 2027 paper. It aims to preserve task-relevant multimodal context under a fixed cache budget by combining:

1. future-query forecasting;
2. sparse cross-modal evidence graph construction;
3. value-aware block utility;
4. budgeted submodular selection;
5. exact, prototype, and residual cache tiers; and
6. uncertainty-guided residual repair.

## Project status

The repository contains installable research infrastructure and a unified
Hugging Face runtime: strict configuration, provenance manifests, structured
logging, model inspection, environment diagnostics, an lmms-eval adapter,
append-only evaluation storage, explicit prefill/token decode, FullKV, simple
exact-cache baselines, and local `lookm_reimpl`, `prefixkv_reimpl`, and
`vl_cache_reimpl` methods. It also contains forecasting, sparse graph
construction, block utility and selection, conservative cache tiers, and
decode-time repair. Qwen2.5-VL now has a version-pinned, audited post-M-RoPE
prototype transform: cached keys are de-rotated with exact model phases,
merged in FP32 phase-neutral space, and re-rotated at the prototype anchor.
Prototype and residual-repair execution remains eager-only and fails closed
outside Transformers 4.55.2. Other post-RoPE adapters still fall back to exact
selection.

The common lock, package imports, CUDA/FlashAttention smoke, static checks,
unit tests, synthetic 100%-retention equivalence, and pinned Qwen2.5-VL-3B HF
eager/SDPA/FlashAttention parity gates pass in exploratory dirty-tree runs.
Qwen2.5-VL-3B FullKV cross-parity now passes for the current-lock vLLM and
SGLang Torch-native wrappers. Dataset progression remains incomplete.
Official LOOK-M and PrefixKV source is pinned and remains unmodified. Their
fail-closed control-plane launcher verifies exact isolated author environments;
only explicitly labeled reimplementations execute inside the common runtime.
This repository currently reports **no paper-eligible measured experimental
results**.

The bounded dirty-tree Experiment-3 validation completed 84/84 rows on A100
GPUs with identical per-sample active byte budgets. It provides implementation
evidence for Qwen2.5-VL forecasting, spatial/attention graph inputs, safe
three-tier construction, and evaluation-only oracle repair, but its one-/two-
example task subsets are not paper evidence. See the
[component-ablation analysis](analysis/component_ablations.md).

The bounded dirty-tree Experiment-4 validation completed 105/105 paired rows
and 686 decode-step records over seven MMStar, DocVQA, ChartQA, and Video-MME
examples. It audits entropy, prototype-risk, combined, and evaluation-only
oracle repair plus threshold, restoration-count, residual-dtype, and placement
sensitivity. One failure was oracle-repairable; entropy and oracle repair each
improved its benchmark score, while the conservative combined threshold also
damaged a DocVQA answer and therefore had zero net score gain. The primary
combined policy changed the failing ChartQA answer to a different wrong number
without improving its score. The measured online overhead is not justified by
this tiny sample. See the
[repair analysis](analysis/residual_repair.md) and
[qualitative appendix](analysis/residual_repair_qualitative.md).

The subsequent 44-cell quality-fix validation corrects partial-prototype
restoration, ranks residual members by weighted reconstruction error, and gates
prototype risk on minimum token uncertainty. On the same seven examples, the
guarded risk policy repaired the sole compressed benchmark failure (score sum
4/7 to 5/7) with no harmful answer changes and 3.983 seconds paired mean
end-to-end overhead. FP8/INT8 retained this quality gain and halved residual
storage to 26.80 MiB, but did not reduce latency; re-decode and conversion
dominate transfers of only about 13 KiB per event. `R=1`, `R=2`, and `R=4` all
repaired the failure, so larger restoration counts are not justified by this
sample. These measurements are still exploratory dirty-tree evidence. See the
[quality-fix analysis](analysis/residual_repair_quality_fix.md).

The bounded Experiment-5 study calibrated on two disjoint Qwen2.5-VL-3B
DocVQA examples, froze `hybrid4`, and completed 54 held-out rows across
Qwen2.5-VL-3B, Qwen2.5-VL-7B, and LLaVA-1.5-7B. Eight of nine uniform
exact-selection model/task cells preserved the tiny-slice FullKV score while
reducing active KV memory by 80.3%--81.6%; the Qwen3B ChartQA cell did not.
All six direct HF eager/SDPA/FlashAttention-2 token-parity probes passed.
The supplemental same-prefix FullKV probe also records full-vocabulary maximum
and mean absolute logit deviation plus KL for SDPA and FlashAttention-2.
Prototype/repair transfer was mixed and remains separated from the
architecture-neutral result. The run is exploratory dirty-tree evidence, not
a paper claim. See the
[generalization analysis](analysis/hyperparameter_generalization.md) and
[architecture report](analysis/architecture_specific_adaptations.md).

The common environment imports the installed vLLM and SGLang stacks and passes
the CUDA smoke. SGLang Stage A passes repeated determinism, active-byte
accounting, request isolation, and 32/32 HF-SDPA token parity. vLLM loads and
serves the pinned checkpoint, and its uncached Stage-A verifier plus 16/16 HF
token parity pass. Prefix-cache cold/warm token divergence remains fail-closed.
Version-pinned native whole-block transactions now pass on A100 for vLLM
0.10.2 (job `17869843`) and SGLang 0.5.1.post3 (job `17871006`): both reproduce
ordinary FullKV at retention 1.0, release 9 of 18 prompt blocks at retention
0.5, preserve logical decode positions, and pass A-B-A isolation. These are
transport-validation runs using a deterministic fallback selector. They do not
authorize compressed paper rows; the experiment matrix remains fail-closed
until the common MosaicKV selector-plan bridge passes. No simulated native row
is produced, and prototype injection/repair remain unsupported in serving
backends.

## Development quick start

Use the single locked Python 3.11 environment for HF, vLLM, SGLang,
evaluation, and development. Setup refuses a dirty worktree and all caches are
placed under `/scratch/djy8hg/cache/mosaickv`:

```bash
source mosaickv/scripts/cache_env.sh
mosaickv/scripts/assert_clean_worktree.sh
mosaickv/scripts/create_envs.sh --sync common
/scratch/djy8hg/env/mosaickv/bin/mosaickv doctor
/scratch/djy8hg/env/mosaickv/bin/mosaickv inspect-model llava-hf/llava-1.5-7b-hf
/scratch/djy8hg/env/mosaickv/bin/mosaickv smoke
```

The common lock uses Torch 2.8.0/CUDA 12.8, Transformers 4.55.2, vLLM 0.10.2,
SGLang 0.5.1.post3, and FlashAttention 2.8.3. Exploratory A100 job `17347802`
verified all 251 pins, required imports, CUDA matrix multiplication, and the
specialized FlashAttention forward/rejection matrix. Model/backend parity and
native Docker remain separate gates. The worktree was dirty, so none of this
evidence is paper-eligible until repeated from a clean commit.

Run `evaluate --config mosaickv/configs/smoke.toml` for the CPU preflight, or use `configs/hf_mosaickv.yaml` and a task to execute the unified HF runtime. Model caches must be outside the home directory.

See [development instructions](docs/development.md) and [GPU diagnostics](slurm/README.md) for local checks and the non-downloading Slurm doctor job.
The common lock, cache policy, local/Slurm setup, and Docker commands are
documented in [environment setup](env/README.md). Creating or synchronizing the
environment is always an explicit clean-tree action.
The current cluster's native-Docker blocker and the commands required on a
Docker-capable host are recorded in
[Docker verification status](docs/docker_verification.md).
The [evaluation harness](docs/evaluation_harness.md) documents deterministic development
subsets, the local-model protocol, lmms-eval scoring ownership, result fields, and resume/merge
semantics.

## Research contract

- All direct comparisons use identical prompts, media, tokenization, generation settings, output lengths, cache budgets, precision, and backend configuration.
- At retention ratio `1.0`, MosaicKV must reproduce the full-cache reference within the preregistered numerical tolerance.
- Synthetic or placeholder numbers are prohibited from measured-results tables.
- Every experimental row must resolve to complete source, configuration, model, dataset, environment, hardware, backend, seed, and measurement provenance.
- External baseline code is pinned under `third_party/` with its license and attribution. Local paper-faithful implementations are named `*_reimpl` and are never represented as official code.

## Documentation

- [Agent rules](../AGENTS.md) — mandatory instructions for all future repository work.
- [Milestone plan](PLAN.md) — phases A-J, dependencies, deliverables, and exit criteria.
- [Scientific integrity](SCIENTIFIC_INTEGRITY.md) — evidence, comparison, results, baseline, and correction policies.
- [Reproducibility](REPRODUCIBILITY.md) — run manifests, full-cache tolerance, protocols, and artifact lineage.
- [Repository audit](docs/repository_audit.md) — detected environment and reusable AAFLOW/AAFLOW+ components.
- [Environment/worktree policy](docs/environment_and_worktree_policy.md) and
  [AAFLOW isolation](docs/aaflow_isolation.md) — canonical-run eligibility,
  scratch-only caches, the common environment, and the no-sibling-import rule.
- [Backend capability matrix](docs/backend_capability_matrix.md) and [model capability matrix](docs/model_capability_matrix.md) — source-backed integration boundaries.
- [Evaluation harness](docs/evaluation_harness.md) — task routes, lmms-eval adapter contract, synthetic CI, result storage, and failure handling.
- [HF byte-matched selector curves](docs/hf_byte_matched_curves.md) — identical
  per-sample byte caps for MosaicKV and Prompt Top-K, fail-closed realized-byte
  matching, Slurm execution, and curve artifacts.
- [Hugging Face adapters](docs/huggingface_adapters.md) — explicit cache loop, cache/query metadata, InternVL boundary, and correctness gates.
- [FullKV reference](docs/fullkv_reference.md) — no-compression contract, synchronized timing, memory accounting, workload schema, and local/Slurm commands.
- [Cache-state model](docs/cache_state.md) — block descriptors, tier membership, logical positions, byte accounting, and lossless reinjection.
- [Future-query forecasting](docs/future_query_forecasting.md) — prompt/draft/hybrid modes, GQA mapping, low-memory centroids, isolation, timings, and evaluation-only diagnostics.
- [Sparse evidence graph](docs/evidence_graph.md) — pooled block nodes, typed edge sources, compatibility filters, sparse storage, fallback, and diagnostics.
- [Block utility and selection](docs/block_utility_and_selection.md) — RoPE-aware attention boundary, signal definitions, signed objective, hard budgets, lazy greedy, and exhaustive diagnostics.
- [Three-tier cache construction](docs/three_tier_cache.md) — exact anchors, conservative RoPE gates, weighted prototypes, pinned CPU residuals, layouts, and diagnostics.
- [Decode-time residual repair](docs/decode_time_repair.md) — entropy/risk triggers, evaluation-only oracle isolation, asynchronous restoration, one-pass re-decode, persistent promotion, and budget eviction.
- [Detailed residual-repair experiment](docs/residual_repair_experiment.md) — bounded policy and storage sweep, paired metrics, trace schema, and exploratory execution protocol.
- [Hyperparameter-generalization experiment](docs/hyperparameter_generalization_experiment.md) — disjoint calibration/freeze boundary, held-out model/task matrix, backend support boundary, and executed exploratory evidence.
- [Unified Hugging Face runtime](docs/huggingface_runtime.md) — method orchestration, packed-cache masks, safety fallbacks, YAML/CLI usage, traces, and validation progression.
- [Decode-latency optimization audit](docs/decode_latency_optimization.md) — implemented fast paths, diagnostic evidence, and correctness-gated next optimizations.
- [vLLM backend](docs/vllm_backend.md) and [native transaction status](docs/vllm_native_blocker.md) — version-pinned atomic compaction, A100 evidence, fail-closed boundaries, and the remaining selector-plan gate.
- [vLLM published-baseline projections](docs/vllm_published_baselines.md) — hash-bound whole-block projections for `lookm_reimpl`, `prefixkv_reimpl`, and `vl_cache_reimpl`, with explicit semantic limits and result labels.
- [SGLang backend](docs/sglang_backend.md) and [native transaction status](docs/sglang_native_blocker.md) — spawn-safe allocator/mapping compaction, A100 evidence, and the remaining selector-plan gate.
- [Simple baselines](docs/simple_baselines.md) — exact-only policies, common budgets, deterministic allocation, traces, and configuration.
- [LOOK-M specification](docs/baselines/lookm_spec.md) and [parity report](docs/baselines/lookm_parity_report.md) — pinned official source, paper equations, source deviations, unified `lookm_reimpl`, and current comparison blockers.
- [PrefixKV specification](docs/baselines/prefixkv_spec.md) and [parity report](docs/baselines/prefixkv_parity_report.md) — adaptive layer profiles, ratio conventions, fixed-distance decoding, strict leakage checks, unified `prefixkv_reimpl`, and the completed non-canonical official LLaVA parity diagnostic.
- [Published KV-policy specification](docs/baselines/published_kv_spec.md) — pinned source mapping, formulas, implementation choices, and fail-closed execution for `elastic_cache_reimpl`, `h2o_reimpl`, `streamingllm_reimpl`, `pyramidkv_reimpl`, `snapkv_reimpl`, and `roco_reimpl`.
- [VL-Cache specification](docs/baselines/vl_cache_spec.md) — ICLR post-vision scoring and sparsity allocation equations, ambiguity decisions, leakage-safe calibration, sensitivity diagnostics, and explicit non-reproduction status for `vl_cache_reimpl`.
- [Paper-native baseline evaluation](docs/baselines/paper_native_evaluation.md) — separate `paper_reproduction` and `unified_comparison` artifact lanes; PrefixKV PPL/ROUGE, VL-Cache CIDEr/ANLS/MathVista, and LOOK-M MileBench tracks; byte-matched controls; and source-hashed paper-versus-local reproduction deltas.

## License

See the repository [LICENSE](../LICENSE). Pinned third-party components retain their own licenses and attribution under [`third_party/`](../third_party/README.md).
