# MosaicKV

MosaicKV is a training-free multimodal KV-cache compression research system for an AAAI 2027 paper. It aims to preserve task-relevant multimodal context under a fixed cache budget by combining:

1. future-query forecasting;
2. sparse cross-modal evidence graph construction;
3. value-aware block utility;
4. budgeted submodular selection;
5. exact, prototype, and residual cache tiers; and
6. uncertainty-guided residual repair.

## Project status

The repository contains installable research infrastructure and a unified eager
Hugging Face runtime: strict configuration, provenance manifests, structured
logging, model inspection, environment diagnostics, an lmms-eval adapter,
append-only evaluation storage, explicit prefill/token decode, FullKV, simple
exact-cache baselines, and local `lookm_reimpl`, `prefixkv_reimpl`, and
`vl_cache_reimpl` methods. It also contains forecasting, sparse graph
construction, block utility and selection, conservative cache tiers, and
decode-time repair. Registered HF adapters expose post-RoPE cached keys, so
unsafe prototype merging fails closed to exact selection.

The common lock, package imports, CUDA smoke, static checks, unit tests, and
synthetic 100%-retention equivalence gate pass. Pretrained-checkpoint parity,
dataset progression, vLLM serving, and SGLang serving have not yet passed under
the common lock. Official LOOK-M and PrefixKV source is pinned for inspection,
but official execution is disabled; only the explicitly labeled
reimplementations belong to the common runtime. This repository currently
reports **no paper-eligible measured experimental results**.

The common environment imports the installed vLLM and SGLang stacks and passes
the CUDA smoke. Their repository measurement wrappers have not yet passed a
model-serving parity gate with the versions in the common lock, so no current
vLLM or SGLang FullKV result is supported. Native MosaicKV integration remains
fail-closed and produces no simulated native rows.

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

The common lock uses Torch 2.5.1/CUDA 12.4, Transformers 4.49.0, vLLM 0.7.2,
and SGLang 0.4.3.post4. It is resolver-consistent but remains unverified until
the clean-tree gates pass. The bounded package-import and CUDA smoke passed on
an A100; model/backend parity and native Docker remain separate, unpassed
gates. Standalone FlashAttention-2 is not installed or claimed.

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
- [Hugging Face adapters](docs/huggingface_adapters.md) — explicit cache loop, cache/query metadata, InternVL boundary, and correctness gates.
- [FullKV reference](docs/fullkv_reference.md) — no-compression contract, synchronized timing, memory accounting, workload schema, and local/Slurm commands.
- [Cache-state model](docs/cache_state.md) — block descriptors, tier membership, logical positions, byte accounting, and lossless reinjection.
- [Future-query forecasting](docs/future_query_forecasting.md) — prompt/draft/hybrid modes, GQA mapping, low-memory centroids, isolation, timings, and evaluation-only diagnostics.
- [Sparse evidence graph](docs/evidence_graph.md) — pooled block nodes, typed edge sources, compatibility filters, sparse storage, fallback, and diagnostics.
- [Block utility and selection](docs/block_utility_and_selection.md) — RoPE-aware attention boundary, signal definitions, signed objective, hard budgets, lazy greedy, and exhaustive diagnostics.
- [Three-tier cache construction](docs/three_tier_cache.md) — exact anchors, conservative RoPE gates, weighted prototypes, pinned CPU residuals, layouts, and diagnostics.
- [Decode-time residual repair](docs/decode_time_repair.md) — entropy/risk triggers, evaluation-only oracle isolation, asynchronous restoration, one-pass re-decode, persistent promotion, and budget eviction.
- [Unified Hugging Face runtime](docs/huggingface_runtime.md) — method orchestration, packed-cache masks, safety fallbacks, YAML/CLI usage, traces, and validation progression.
- [vLLM backend](docs/vllm_backend.md) and [native blocker](docs/vllm_native_blocker.md) — common-lock status, installed-source boundary, fail-closed feature flag, and the missing upstream sparse-position interface.
- [SGLang backend](docs/sglang_backend.md) and [native blocker](docs/sglang_native_blocker.md) — common-lock status, installed-source boundary, and the missing atomic sparse-cache interface.
- [Simple baselines](docs/simple_baselines.md) — exact-only policies, common budgets, deterministic allocation, traces, and configuration.
- [LOOK-M specification](docs/baselines/lookm_spec.md) and [parity report](docs/baselines/lookm_parity_report.md) — pinned official source, paper equations, source deviations, unified `lookm_reimpl`, and current comparison blockers.
- [PrefixKV specification](docs/baselines/prefixkv_spec.md) and [parity report](docs/baselines/prefixkv_parity_report.md) — adaptive layer profiles, ratio conventions, fixed-distance decoding, strict leakage checks, unified `prefixkv_reimpl`, and the completed non-canonical official LLaVA parity diagnostic.
- [VL-Cache specification](docs/baselines/vl_cache_spec.md) — ICLR post-vision scoring and sparsity allocation equations, ambiguity decisions, leakage-safe calibration, sensitivity diagnostics, and explicit non-reproduction status for `vl_cache_reimpl`.

## License

See the repository [LICENSE](../LICENSE). Pinned third-party components retain their own licenses and attribution under [`third_party/`](../third_party/README.md).
