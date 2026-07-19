# MosaicKV

MosaicKV is a planned training-free multimodal KV-cache compression system for an AAAI 2027 research paper. It aims to preserve task-relevant multimodal context under a fixed cache budget by combining:

1. future-query forecasting;
2. sparse cross-modal evidence graph construction;
3. value-aware block utility;
4. budgeted submodular selection;
5. exact, prototype, and residual cache tiers; and
6. uncertainty-guided residual repair.

## Project status

The repository now contains the installable research-infrastructure scaffold: strict configuration, provenance manifests, structured logging, static model inspection, environment diagnostics, a unified lmms-eval adapter, append-only evaluation result storage, and synthetic CPU validation. The MosaicKV compression algorithm itself has **not** been implemented, and this repository currently reports **no measured experimental results**. Design language describes intended behavior, not validated performance.

The planned implementation sequence covers a Hugging Face full-cache reference, the backend-independent MosaicKV core, correctness testing, simple and published baselines, quality and systems evaluation, vLLM and SGLang integrations, and artifact packaging.

## Development quick start

Use the isolated Python 3.11 environment created for this project:

```bash
/scratch/djy8hg/env/mosaickv_dev/bin/python -m pip install -e 'mosaickv[dev]'
/scratch/djy8hg/env/mosaickv_dev/bin/mosaickv doctor
/scratch/djy8hg/env/mosaickv_dev/bin/mosaickv inspect-model llava-hf/llava-1.5-7b-hf
/scratch/djy8hg/env/mosaickv_dev/bin/mosaickv smoke
```

Run `evaluate --config mosaickv/configs/smoke.toml` or `benchmark` to validate a configuration and create a provenance manifest. These commands deliberately return `status: not_run` until the corresponding research implementation and correctness gates exist.

See [development instructions](docs/development.md) and [GPU diagnostics](slurm/README.md) for local checks and the non-downloading Slurm doctor job.
Separate locked HF, vLLM, SGLang, and CPU-mock environments are documented in
[environment setup](env/README.md). Creating them is always an explicit action.
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
- [Backend capability matrix](docs/backend_capability_matrix.md) and [model capability matrix](docs/model_capability_matrix.md) — source-backed integration boundaries.
- [Evaluation harness](docs/evaluation_harness.md) — task routes, lmms-eval adapter contract, synthetic CI, result storage, and failure handling.

## License

See the repository [LICENSE](../LICENSE). Third-party components, once added, retain their own licenses and attribution.
