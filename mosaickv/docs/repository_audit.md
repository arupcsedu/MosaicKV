# Repository and environment audit

Audit target: clean MosaicKV common environment and repository state on
2026-07-21. This document describes only the active installation path.

## Repository layout

```text
MosaicKV/
├── AGENTS.md
├── mosaickv/
│   ├── pyproject.toml
│   ├── src/mosaickv/
│   │   ├── adapters/
│   │   ├── backends/
│   │   ├── baselines/
│   │   ├── evaluation/
│   │   ├── forecasting/
│   │   ├── graph/
│   │   ├── prototypes/
│   │   ├── repair/
│   │   ├── residual/
│   │   └── selection/
│   ├── configs/
│   ├── docs/
│   ├── env/common/
│   ├── scripts/
│   ├── slurm/
│   └── tests/{unit,integration,gpu}/
└── third_party/{LOOK-M,PrefixKV}/
```

The installable package provides strict configuration, manifests, JSON
logging, CLI diagnostics, synthetic evaluation, cache states, Hugging Face
adapters, MosaicKV components, baselines, and backend prototypes. Presence of
source code is not completion or model-support evidence.

## AAFLOW isolation

MosaicKV has no AAFLOW or AAFLOW+ runtime dependency. No sibling checkout is
added to `PYTHONPATH`, and no source is dynamically imported from outside the
package. Patterns considered reusable during design were standardized message
construction, deterministic sample selection, manifests, and append-only
results; MosaicKV implements these behind its own package boundary. Any future
code reuse must be pinned, licensed, attributed, and isolated under
`third_party/` or a MosaicKV compatibility namespace.

## Python and environment manager

- Python: CPython 3.11.15.
- Environment: `/scratch/djy8hg/env/mosaickv`.
- Authoritative dependency input: `mosaickv/env/common/requirements.in`.
- Exact lock: `mosaickv/env/common/requirements.lock`.
- Synchronizer: `uv==0.11.29`.
- Bootstrap: micromamba when available, otherwise a healthy CPython 3.11
  `venv`.
- Package install: editable with dependency resolution disabled after exact
  lock synchronization.

The setup script refuses a dirty worktree and does not run heavyweight backend
imports on the login node.

## Installed packages

| Package | Common version | Status |
|---|---:|---|
| torch | 2.5.1 | Installed; CUDA 12.4 wheel runtime |
| transformers | 4.49.0 | Installed |
| accelerate | 1.13.0 | Installed |
| flash-attn | — | Not installed; standalone FlashAttention-2 unsupported |
| vllm | 0.7.2 | Installed; import verified |
| sglang | 0.4.3.post4 | Installed; patched registration compatibility verified |
| lmms_eval | 0.7.2 | Installed |
| datasets | 4.1.1 | Installed |

`pip check` passes. The exact verifier checks every distribution in the common
lock rather than relying on this summary table.

## CUDA and GPU

Clean-tree Slurm job `17182570` verified:

- NVIDIA A100-SXM4-80GB;
- one GPU for the environment smoke;
- NVIDIA driver 595.71.05;
- PyTorch CUDA runtime 12.4; and
- a synchronized CUDA matrix multiplication.

GPU availability on a login node is not used as validation evidence. Hardware
and backend support must be rechecked on the allocated node for every canonical
run.

## Cache and secret boundary

`mosaickv/scripts/cache_env.sh` places pip, uv, Hugging Face, datasets, Torch,
Triton, compiler, backend, Ray, and temporary caches under
`/scratch/djy8hg/cache/mosaickv`. No cache may resolve under the home directory.
`HF_TOKEN` is consumed only from the process environment and is never written
to a file.

## Current support boundary

The package/import/CUDA environment smoke passed. The synthetic CPU retention
test also passed with exact equivalence. Exact checkpoint loading and serving
parity have not been rerun under the common lock, so model/backend combinations
remain unsupported until their clean-tree gates pass. See
`model_capability_matrix.md` and `backend_capability_matrix.md`.

Docker definitions use the same common lock, but this cluster provides no
Docker-compatible engine or rootless prerequisites. No Docker run is claimed;
see `docker_verification.md`.
