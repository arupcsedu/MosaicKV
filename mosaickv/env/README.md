# Reproducible environments

MosaicKV uses independent Python 3.11 environments because the audited backend
releases require incompatible Torch, Transformers, and CUDA package stacks.
Every non-comment row in a `requirements.lock` file is an exact `name==version`
pin resolved for Linux x86_64 and CPython 3.11. Lock resolution is not runtime
validation: HF, vLLM, and SGLang remain **unverified** until their import and
CUDA smoke command completes on the target host.

## Purpose and compatibility

| Environment | Purpose | Core stack | CUDA/driver contract | Current support status |
|---|---|---|---|---|
| `hf` | Eager full-cache reference, cache research, and lmms-eval quality evaluation | Torch 2.11.0, Transformers 4.57.6, lmms-eval 0.7.2, FlashAttention-2 2.8.3.post1 | CUDA 13.0 runtime and devel toolkit; Linux driver major >=580; FA2 on Ampere/Ada/Hopper | Unverified |
| `vllm` | vLLM 0.11.2 integration work | Torch 2.9.0, Transformers 4.57.6, vLLM 0.11.2 | CUDA 12.8 wheel stack; Linux driver major >=525; compute capability >=7.0 | Unverified |
| `sglang` | SGLang 0.5.10.post1 integration work | Torch 2.9.1, Transformers 5.3.0, SGLang 0.5.10.post1 | CUDA 12.8/12.9 wheel stack; Linux driver major >=525; compute capability >=8.0 | Unverified |
| `mock` | CPU-only unit tests, linting, typing, and synthetic smoke tests | NumPy, pytest, Ruff, MyPy, pre-commit | No CUDA/backend support | Mock-only |

Standalone FlashAttention-2 is present only in `hf`, where the target A100 and
CUDA toolkit meet its documented support boundary. vLLM uses its pinned native
attention dependencies. SGLang itself requires `flash-attn-4`; that distinct
upstream dependency remains pinned and must not be reported as FA2.

The CUDA driver thresholds follow NVIDIA's major-version compatibility table:
CUDA 13.x needs driver major 580 or newer and CUDA 12.x needs driver major 525
or newer. The installed A100 driver observed by the earlier Slurm doctor was
595.71.05, but each environment must be verified again after creation.

## Cache and token policy

Use a shared cache outside the home directory. The provided scripts default to
`/scratch/djy8hg/cache/mosaickv` and set `HF_HOME`, `HF_HUB_CACHE`,
`HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`, and
the backend cache variable. `verify_envs.py` fails if a required cache is unset,
relative, or below the current home directory.

`HF_TOKEN` is optional and is read only from the process environment. Do not
put it in a requirements file, Dockerfile, Slurm script, `.env` file, command
argument, or repository artifact. To use an already exported token, pass its
name without a value (`docker run -e HF_TOKEN ...`); the verifier reports only
whether it is present.

## Local creation and verification

Creating an environment is an explicit action. No setup script runs during
package installation, tests, or import. From the repository root:

```bash
export MOSAICKV_PYTHON=/scratch/djy8hg/env/drc_rag_benchmarks_yml_20260421/bin/python
export MOSAICKV_ENV_ROOT=/scratch/djy8hg/env
export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv
./mosaickv/scripts/create_envs.sh mock
./mosaickv/scripts/create_envs.sh hf
./mosaickv/scripts/create_envs.sh vllm
./mosaickv/scripts/create_envs.sh sglang
```

The script refuses to overwrite an existing environment and never persists
`HF_TOKEN`. On a GPU allocation, export the same cache variables and run, for
example:

```bash
export XDG_CACHE_HOME="$MOSAICKV_CACHE_ROOT/xdg"
export HF_HOME="$MOSAICKV_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$MOSAICKV_CACHE_ROOT/datasets"
export TRANSFORMERS_CACHE="$MOSAICKV_CACHE_ROOT/transformers"
export TORCH_HOME="$MOSAICKV_CACHE_ROOT/torch"
export VLLM_CACHE_ROOT="$MOSAICKV_CACHE_ROOT/vllm"
export SGLANG_CACHE_DIR="$MOSAICKV_CACHE_ROOT/sglang"
/scratch/djy8hg/env/mosaickv_hf/bin/python mosaickv/scripts/verify_envs.py \
  --environment hf --require-cuda
```

Omitting `--require-cuda` performs an import-only preflight and emits
`support_verified: false`; it is not evidence of backend support.

## Container setup

Build from the MosaicKV repository root. Tokens are never accepted as build
arguments and model caches are runtime volumes outside the container user's
home:

```bash
docker build -f mosaickv/Dockerfile.hf -t mosaickv-hf:0.1.0 .
docker build -f mosaickv/Dockerfile.vllm -t mosaickv-vllm:0.1.0 .
docker build -f mosaickv/Dockerfile.sglang -t mosaickv-sglang:0.1.0 .

docker run --rm --gpus all -e HF_TOKEN \
  -v /scratch/djy8hg/cache/mosaickv:/var/cache/mosaickv \
  mosaickv-hf:0.1.0
docker run --rm --gpus all -e HF_TOKEN \
  -v /scratch/djy8hg/cache/mosaickv:/var/cache/mosaickv \
  mosaickv-vllm:0.1.0
docker run --rm --gpus all -e HF_TOKEN \
  -v /scratch/djy8hg/cache/mosaickv:/var/cache/mosaickv \
  mosaickv-sglang:0.1.0
```

The image default command is the corresponding CUDA verification. A successful
image build alone does not establish support.

## Slurm setup

`env_smoke.sbatch` is a verification-only array: task 0 is HF, task 1 is vLLM,
and task 2 is SGLang. It never creates or installs an environment and never
loads model weights. After explicit local creation:

```bash
export MOSAICKV_ENV_ROOT=/scratch/djy8hg/env
export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv
sbatch --reservation=bi_fox_dgx mosaickv/slurm/env_smoke.sbatch
```

If `HF_TOKEN` is needed, export it in the submitting shell; `#SBATCH
--export=ALL` forwards the environment without embedding the value in the job
script. Check that the reservation is active and contains a compatible node
before submitting. Preserve each verifier JSON record with the environment
lock SHA and run manifest. Only a row with `status: support_verified`, no
errors, and `cuda.matmul_passed: true` supports a runtime availability claim.
