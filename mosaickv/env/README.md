# Reproducible common environment

MosaicKV uses one CPython 3.11 environment for its Hugging Face, vLLM,
SGLang, evaluation, and development paths:

```text
/scratch/djy8hg/env/mosaickv
```

The authoritative inputs are `env/common/requirements.in`,
`env/common/requirements.lock`, and the version-specific patch under
`env/patches/`. The lock contains every direct and transitive
Python distribution as an exact `name==version` pin for Linux x86_64. It was
resolved with uv for a manylinux 2.28 target. The core intersection is:

| Component | Common version |
|---|---:|
| Python | 3.11 |
| PyTorch / CUDA wheel runtime | 2.5.1 / CUDA 12.4 |
| Transformers | 4.49.0 |
| vLLM | 0.7.2 |
| SGLang | 0.4.3.post4 |
| lmms-eval | 0.7.2 |

SGLang 0.4.3.post4 declares its SRT dependency as
`vllm>=0.6.4.post1,<=0.7.2`; 0.7.2 is therefore the newest vLLM in that
published intersection. Newer previously audited releases cannot share an
environment: vLLM 0.11.0 requires `outlines-core==0.2.11`, whereas SGLang
0.5.5's `outlines==0.1.11` requires `outlines-core==0.1.26`. The old
`env/hf`, `env/vllm`, `env/sglang`, and `env/mock` files are retained only
for historical auditability. They must not drive a new run.

The common input spells out SGLang's SRT dependency set instead of activating
the legacy `srt` extra. That extra requires `decord==0.6.0`, whose only Linux
wheel is tagged for CPython 3.6 and fails both pip and uv platform checks under
Python 3.11. The lock uses `decord2==3.4.0`, which provides the same `decord`
module API used by SGLang (`VideoReader` and `cpu`) in a CPython 3.11,
manylinux 2.28 wheel. This is an environment compatibility decision, not an
algorithm change, and video support remains unverified until a video smoke
passes.

SGLang 0.4.3.post4 predates Transformers' native Qwen2.5-VL registration. Its
bundled compatibility config uses the same class name, so Transformers 4.49
rejects the SGLang processor registration unless `exist_ok=True` is passed.
`scripts/apply_env_patches.sh` applies the two-argument registration patch only
to exactly SGLang 0.4.3.post4, recognizes an already-patched source, and fails
closed on any other source state. The patch SHA and patched target SHA are
verified before support can be claimed and the patch SHA is stored in every run
manifest. This does not change inference math.

The PyTorch lock already includes `nvidia-cuda-nvrtc-cu12==12.4.127`.
`cache_env.sh` adds that wheel's exact library directory to `LD_LIBRARY_PATH`
so `sgl_kernel` does not depend on an unrecorded cluster CUDA module.

Standalone FlashAttention-2 is intentionally absent from the common lock. HF
correctness starts with eager attention. vLLM and SGLang use the kernels pinned
by their own dependency graph. Do not claim SDPA, FlashAttention-2, vLLM, or
SGLang support until the relevant clean-tree smoke passes.

## Cache policy

No MosaicKV command may cache under the user's home directory. Source the
shared policy before setup, tests, model access, server launch, or evaluation:

```bash
source mosaickv/scripts/cache_env.sh
```

The default root is `/scratch/djy8hg/cache/mosaickv`. The policy covers pip,
uv, XDG, Hugging Face Hub/assets/datasets, Transformers, Torch, TorchInductor,
Triton, Numba, CUDA JIT, FlashInfer, vLLM, SGLang, pre-commit, Matplotlib,
Weights & Biases, Ray, and temporary files. It also sets
`PIP_CONFIG_FILE=/dev/null` so user pip configuration cannot silently change
the cache or index. Override only with another absolute path outside home:

```bash
export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv
source mosaickv/scripts/cache_env.sh
```

`HF_TOKEN` is read only from the process environment. Never put its value in a
requirements file, Dockerfile, Slurm script, command-line argument, or checked
artifact.

## Clean-tree environment setup

The setup script refuses a dirty or untracked worktree. It also refuses to
mutate an existing environment unless `--sync` is explicit:

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh
mosaickv/scripts/assert_clean_worktree.sh
mosaickv/scripts/create_envs.sh --sync common
```

For a new prefix, omit `--sync`. The script uses micromamba when available,
otherwise a healthy CPython 3.11, then reconciles the exact lock with uv and
installs MosaicKV editable without dependency resolution. An import-only check
is deliberately not run by the setup script: on this cluster, imports of native
backend stacks belong in the bounded Slurm smoke. A successful synchronization
does not establish GPU/backend support.

Regenerate and compare the lock with:

```bash
mosaickv/scripts/lock_common_env.sh
```

## Slurm verification

The GPU smoke never installs packages or loads model weights. It verifies all
locked distributions, imports every module in an isolated subprocess with a
120-second per-import deadline, checks that all cache paths are outside home,
and runs a tiny synchronized CUDA matrix multiply:

```bash
sbatch --reservation=bi_fox_dgx mosaickv/slurm/env_smoke.sbatch
```

Only `status: support_verified`, an empty error list, and
`cuda.matmul_passed: true` establish environment support. Model/backend parity
remains a separate gate.

## Docker

All four Dockerfile entry points use the same common lock; the backend-named
files are compatibility aliases, not different environments:

```bash
GIT_SHA=$(git rev-parse HEAD)
docker build --build-arg MOSAICKV_GIT_SHA="$GIT_SHA" \
  -f mosaickv/Dockerfile -t mosaickv:common .

docker run --rm \
  -v /scratch/djy8hg/cache/mosaickv:/scratch/djy8hg/cache/mosaickv \
  mosaickv:common

docker run --rm --gpus all -e MOSAICKV_REQUIRE_CUDA=1 -e HF_TOKEN \
  -v /scratch/djy8hg/cache/mosaickv:/scratch/djy8hg/cache/mosaickv \
  mosaickv:common
```

The CPU-safe default run verifies pins and runs the synthetic smoke without
loading model weights. The GPU form additionally imports vLLM, SGLang,
FlashInfer, and their native surfaces and runs CUDA matrix multiplication.
Model caches remain on the mounted scratch path. A successful image build is
not evidence of backend support; preserve the complete `docker run` output and
image digest with the run manifest.
