# Reproducible common environment

MosaicKV uses one CPython 3.11 environment for its Hugging Face, vLLM,
SGLang, evaluation, and development paths:

```text
/scratch/djy8hg/env/mosaickv
```

The authoritative inputs are `env/common/requirements.in` and
`env/common/requirements.lock`. No installed-package patch is part of the
active environment. The lock contains every direct and transitive
Python distribution as an exact `name==version` pin for Linux x86_64. It was
resolved with uv for a manylinux 2.28 target. The core intersection is:

| Component | Common version |
|---|---:|
| Python | 3.11 |
| PyTorch / CUDA wheel runtime | 2.8.0 / CUDA 12.8 |
| Transformers | 4.55.2 |
| FlashAttention | 2.8.3 |
| vLLM | 0.10.2 |
| SGLang | 0.5.1.post3 |
| lmms-eval | 0.7.2 |

`env/common` is the only documented installation source and `common` is the
only documented argument to the setup script.

SGLang's runtime packages are pinned explicitly instead of installing its
`runtime-common` extra. That extra fixes `outlines==0.1.11`, while vLLM fixes
`outlines-core==0.2.11`; unconstrained deterministic generation does not load
SGLang's optional Outlines integration. The explicit SGLang runtime set is
`flashinfer-python==0.2.14.post1`, `sgl-kernel==0.3.7`,
`torch-memory-saver==0.0.8`, `torchao==0.9.0`, `uvloop==0.21.0`, and
`xgrammar==0.1.23`. This resolver decision does not establish serving parity.

FlashAttention's upstream build script prefers a downloadable binary wheel.
That wheel requires GLIBC 2.32 and cannot load on this cluster's GLIBC 2.28
nodes. Build the pinned 2.8.3 source against the node with:

```bash
sbatch mosaickv/slurm/flash_attn_wheel.sbatch
```

The wheel is written under
`/scratch/djy8hg/cache/mosaickv/flash-attn-wheelhouse-source`; setup prefers it
automatically and force-reinstalls that exact local artifact after lock
synchronization. If the common environment is absent, the Slurm wheel script
bootstraps only the pinned Torch/build prerequisites into its eventual prefix;
`create_envs.sh --sync common` must then reconcile the complete lock. Setup
fails explicitly when the source wheel is absent. Do not treat a resolved pin,
wheel build, or import as SDPA/FlashAttention/vLLM/SGLang model-parity evidence.
The audited SM80 wheel has SHA-256
`f91c8f0e024b253ba29b3cfc178ad3d90dcb37617baed02eb5175ae7213f8560`;
the setup script verifies the build-produced sidecar checksum before installing
the local artifact.
The provided Slurm build applies the versioned source patch
`env/patches/flash-attn-2.8.3-a100-multimodal-inference.patch` (SHA-256
`e919aa142eba57abd98af403de282ee7ecf622ad22d745da22355a935b654768`).
It is an inference-only wheel specialized for SM80, FP16/BF16, and the
64/80/128 head dimensions used by the requested vision and language models;
backward execution, dropout, ALiBi, softcap, and local-attention behavior are
intentionally disabled. Uneven head dimensions remain enabled because Qwen's
vision encoder uses dimension 80. Four backward dispatch
translation units remain linked because the extension loader requires their
symbols, but the public backward entry points fail explicitly before dispatch.
These constraints cover the audited Qwen2.5-VL language/vision dimensions
128/80 and LLaVA language/vision dimensions 128/64, but they are not general
FlashAttention support. Build a separate, explicitly
versioned wheel for another compute capability, head dimension, training, or
kernel feature set. `verify_envs.py --require-cuda` executes dense and varlen
forward kernels at all three dimensions and verifies that the specialized
wheel rejects an uncompiled head dimension before reporting environment
support.
The upstream 2.8.3 source archive is independently fixed to SHA-256
`1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d`;
the build refuses a different archive before extraction.

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
sbatch mosaickv/slurm/flash_attn_wheel.sbatch
# Wait for the wheel job to complete successfully.
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

Exploratory A100 job `17347802` produced exactly that verdict for lock SHA-256
`fdc56ad9edb03e8bad5cb2108322c7a12b0f33bdd5533ed596dc6b695a2db9c0`:
all 251 locked distributions matched, every required module imported, the CUDA
matrix multiply passed, and the specialized FlashAttention dense/varlen
forward and rejection checks passed. The dirty worktree prevents treating this
as a canonical artifact, but the common environment is no longer merely an
unverified resolved lock.

## Docker

The canonical Dockerfile uses the common lock:

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
