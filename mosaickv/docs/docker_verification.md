# Docker verification status

Status: **blocked by cluster platform; no Docker run is claimed**.

## What is container-ready

The canonical `mosaickv/Dockerfile` builds one common image. It:

- use PyTorch 2.5.1 with CUDA 12.4;
- install all 243 exact common-lock distributions into
  `/opt/mosaickv-venv`;
- apply and verify the versioned SGLang/Transformers compatibility patch;
- expose the locked wheel-provided NVRTC library to native kernels;
- install MosaicKV editable without dependency resolution;
- place pip, uv, Hugging Face, compiler, backend, and temporary caches under
  `/scratch/djy8hg/cache/mosaickv`; and
- run `scripts/docker_smoke.sh` as a non-root user without loading model
  weights.

The host common environment exercises the same lock, patch, cache bootstrap,
verifier, doctor, and synthetic smoke used by the image entry point. This is
useful compatibility evidence, but it is not a substitute for an actual image
build and `docker run`.

## Platform evidence

Slurm job `17183762` ran on `udc-an26-1` from clean commit `fc40534` and found:

```text
docker=unavailable
podman=unavailable
nerdctl=unavailable
docker_socket=absent
```

Slurm job `17183814` checked whether a private rootless engine could be used.
Although `newuidmap`, `newgidmap`, and `unshare` are installed, the account has
no `/etc/subuid` or `/etc/subgid` allocation; `fuse-overlayfs` and
`slirp4netns` are absent; and the unprivileged-user-namespace control is not
available. Installing a daemon or changing cluster account mappings is outside
the repository and user environment scope.

Apptainer 1.4.5 and 1.5.0 are available as modules. An Apptainer run must not
be reported as a Docker run, so it was not used to satisfy this gate.

## Required external verification

On a host with Docker Engine and NVIDIA Container Toolkit, from a clean tree:

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh
mosaickv/scripts/assert_clean_worktree.sh

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

Preserve the image digest and complete output. Docker support remains
unverified until both commands exit zero; model/backend parity is a separate
gate.
