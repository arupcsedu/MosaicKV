# Environment and clean-worktree policy

Status: mandatory for canonical AAAI 2027 artifact runs.

## Eligibility boundary

A run is paper-eligible only when all of the following are true:

1. `mosaickv/scripts/assert_clean_worktree.sh` succeeds immediately before the
   run;
2. `/scratch/djy8hg/env/mosaickv` exactly matches
   `env/common/requirements.lock` and every versioned patch in `env/patches/`
   is verified as applied exactly once;
3. `mosaickv/scripts/cache_env.sh` is sourced and every reported cache path is
   outside the user's home directory;
4. the manifest records the clean git SHA, common-lock SHA, resolved config
   SHA, and all provenance required by `AGENTS.md`; and
5. the relevant CPU, CUDA, model, and backend parity gates pass from that SHA.

All result files produced before this policy was introduced, including runs
from the old backend-specific environments or a dirty worktree, are
exploratory. They may diagnose behavior but must not be copied into a measured
paper table. Re-run them from a clean common-environment commit.

## Environment lifecycle

There is one backend environment: `/scratch/djy8hg/env/mosaickv`. The
backend-specific names accepted by `create_envs.sh` are deprecated aliases and
resolve to this same prefix. Ordinary run jobs never install or upgrade
packages. Changes to dependencies require this order:

1. edit `env/common/requirements.in`;
2. resolve and review the candidate lock under the scratch cache;
3. commit the input, lock, any version-specific patch, setup code, and
   documentation;
4. confirm the worktree is clean;
5. run `create_envs.sh --sync common`;
6. run the bounded import and CUDA smoke through Slurm;
7. record verification evidence in a follow-up commit.

A successful resolution or install is not a support claim. Failed import,
CUDA, parity, and container checks remain visible and block the affected claim.

## Cache boundary

The shared cache script places pip, uv, model, dataset, compilation, server,
and temporary caches under `/scratch/djy8hg/cache/mosaickv`. It deliberately
ignores user pip configuration with `PIP_CONFIG_FILE=/dev/null`. Slurm and
Docker entry points source the same policy. The Docker runtime mounts the host
scratch cache at the identical absolute path, avoiding path-dependent manifests
and accidental downloads into a container user's home.

Shared-filesystem/network setup uses a 300-second uv HTTP timeout, ten retries,
and four concurrent downloads. These values affect installation reliability,
not runtime measurements, and are versioned here because container and host
setup must behave alike.

Environment verification imports each module in an isolated subprocess with a
hard wall-clock deadline. Run the complete import/backend surface through
Slurm; do not use a login node for heavyweight backend imports.

## Containers

The canonical `mosaickv/Dockerfile` and the three legacy-named Dockerfiles all
install `env/common/requirements.lock` into `/opt/mosaickv-venv`. An image is
identified by its digest and the clean git SHA supplied as
`MOSAICKV_GIT_SHA`. Container verification must preserve both the CPU-safe
`docker run` output and, on a GPU host, the output with
`MOSAICKV_REQUIRE_CUDA=1`. If Docker is unavailable on the host, record that
platform blocker; an Apptainer run is useful compatibility evidence but must
not be described as a Docker run.
