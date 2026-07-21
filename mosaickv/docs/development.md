# Development environment

HF, vLLM, SGLang, evaluation, and development share the exact common
environment at `/scratch/djy8hg/env/mosaickv`. Do not install into a
backend-specific prefix.

From the repository root, after committing the intended source and lock:

```bash
source mosaickv/scripts/cache_env.sh
mosaickv/scripts/assert_clean_worktree.sh
mosaickv/scripts/create_envs.sh --sync common

/scratch/djy8hg/env/mosaickv/bin/mosaickv doctor
/scratch/djy8hg/env/mosaickv/bin/mosaickv smoke
cd mosaickv
PYTHON_BIN=/scratch/djy8hg/env/mosaickv/bin/python ./scripts/check.sh
```

`cache_env.sh` covers pip, uv, model, dataset, compiler, backend, and temporary
caches and rejects a cache root inside home. The default is
`/scratch/djy8hg/cache/mosaickv`. `HF_TOKEN`, when needed, is inherited from
the process environment and is never written by these scripts.

The common lock is resolver-consistent, but support is not established by an
install. Run the bounded `slurm/env_smoke.sbatch` import/CUDA verifier, then
model/backend parity. Preserve failures. Canonical validation requires a clean
worktree; dirty-tree checks are exploratory.

The canonical MyPy gate is strict over production code under `src/` and the
environment verifier. Tests are exercised by pytest rather than included in
the production typing claim. Isolated official-baseline scripts may depend on
their pinned third-party environments and are intentionally outside the common
MyPy surface.
