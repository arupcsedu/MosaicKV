# V10 review follow-up

V9 passed the corrected, evaluation-only split-prefill control on both
consumed COCO inputs. All 32 predictions matched, maximum logit error was
zero, and both initial and final caches were bitwise equal. This does not
replace the failed historical native path, validate the lossy BF16 source
variant, or certify latency or independent quality.

V10 advances three independent tasks:

- **R07:** Calculate exactly reachable *initial* storage from the saved V7
  backing-allocation records. Charge cross-layer shared headers once.
  Distinguish existing uniform protection plans from conditional
  combinations of layer-local costs. Do not build or benchmark those
  combinations, change mandatory positions, or add dummy allocations.
- **R08:** Reconstruct the bytes of four preselected generated configs using
  their indexed canonical JSON manifests and pinned serializer. Validate
  exact ledger and shard identities without loading the historical YAML.
  Retain all original errors. This does not certify media, source groups, or
  freshness. Entries without saved canonical JSON remain unresolved.
- **R09:** Measure one consumed image at three fixed synthetic text extensions
  on the existing BF16 full-MHA path. Measure raw BF16 prompt K/V, 3-bit
  shared-source, and local direct projected-K/V 3-bit separately with the
  same processed input at each length, no replay, and 128 predictions.
  Compare sampled prompt fill cost, not benchmark quality or native E2E
  latency. Keep observed workspace, allocator peaks, and persistent
  backing bytes distinct. GQA source construction remains unsupported.

The V10 registration binds the V9 report, inventory, code, and separate
section-anchored manuscript proposal. The proposal is for author review;
it does not modify the submitted PDF or certify any approval.

## Reproduction

Use the clean V10 checkout with all six pinned submodules initialized.
For CPU checks, direct caches to a scratch directory inside the workspace.
The Slurm worker uses the existing offline shared checkpoint cache.

```bash
export MOSAICKV_CACHE_ROOT=/sfs/weka/scratch/djy8hg/workdir/fox-review-v10/cache
source mosaickv/scripts/cache_env.sh
module load gcc/11.4.0
export CXX="$(command -v g++)" CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
python=/scratch/djy8hg/env/mosaickv/bin/python
out=/sfs/weka/scratch/djy8hg/workdir/fox-review-v10/run-20261110-1
bash mosaickv/scripts/assert_clean_worktree.sh
"$python" mosaickv/scripts/evicastkv_review_closure_v2.py tests --root "$out/full-cpu"
"$python" mosaickv/scripts/evicastkv_review_v10.py prepare --root "$out" --execute
unset MOSAICKV_CACHE_ROOT CUDA_VISIBLE_DEVICES
"$python" mosaickv/scripts/evicastkv_review_v10.py submit --root "$out" --execute
```

The launcher is dry by default. It admits at most one diagnostic A100
worker (12 CPUs, 20 minutes), observes the existing aggregate six-A100
ceiling, and submits an `afterany` CPU analyzer. All outputs are write-once.
Failed or unacknowledged submissions are recorded, not automatically
retried. No campaign, paper result, or production-default change follows
from a diagnostic pass.
