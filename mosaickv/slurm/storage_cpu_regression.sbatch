#!/usr/bin/env bash
set -euo pipefail
source_root=${1:?source root}
output_root=${2:?new output directory}
expected_sha=${3:?immutable source SHA}
cd "$source_root"
source mosaickv/scripts/cache_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
test "$(git rev-parse HEAD)" = "$expected_sha"
bash mosaickv/scripts/assert_clean_worktree.sh
if [[ -e "$output_root" ]]; then
  echo "CPU regression output must be write-once" >&2
  exit 2
fi
mkdir -p "$output_root"
git rev-parse HEAD
git submodule status
exec /scratch/djy8hg/env/mosaickv/bin/python -m pytest -q \
  mosaickv/tests/unit mosaickv/tests/integration -m 'not gpu' \
  --tb=short -ra --junitxml="$output_root/cpu_regression.xml"
