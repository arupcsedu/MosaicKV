#!/usr/bin/env bash
# Resource requests and immutable job receipts are supplied by the finite launcher.
set -euo pipefail

if [[ $# != 4 ]]; then
  printf 'Usage: %s SOURCE ROOT tests|prepare|worker|analyze SOURCE_SHA\n' "$0" >&2
  exit 2
fi
SOURCE_ROOT=$1
RUN_ROOT=$2
MODE=$3
EXPECTED_SHA=$4
cd "$SOURCE_ROOT"
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$SOURCE_ROOT/mosaickv/src:$SOURCE_ROOT/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ATEN_CPU_CAPABILITY=avx2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
export MOSAICKV_ALLOW_DIRTY_EXPLORATORY=0
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
DRIVER=mosaickv/scripts/evicastkv_fidelity_byte_v20.py

bash mosaickv/scripts/assert_clean_worktree.sh
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]] || {
  printf 'V20 source SHA differs from its immutable launch receipt\n' >&2
  exit 2
}

if [[ "$MODE" != worker ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

case "$MODE" in
  tests)
    module load gcc/11.4.0
    export CXX="$(command -v g++)"
    # Frozen CPU numerical controls; Slurm reserves four CPUs, torch uses two.
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    "$PYTHON" - "$RUN_ROOT" "$EXPECTED_SHA" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root, revision = Path(sys.argv[1]), sys.argv[2]
target = root / "focused-cpu"
target.mkdir(exist_ok=False)
command = [sys.executable, "-m", "pytest", "-q",
           "mosaickv/tests/unit/test_exact_selected_control.py",
           "mosaickv/tests/unit/test_native_quantized_source_control.py",
           "mosaickv/tests/unit/test_fidelity_byte_v20.py",
           "-o", f"cache_dir={os.environ['MOSAICKV_CACHE_ROOT']}/pytest-fox-v20-focused",
           f"--junitxml={target}/cpu-tests.xml"]
with (target / "cpu-tests.log").open("x") as stream:
    try:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=300, check=False)
        code = completed.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
record = dict(source_sha=revision, command=command, returncode=code,
              status="passed" if code == 0 else "failed", timeout_seconds=300,
              log_sha256=hashlib.sha256((target / "cpu-tests.log").read_bytes()).hexdigest())
with (target / "cpu-tests.json").open("x") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
if code != 0:
    raise SystemExit("focused V20 regressions failed; empirical work remains blocked")
PY
    exec "$PYTHON" mosaickv/scripts/evicastkv_review_closure_v2.py tests \
      --root "$RUN_ROOT/full-cpu" --cpu-timeout-seconds 1200
    ;;
  prepare)
    # An afterany dependency always reaches this callback. Failed CPU tests
    # trigger a CPU analyzer only, never a GPU worker.
    set +e
    "$PYTHON" - "$RUN_ROOT" "$EXPECTED_SHA" <<'PY'
import json
from pathlib import Path
import sys

root, revision = Path(sys.argv[1]), sys.argv[2]
for directory in ("focused-cpu", "full-cpu"):
    record = json.loads((root / directory / "cpu-tests.json").read_text())
    if (record.get("source_sha") != revision or record.get("status") != "passed"
            or record.get("returncode") != 0):
        raise SystemExit(f"{directory} did not pass on the immutable V20 source")
PY
    prepare_status=$?
    if [[ "$prepare_status" == 0 ]]; then
      "$PYTHON" "$DRIVER" prepare --root "$RUN_ROOT"
      prepare_status=$?
    fi
    set -e
    "$PYTHON" - "$RUN_ROOT" "$EXPECTED_SHA" "$prepare_status" <<'PY'
import json
from pathlib import Path
import sys

root, revision, code = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
with (root / "jobs/prepare-completion.json").open("x") as stream:
    json.dump(dict(source_sha=revision, returncode=code,
                   status="passed" if code == 0 else "failed"), stream, indent=2)
    stream.write("\n")
PY
    bash mosaickv/scripts/submit_fox_fidelity_byte_v20.sh \
      --root "$RUN_ROOT" --dispatch --source-sha "$EXPECTED_SHA"
    exit "$prepare_status"
    ;;
  worker)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[01]$ ]] || {
      printf 'Only the registered V20 worker indices 0 and 1 are allowed\n' >&2
      exit 2
    }
    exec "$PYTHON" "$DRIVER" worker --root "$RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID"
    ;;
  analyze)
    [[ "${SLURM_CPUS_PER_TASK:-}" == 12 ]] || {
      printf 'V20 analysis requires the registered twelve-CPU controls\n' >&2
      exit 2
    }
    exec "$PYTHON" "$DRIVER" analyze --root "$RUN_ROOT"
    ;;
  *)
    printf 'Unregistered V20 callback mode: %s\n' "$MODE" >&2
    exit 2
    ;;
esac
