#!/bin/bash
#SBATCH --job-name=evicast-cpu-suite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
set -euo pipefail
cd "$1"
test "$(git rev-parse HEAD)" = "$3"
bash mosaickv/scripts/assert_clean_worktree.sh
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
if [[ "$4" == worker ]]; then
  exec /scratch/djy8hg/env/mosaickv/bin/python -m pytest mosaickv/tests/unit \
    -q -p no:cacheprovider --junitxml="$2/results.xml"
fi
export EVICAST_CPU_AUDIT_ROOT="$2" EVICAST_CPU_AUDIT_SHA="$3"
exec /scratch/djy8hg/env/mosaickv/bin/python - <<'PY'
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from mosaickv.evaluation.fox_campaign import sha, write_once
root = Path(os.environ['EVICAST_CPU_AUDIT_ROOT'])
path = root / 'results.xml'
result = dict(source_sha=os.environ['EVICAST_CPU_AUDIT_SHA'], GPU_requests=0)
if not path.exists():
    result.update(status='incomplete', errors=['JUnit missing'])
else:
    xml = ET.parse(path).getroot()
    suites = [xml] if xml.tag == 'testsuite' else list(xml.iter('testsuite'))
    counts = {k: sum(int(s.get(k, 0)) for s in suites)
              for k in ['tests', 'errors', 'failures', 'skipped']}
    passed = counts['tests'] > 0 and not any(counts[k] for k in ['errors', 'failures', 'skipped'])
    result.update(counts=counts, status='passed' if passed else 'failed', junit_sha256=sha(path))
    result['failed_cases'] = [dict(name=c.get('name'), classname=c.get('classname'))
                              for c in xml.iter('testcase')
                              if c.find('failure') is not None or c.find('error') is not None]
write_once(root / 'GATE_cpu_suite.json', result)
print(result)
PY
