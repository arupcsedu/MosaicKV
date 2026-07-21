# Slurm validation

All supported jobs use `/scratch/djy8hg/env/mosaickv`, source the shared
scratch-only cache policy, and reject a dirty worktree.

## Environment smoke

`env_smoke.sbatch` is the canonical no-download GPU environment gate. It
verifies every common-lock distribution, imports each backend in a bounded
subprocess, checks cache placement, validates the installed compatibility
patch, and runs a tiny synchronized CUDA matrix multiplication:

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh
sbatch --reservation=bi_fox_dgx mosaickv/slurm/env_smoke.sbatch
```

Only `status: support_verified`, an empty error list, and
`cuda.matmul_passed: true` establish environment support. This job does not
load model weights and does not establish model/backend parity.

## GPU doctor

`doctor_gpu.sbatch` preserves the requested two-node `bii-gpu` diagnostic
profile. It runs the read-only doctor and synthetic GPU tests without loading
model weights:

```bash
sbatch mosaickv/slurm/doctor_gpu.sbatch
```

The job fails if either task cannot see its allocated GPU. Confirm that the
reservation currently contains two nodes before submission.

## Model and backend jobs

The remaining Slurm files are implementation or development harnesses, not
documented replication entry points. Do not submit them merely because the
file exists. Each must first be audited against the common-lock package APIs,
its pinned checkpoint must be available outside the home directory, and its
clean-tree parity gate must be explicitly enabled in the corresponding backend
guide.

In particular, the vLLM and SGLang serving wrappers have not passed a
common-lock model-serving gate. Their FullKV, setup, and HF-parity jobs are
unsupported until the wrapper port and parity work described in
`docs/vllm_backend.md` and `docs/sglang_backend.md` is complete.

## Versioned experiment arrays

Validate and expand a matrix before submitting anything. Expansion writes one
read-only, SHA-named run config per array index and refuses to overwrite an
existing expansion. The validator fails if a runnable sweep contains an
unsupported model/backend/method/task combination.

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh

/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/validate_matrix.py \
  mosaickv/configs/experiments/pilot.yaml

/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/expand_matrix.py \
  mosaickv/configs/experiments/pilot.yaml \
  --output-directory /scratch/djy8hg/runs/mosaickv/matrices/pilot-v1

INDEX=/scratch/djy8hg/runs/mosaickv/matrices/pilot-v1/jobs.jsonl
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/validate_matrix.py --expanded-index "${INDEX}"
COUNT=$(wc -l < "${INDEX}")
sbatch --array="0-$((COUNT - 1))" \
  --export="ALL,MOSAICKV_MATRIX_INDEX=${INDEX}" \
  mosaickv/slurm/mosaickv_array.sbatch
```

Re-submitting the same array index uses the same deterministic run ID. The
evaluation harness resumes incomplete JSONL samples and validates completed
immutable manifests. Each Slurm attempt writes logs under that run's
`attempts/<job-id>/` directory, so retries do not overwrite failure evidence.

`main_performance.yaml` and `repair.yaml` are intentionally disabled. Their
required protocols are versioned, but expansion emits no runnable jobs until
the compressed-method microbenchmark and adapter prototype/repair gates exist.

## Cache and token policy

All jobs must source `mosaickv/scripts/cache_env.sh`. Model, dataset, compiler,
backend, temporary, and result caches must stay under `/scratch/djy8hg`.
`HF_TOKEN`, when needed, is inherited from the process environment and is
never written to a script, log, configuration, or manifest.
