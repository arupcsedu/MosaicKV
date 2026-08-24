# V102 common-environment verification record

Date: 2026-08-22

Verified source snapshot: `309551eb4c1f107d5cd831f96a0baa58125a5d44`

The shared environment at `/scratch/djy8hg/env/mosaickv` was verified against
`env/common/requirements.lock` without installing, upgrading, or removing any
package. All cache and temporary paths were rooted below
`/scratch/djy8hg/cache/mosaickv`, and Python user-site packages were disabled.

## Environment result

- Python: 3.11.15
- Lock SHA-256:
  `fdc56ad9edb03e8bad5cb2108322c7a12b0f33bdd5533ed596dc6b695a2db9c0`
- Exactly verified distributions: 251
- Transformers: 4.55.2
- Tokenizers: 0.21.4
- PyYAML: 6.0.3
- Pytest: 9.0.2
- `pip check`: passed
- Required CPU imports: passed with zero verifier errors
- GPU execution: not requested on the CPU login host

## Repository test result

The complete test tree was run with `PYTHONNOUSERSITE=1` and
`PYTEST_DEBUG_TEMPROOT=/scratch/djy8hg/cache/mosaickv/pytest`:

- passed: 743
- expected skips: 2
- failed: 0
- warnings: 1 (`torch.cuda` could not initialize NVML on the CPU host)
- wall time: 360.26 seconds

The two skips require a visible GPU or an explicitly pinned checkpoint and do
not weaken the CPU gate. The earlier apparent Transformers/Tokenizers mismatch
was user-site leakage from an incompletely bootstrapped command, not a defect
in the common environment or lock. The scratch-rooted pytest temporary path
also satisfies the native-backend artifact guard.

This record verifies the CPU/common-environment gate only. It does not
authorize ledger materialization, disjointness checking, calibration, GPU or
Slurm execution, candidate freezing, held-out evaluation, or a paper claim.
