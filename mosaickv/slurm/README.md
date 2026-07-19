# Slurm validation

`doctor_gpu.sbatch` preserves the requested two-node `bii-gpu` profile and runs
one read-only doctor, one NumPy smoke process, and the GPU-marked pytest suite
per node. It requests one GPU per node, the minimum explicit GPU GRES required
by the partition QOS. It does not download or load model weights. Submit from
the package project directory:

```bash
sbatch slurm/doctor_gpu.sbatch
```

The job intentionally fails if `nvidia-smi` and optional PyTorch detection both
report no visible GPU on either task.

On 2026-07-19, `scontrol show reservation bi_fox_dgx` reported only one node in
the live reservation, so the two-node profile was not schedulable. The same
diagnostic was validated with a one-node, short backfill override as Slurm job
`17091854`; it completed on `udc-an26-1`, detected one NVIDIA A100-SXM4-80GB and
driver 595.71.05, loaded no model weights, passed the exact synthetic smoke
test, and passed the GPU pytest (`1 passed`). Re-check reservation membership
before using any node-count override.
