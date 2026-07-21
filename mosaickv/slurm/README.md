# Slurm validation

All current jobs use `/scratch/djy8hg/env/mosaickv`, source the shared
scratch-only cache policy, and reject a dirty worktree. The older job IDs below
are historical diagnostic evidence from backend-specific environments; they do
not validate the current common lock and are not paper eligible.

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

`vllm_fullkv.sbatch` runs one pinned Qwen2.5-VL-3B image/prompt through vLLM
FullKV, repeats the identical request for prefix/multimodal-cache observation,
and validates the raw trace. It never enables native MosaicKV and defaults to
offline model loading from the external shared cache:

```bash
sbatch slurm/vllm_fullkv.sbatch
```

The job refuses to fall back to another Python environment. Create and verify
`/scratch/djy8hg/env/mosaickv` explicitly before submission with
`mosaickv/scripts/create_envs.sh --sync common`.

If an explicitly requested environment installation was interrupted after the
virtualenv was created, submit `vllm_setup_and_fullkv.sbatch` once. It resumes
the exact lock without dependency re-resolution, runs `pip check` and the CUDA
environment verifier, and then delegates to the ordinary immutable-environment
benchmark job. Normal benchmark runs must continue to use
`vllm_fullkv.sbatch`; they do not install or modify packages.

`vllm_hf_fullkv_parity.sbatch` produces a separate HF eager FullKV row with
the same pinned Qwen checkpoint, synthetic sample, generation length, seed,
precision, and block configuration. Compare its manifest input hashes and HF
trace token IDs with a successful vLLM trace; do not combine their latency
rows because the attention backend is intentionally different.

`sglang_fullkv.sbatch` launches the pinned Qwen2.5-VL checkpoint through a
persistent SGLang HTTP server with deterministic inference, overlap scheduling
disabled, CUDA graphs disabled, and server warmup skipped. It defaults to two
distinct synthetic requests, repeats each request for Radix prefix-cache
telemetry, validates exact logical active-KV bytes, and checks unique
session-free request IDs. It explicitly loads `gcc/11.4.0` for SGLang's C++20
TVM-FFI JIT kernels and places that JIT cache under the external cache root:

```bash
sbatch slurm/sglang_fullkv.sbatch
```

The job never installs packages. Use `sglang_setup_and_fullkv.sbatch` only for
an explicitly authorized first install or interrupted-install recovery. The
7B smoke uses the same immutable environment with
`MOSAICKV_SGLANG_CONFIG=configs/sglang_fullkv_7b.yaml`; it may download the
pinned checkpoint only when `MOSAICKV_ALLOW_MODEL_DOWNLOAD=1` is set and
`HF_TOKEN`, if required, is already present in the process environment.

SGLang validation record: environment job `17159768_2` completed with
`support_verified=true`; final Qwen2.5-VL-3B job `17160103` and controlled HF
eager job `17160104` completed on `udc-an26-1`. The cross-backend sample
comparisons failed token parity and remain failed artifacts, so no optimized
SGLang profile is enabled. Final 7B job `17160299` passed the same Stage A
checks. All are dirty-worktree development checks, not paper-eligible result
rows; see the [SGLang backend guide](../docs/sglang_backend.md).

Validation record: setup/recovery job `17157973` passed the exact environment
and CUDA step, and standalone environment job `17158688_1` completed with
`support_verified=true`. Final vLLM job `17158441` and HF parity job `17158501`
completed on `udc-an26-1`. The generated parity artifact records identical
input hashes, 16/16 matching token IDs, and identical decoded text. These
dirty-worktree development runs are not paper-eligible results.

On 2026-07-19, `scontrol show reservation bi_fox_dgx` reported only one node in
the live reservation, so the two-node profile was not schedulable. The same
diagnostic was validated with a one-node, short backfill override as Slurm job
`17091854`; it completed on `udc-an26-1`, detected one NVIDIA A100-SXM4-80GB and
driver 595.71.05, loaded no model weights, passed the exact synthetic smoke
test, and passed the GPU pytest (`1 passed`). Re-check reservation membership
before using any node-count override.
