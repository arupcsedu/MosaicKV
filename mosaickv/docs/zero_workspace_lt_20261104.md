# Bounded cuBLASLt source projection, explicit FP32 and no algorithm workspace

Preserve the native-Triton tile negative. Source words and corrected K/V passed,
repairs were small/zero, but operator cost was 3.9–4.2x the original; register
use was 239/255. No full-request retry is justified for that kernel.

The historical full-wide native GEMM was faster but needed full source/output
buffers. Its 128-row alternative had large corrections, plausibly due to a
different accumulation/reduction algorithm. That cause is a hypothesis, NOT
established profiling evidence. This successor explicitly controls those
properties rather than inheriting PyTorch's selected GEMM behavior.

Use installed pinned cuBLASLt through a checked ctypes ABI. No extra toolkit,
packages, headers or libraries are installed, patched or vendored. Validate
enum values and struct layout against the actual installed headers; record
header/library hashes and version. Create private request descriptors with
BF16 input/output, CUBLAS_COMPUTE_32F, row-major source/output, transposed original
weights, zero max workspace and compute-type-only reduction mask. Choose the
FIRST successful vendor heuristic with zero workspace, NONE/COMPUTE_TYPE
reduction and explicit FP32 accumulator capability; no timing search. Submit
the selected immutable algorithm with null workspace/zero bytes. PyTorch model
matmul flags and model weights are never changed. See the [NVIDIA API](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulpreferenceattributes-t).

Two separate BF16 tensors, each 128x4096, are the ONLY explicit source/output
workspaces: exactly 2 MiB backing. Decode original source/coherent columns per
chunk, run wide GEMM, copy layout into the existing head-major lease, then
apply original RoPE and fixed exact repairs. Charge both buffers and every
repair; no source or output view hides a larger backing. Preserve original
dequantized words independently, freeze repairs once (32,768-byte/layer cap),
and check paddings 0/1/64/127, all K/V words and all history sentinels.

This is a separately preregistered vendor-native operator, not a claim of zero
spills in opaque vendor code. Custom source-unpack kernels must expose zero
scratch/spills; cuBLASLt internal register/spill use is NOT EXPOSED and must
not be invented as zero. The API-guaranteed algorithm workspace is zero;
vendor code/context allocations and whole-request GPU peak remain uncertified.
Do not conflate this tensor-cache boundary with total GPU footprint.

Both sequences use private CUDA graphs. Candidate replay includes decode,
GEMM, layout copies, RoPE and numerical repair. Five warmups, twenty randomized
paired repetitions, 10,000 bootstrap replicates, adjusted operator upper ratio
<=0.80 unchanged. Fitting/descriptor creation is recorded but NOT timed as
operator replay; a future full-request test must include all those costs and
retirement, offered identically to every compatible baseline. Its adjusted
upper ratio remains <=0.96 against all original AND optimized readers.

CPU tests + ABI check, two one-A100/12-CPU <=15min operators, afterany analysis,
reservation bi_fox_dgx, max six aggregate A100s. Caches outside home. No model
or automatic fresh/cohort/calibration/paper run. Six main mechanisms disabled
precision variant, independent quality N=0. Operator success cannot establish
quality or end-to-end superiority.
