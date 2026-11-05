"""Private pinned cuBLASLt BF16 GEMM with explicit FP32 and zero workspace.

ABI values/structures are checked against the installed NVIDIA headers. No
header/library is vendored, installed, patched, or silently substituted.
"""

import ctypes as C
import importlib.metadata
import re
import threading
from functools import lru_cache
from pathlib import Path

import torch

from mosaickv.evaluation.fox_campaign import sha


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class Heuristic(C.Structure):
    _fields_ = [
        ("algo", Algo),
        ("workspaceSize", C.c_size_t),
        ("state", C.c_int),
        ("wavesCount", C.c_float),
        ("reserved", C.c_int * 4),
    ]


def abi_proof():
    import nvidia.cublas
    import nvidia.cuda_runtime

    root = Path(next(iter(nvidia.cublas.__path__)))
    runtime = Path(next(iter(nvidia.cuda_runtime.__path__)))
    files = [
        root / "include/cublasLt.h",
        root / "include/cublas_api.h",
        runtime / "include/library_types.h",
    ]
    text = "\n".join(p.read_text() for p in files)
    enums = dict(
        CUBLASLT_ORDER_ROW=1,
        CUBLASLT_MATRIX_LAYOUT_ORDER=1,
        CUBLASLT_MATMUL_DESC_TRANSB=4,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES=1,
        CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK=3,
        CUBLASLT_REDUCTION_SCHEME_NONE=0,
        CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE=2,
        CUBLASLT_ALGO_CONFIG_ID=0,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME=3,
        CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS=15,
        CUBLAS_COMPUTE_32F=68,
        CUDA_R_16BF=14,
        CUDA_R_32F=0,
        CUBLAS_OP_T=1,
    )
    for name, value in enums.items():
        if not re.search(r"\b" + name + r"\s*=\s*" + str(value) + r"\s*[,}]", text):
            raise RuntimeError(f"unverified installed cuBLASLt enum {name}")
    if (
        C.sizeof(Algo),
        C.sizeof(Heuristic),
        Heuristic.workspaceSize.offset,
        Heuristic.state.offset,
        Heuristic.wavesCount.offset,
        Heuristic.reserved.offset,
    ) != (64, 96, 64, 72, 76, 80):
        raise RuntimeError("unexpected installed 64-bit cuBLASLt ABI")
    if not re.search(r"uint64_t\s+data\[8\];\s*}\s*cublasLtMatmulAlgo_t", text):
        raise RuntimeError("unverified installed algorithm descriptor")
    return dict(
        installed_version=importlib.metadata.version("nvidia-cublas-cu12"),
        header_hashes={str(p): sha(p) for p in files},
        library_path=str(root / "lib/libcublasLt.so.12"),
        enums_verified=enums,
        algo_bytes=64,
        heuristic_bytes=96,
        actual_GEMM_not_simulation=True,
        no_package_installed_or_modified=True,
    )


class Library:
    def __init__(self):
        self.abi = abi_proof()
        self.lib = C.CDLL(self.abi["library_path"])
        self.abi["library_sha256"] = sha(Path(self.abi["library_path"]))
        p, i, z = C.c_void_p, C.c_int, C.c_size_t
        signatures = {
            "cublasLtCreate": [C.POINTER(p)],
            "cublasLtDestroy": [p],
            "cublasLtMatmulDescCreate": [C.POINTER(p), i, i],
            "cublasLtMatmulDescDestroy": [p],
            "cublasLtMatmulDescSetAttribute": [p, i, p, z],
            "cublasLtMatrixLayoutCreate": [C.POINTER(p), i, C.c_uint64, C.c_uint64, C.c_int64],
            "cublasLtMatrixLayoutDestroy": [p],
            "cublasLtMatrixLayoutSetAttribute": [p, i, p, z],
            "cublasLtMatmulPreferenceCreate": [C.POINTER(p)],
            "cublasLtMatmulPreferenceDestroy": [p],
            "cublasLtMatmulPreferenceSetAttribute": [p, i, p, z],
            "cublasLtMatmulAlgoGetHeuristic": [
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                i,
                C.POINTER(Heuristic),
                C.POINTER(i),
            ],
            "cublasLtMatmulAlgoConfigGetAttribute": [C.POINTER(Algo), i, p, z, C.POINTER(z)],
            "cublasLtMatmulAlgoCapGetAttribute": [C.POINTER(Algo), i, p, z, C.POINTER(z)],
            "cublasLtMatmul": [p, p, p, p, p, p, p, p, p, p, p, p, C.POINTER(Algo), p, z, p],
        }
        for name, types in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = types, i
        self.lib.cublasLtGetVersion.restype = z
        self.lib.cublasLtGetStatusName.argtypes = [i]
        self.lib.cublasLtGetStatusName.restype = C.c_char_p
        self.abi["API_version"] = self.lib.cublasLtGetVersion()

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            status = self.lib.cublasLtGetStatusName(code)
            raise RuntimeError(f"{name}: {code} {status.decode() if status else 'unknown'}")


def eligible_heuristic(state, workspace, reduction, flags):
    return (
        all(type(x) is int for x in (state, workspace, reduction, flags))
        and state == 0
        and workspace == 0
        and reduction in (0, 2)
        and flags & 0xFF00 == 0x0200
    )


@lru_cache(maxsize=1)
def library():
    # Immutable library/ABI only, NEVER a CUDA request handle or tensor.
    return Library()


class Plans:
    """CPU descriptors and immutable algorithm choices, no GPU workspace owner."""

    def __init__(self, rows, width):
        if (
            type(rows) is not tuple
            or not rows
            or any(type(n) is not int or not 0 < n <= 128 for n in rows)
            or type(width) is not int
            or width != 4096
        ):
            raise ValueError("registered bounded 128-row/full-width plans required")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("private single-GPU cuBLASLt plan required")
        self.library, self.owners, self.plans = library(), [], {}
        self.owner_thread, self.closed = threading.get_ident(), False
        self.width = width
        try:
            self.handle = self.new("cublasLtCreate", "cublasLtDestroy")
            self.operation = self.new(
                "cublasLtMatmulDescCreate", "cublasLtMatmulDescDestroy", 68, 0
            )
            trans = C.c_int(1)
            self.library.call(
                "cublasLtMatmulDescSetAttribute", self.operation, 4, C.byref(trans), C.sizeof(trans)
            )
            self.preference = self.new(
                "cublasLtMatmulPreferenceCreate", "cublasLtMatmulPreferenceDestroy"
            )
            for attr, value in ((1, C.c_uint64(0)), (3, C.c_uint32(2))):
                self.library.call(
                    "cublasLtMatmulPreferenceSetAttribute",
                    self.preference,
                    attr,
                    C.byref(value),
                    C.sizeof(value),
                )
            weight = self.layout(width, width)
            for n in sorted(set(rows)):
                a, out = self.layout(n, width), self.layout(n, width)
                options = (Heuristic * 8)()
                count = C.c_int()
                self.library.call(
                    "cublasLtMatmulAlgoGetHeuristic",
                    self.handle,
                    self.operation,
                    a,
                    weight,
                    out,
                    out,
                    self.preference,
                    8,
                    options,
                    C.byref(count),
                )
                chosen = None
                for option in options[: count.value]:
                    if option.state or option.workspaceSize:
                        continue
                    reduction = self.attribute(option.algo, 3, C.c_uint32, False)
                    flags = self.attribute(option.algo, 15, C.c_uint64, True)
                    if eligible_heuristic(option.state, option.workspaceSize, reduction, flags):
                        chosen = Algo.from_buffer_copy(C.string_at(C.byref(option.algo), 64))
                        record = dict(
                            rows=n,
                            workspace_bytes=0,
                            compute_type="CUBLAS_COMPUTE_32F",
                            accumulator_flags=flags,
                            reduction_scheme=reduction,
                            algorithm_id=self.attribute(chosen, 0, C.c_int, False),
                            selection="first_eligible_vendor_heuristic_no_timing_search",
                        )
                        break
                if chosen is None:
                    raise RuntimeError(f"no zero-workspace FP32-accumulator GEMM for {n} rows")
                self.plans[n] = (a, weight, out, chosen, record)
            self.alpha, self.beta = C.c_float(1.0), C.c_float(0.0)
        except BaseException:
            self.close()
            raise

    def new(self, create, destroy, *args):
        handle = C.c_void_p()
        self.library.call(create, C.byref(handle), *args)
        self.owners.append((destroy, handle))
        return handle

    def layout(self, rows, cols):
        descriptor = self.new(
            "cublasLtMatrixLayoutCreate", "cublasLtMatrixLayoutDestroy", 14, rows, cols, cols
        )
        order = C.c_int(1)
        self.library.call(
            "cublasLtMatrixLayoutSetAttribute", descriptor, 1, C.byref(order), C.sizeof(order)
        )
        return descriptor

    def attribute(self, algo, attribute, kind, cap):
        value, written = kind(), C.c_size_t()
        self.library.call(
            "cublasLtMatmulAlgo" + ("Cap" if cap else "Config") + "GetAttribute",
            C.byref(algo),
            attribute,
            C.byref(value),
            C.sizeof(value),
            C.byref(written),
        )
        if written.value != C.sizeof(value):
            raise RuntimeError("incomplete algorithm-attribute result")
        return value.value

    def execute(self, source, weight, out, rows):
        if self.closed or threading.get_ident() != self.owner_thread or rows not in self.plans:
            raise RuntimeError("live private immutable cuBLASLt plan required")
        if (
            any(
                t.device.type != "cuda"
                or t.dtype != torch.bfloat16
                or not t.is_contiguous()
                or t.data_ptr() % 256
                for t in (source, weight, out)
            )
            or source.numel() < rows * self.width
            or out.numel() < rows * self.width
            or weight.shape != (self.width, self.width)
            or len({t.untyped_storage().data_ptr() for t in (source, weight, out)}) != 3
        ):
            raise ValueError("separate aligned CUDA BF16 source/weights/output required")
        a, b, d, algo, _ = self.plans[rows]
        self.library.call(
            "cublasLtMatmul",
            self.handle,
            self.operation,
            C.byref(self.alpha),
            source.data_ptr(),
            a,
            weight.data_ptr(),
            b,
            C.byref(self.beta),
            out.data_ptr(),
            d,
            out.data_ptr(),
            d,
            C.byref(algo),
            None,
            0,
            torch.cuda.current_stream().cuda_stream,
        )

    def record(self):
        return dict(
            abi=self.library.abi,
            algorithms=[self.plans[n][-1] for n in sorted(self.plans)],
            algorithm_workspace_bytes=0,
            host_descriptors_not_GPU_KV=True,
            no_torch_matmul_precision_flag_changed=True,
        )

    def close(self):
        if self.closed:
            return
        errors = []
        for destroy, handle in reversed(self.owners):
            try:
                self.library.call(destroy, handle)
            except RuntimeError as error:
                errors.append(str(error))
        self.owners.clear()
        self.closed = True
        if errors:
            raise RuntimeError("; ".join(errors))
