"""Instantiate the already verified joint configuration, not its old search.

No CUDA handle/tensor/request is cached. Immutable installed-header proof may
be reused, just like the base library ABI proof. Every new private handle still
checks the exact frozen configuration and zero-workspace feasibility.
"""

import ctypes as C
from functools import lru_cache
from pathlib import Path

import torch

from mosaickv.evaluation.fox_campaign import sha
from mosaickv.evaluation.joint_all_layer_compatibility import FROZEN
from mosaickv.evaluation.joint_kernel_compatibility import (
    CONFIG,
    KernelPlans,
    fingerprint,
    verify_header,
)
from mosaickv.evaluation.zero_workspace_lt import Algo, Heuristic, Plans


@lru_cache(maxsize=4)
def header_proof(path, expected):
    if sha(Path(path)) != expected:
        raise RuntimeError("installed joint header differs from the loaded library ABI proof")
    verify_header(Path(path).read_text())
    return path, expected


def frozen_attributes():
    # FROZEN order is algorithm, tile, split-k, reduction, swizzle, custom,
    # stages, inner. Attribute widths are the installed ABI, not Python ints.
    return tuple(
        (attr, value, kind) for (attr, kind), value in zip(CONFIG, FROZEN[1:], strict=True)
    )


class FrozenKernelPlans(KernelPlans):
    def __init__(self, rows, width=4096):
        if (
            type(rows) is not tuple
            or not rows
            or any(type(n) is not int for n in rows)
            or type(width) is not int
            or width != 4096
        ):
            raise ValueError("explicit full-width integer frozen geometry required")
        if set(rows) not in ({88, 89, 128}, {92, 93, 128}):
            raise NotImplementedError(
                "only the two all-layer-audited frozen geometries are enabled"
            )
        Plans.__init__(self, rows, width)
        try:
            header = next(p for p in self.library.abi["header_hashes"] if p.endswith("cublasLt.h"))
            header_proof(header, self.library.abi["header_hashes"][header])
            p, i, z = C.c_void_p, C.c_int, C.c_size_t
            signatures = {
                "cublasLtMatmulAlgoInit": [p, i, i, i, i, i, i, i, C.POINTER(Algo)],
                "cublasLtMatmulAlgoConfigSetAttribute": [C.POINTER(Algo), i, p, z],
                "cublasLtMatmulAlgoCheck": [
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    C.POINTER(Algo),
                    C.POINTER(Heuristic),
                ],
            }
            for name, types in signatures.items():
                getattr(self.library.lib, name).argtypes = types
                getattr(self.library.lib, name).restype = i
            self.candidates, self.rejections, self.ids = [], [], [FROZEN[0]]
            algo = Algo()
            self.library.call(
                "cublasLtMatmulAlgoInit",
                self.handle,
                68,
                0,
                14,
                14,
                14,
                14,
                FROZEN[0],
                C.byref(algo),
            )
            for attr, value, kind in frozen_attributes():
                scalar = kind(value)
                self.library.call(
                    "cublasLtMatmulAlgoConfigSetAttribute",
                    C.byref(algo),
                    attr,
                    C.byref(scalar),
                    C.sizeof(scalar),
                )
            self.consider(algo, "previously_verified_frozen_joint_configuration")
            if len(self.candidates) != 1 or fingerprint(self.candidates[0][1]) != FROZEN:
                raise RuntimeError("the exact frozen joint configuration is unavailable")
        except BaseException:
            self.close()
            raise


@torch.inference_mode()
def gpu_controls():
    """Compare actual default/tail joint words, not opaque descriptor bytes."""
    import torch

    torch.manual_seed(20261105)
    source = torch.randn((128, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    before, after = torch.empty_like(source), torch.empty_like(source)
    records = []
    for tail in (88, 92):
        old, new = KernelPlans((128, tail, tail + 1)), None
        try:
            new = FrozenKernelPlans((128, tail, tail + 1))
            selected = [(a, r) for a, r in old.candidates if fingerprint(r) == FROZEN]
            if len(selected) != 1:
                raise RuntimeError("the predecessor configuration must still be available")
            for plans, choice in ((old, selected[0]), (new, new.candidates[0])):
                a, b, d, _, _ = plans.plans[tail + 1]
                algo, record = choice
                plans.plans[tail + 1] = a, b, d, algo, record
            for n in (tail, tail + 1, 128):
                if fingerprint(old.describe(old.plans[n][3])) != fingerprint(
                    new.describe(new.plans[n][3])
                ):
                    raise RuntimeError("an original full/tail algorithm silently changed")
                old.execute(source, weight, before, n)
                new.execute(source, weight, after, n)
                torch.cuda.synchronize()
                different = int((before[:n].view(torch.int16) != after[:n].view(torch.int16)).sum())
                records.append(
                    dict(
                        tail=tail,
                        rows=n,
                        differing_words=different,
                        configuration=fingerprint(new.describe(new.plans[n][3])),
                    )
                )
                if different:
                    raise RuntimeError("frozen construction changed an original GEMM word")
        finally:
            old.close()
            if new is not None:
                new.close()
    return dict(
        complete=True,
        records=records,
        no_CUDA_handle_or_tensor_cache=True,
        zero_algorithm_workspace=True,
        no_checkpoint_or_benchmark_inputs=True,
    )
