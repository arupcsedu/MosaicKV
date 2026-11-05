"""Bounded numerical kernel study; never a replacement model implementation.

The saved CUPTI trace identifies a 64x64, six-stage MMA16816 projection.
That name is a hint, NOT proof of a particular public cuBLASLt algorithm.
Only actual output words decide compatibility. No benchmark labels are read.
"""

import ctypes as C
import re
from pathlib import Path

import torch

from mosaickv.evaluation import joint_row_feasibility as original
from mosaickv.evaluation.zero_workspace_lt import Algo, Heuristic, Plans, eligible_heuristic

MAX_IDS, MAX_PLANS, MAX_CUSTOM = 128, 32, 4
current = original.current
ENUMS = dict(
    CUBLASLT_MATMUL_TILE_64x64=15,
    CUBLASLT_MATMUL_STAGES_64x6=18,
    CUBLASLT_MATMUL_INNER_SHAPE_MMA16816=4,
    CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX=7,
    CUBLASLT_ALGO_CONFIG_TILE_ID=1,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM=2,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING=4,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION=5,
    CUBLASLT_ALGO_CONFIG_STAGES_ID=6,
    CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID=7,
)
CONFIG = (
    (1, C.c_uint32),
    (2, C.c_int32),
    (3, C.c_uint32),
    (4, C.c_uint32),
    (5, C.c_uint32),
    (6, C.c_uint32),
    (7, C.c_uint16),
)


def verify_header(text):
    for name, value in ENUMS.items():
        if not re.search(r"\b" + name + r"\s*=\s*" + str(value) + r"\s*[,}]", text):
            raise RuntimeError(f"unverified installed kernel-study enum {name}")


def fingerprint(record):
    return tuple(
        record[k]
        for k in (
            "algorithm_id",
            "tile",
            "split_k",
            "reduction",
            "swizzle",
            "custom",
            "stages",
            "inner",
        )
    )


def common_exact(rows):
    keys = [(r["step"], r["component"]) for r in rows]
    if len(keys) != 6 or set(keys) != {(s, c) for s in original.WINDOWS for c in (0, 1)}:
        raise ValueError("all six actual prefix/component observations required")
    sets = []
    for row in rows:
        records = row["kernels"]
        ids = [fingerprint(r["configuration"]) for r in records]
        if not records or len(ids) != len(set(ids)) or len(records) > MAX_PLANS:
            raise ValueError("nonempty bounded unique numerical configurations required")
        sets.append(
            {
                fingerprint(r["configuration"])
                for r in records
                if r["incoming_row"]["bitwise_equal"] and r["prompt_rows"]["bitwise_equal"]
            }
        )
    return sorted(set.intersection(*sets))


class KernelPlans(Plans):
    def __init__(self, rows):
        super().__init__(rows, 4096)
        try:
            header = next(p for p in self.library.abi["header_hashes"] if p.endswith("cublasLt.h"))
            verify_header(Path(header).read_text())
            lib, p, i, z = self.library.lib, C.c_void_p, C.c_int, C.c_size_t
            signatures = {
                "cublasLtMatmulAlgoGetIds": [p, i, i, i, i, i, i, i, C.POINTER(i), C.POINTER(i)],
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
                getattr(lib, name).argtypes, getattr(lib, name).restype = types, i
            self.candidates, self.rejections = [], []
            # Unmodified vendor choices first, then explicit geometry-matched choices.
            a, weight, out, _, _ = self.plans[max(rows)]
            options, count = (Heuristic * 32)(), i()
            self.library.call(
                "cublasLtMatmulAlgoGetHeuristic",
                self.handle,
                self.operation,
                a,
                weight,
                out,
                out,
                self.preference,
                32,
                options,
                C.byref(count),
            )
            for option in options[: min(count.value, 6)]:
                self.consider(option.algo, "unmodified_vendor_heuristic")
            ids, count = (i * MAX_IDS)(), i()
            self.library.call(
                "cublasLtMatmulAlgoGetIds",
                self.handle,
                68,
                0,
                14,
                14,
                14,
                14,
                MAX_IDS,
                ids,
                C.byref(count),
            )
            self.ids = list(ids[: count.value])
            for identifier in self.ids:
                if len(self.candidates) >= MAX_PLANS:
                    break
                base = Algo()
                code = lib.cublasLtMatmulAlgoInit(
                    self.handle, 68, 0, 14, 14, 14, 14, identifier, C.byref(base)
                )
                if code:
                    self.rejections.append(dict(algorithm_id=identifier, init_status=code))
                    continue
                maximum = self.attribute(base, 7, C.c_int32, True)
                for custom in range(min(maximum, MAX_CUSTOM) + 1):
                    if len(self.candidates) >= MAX_PLANS:
                        break
                    algo = Algo.from_buffer_copy(base)
                    status = []
                    for attr, value, kind in (
                        (1, 15, C.c_uint32),
                        (2, 1, C.c_int32),
                        (3, 0, C.c_uint32),
                        (4, 0, C.c_uint32),
                        (5, custom, C.c_uint32),
                        (6, 18, C.c_uint32),
                        (7, 4, C.c_uint16),
                    ):
                        scalar = kind(value)
                        status.append(
                            lib.cublasLtMatmulAlgoConfigSetAttribute(
                                C.byref(algo), attr, C.byref(scalar), C.sizeof(scalar)
                            )
                        )
                    if any(status):
                        self.rejections.append(
                            dict(algorithm_id=identifier, custom=custom, set_statuses=status)
                        )
                    else:
                        self.consider(algo, "trace_geometry_hint_not_claimed_kernel_identity")
            if not self.candidates:
                raise RuntimeError("no executable zero-workspace kernel configurations")
        except BaseException:
            self.close()
            raise

    def describe(self, algo):
        values = [self.attribute(algo, attr, kind, False) for attr, kind in CONFIG]
        return dict(
            zip(
                ("tile", "split_k", "reduction", "swizzle", "custom", "stages", "inner"),
                values,
                strict=True,
            ),
            algorithm_id=self.attribute(algo, 0, C.c_int32, False),
            numerical_flags=self.attribute(algo, 15, C.c_uint64, True),
        )

    def consider(self, algo, origin):
        record = self.describe(algo)
        if any(fingerprint(record) == fingerprint(r) for _, r in self.candidates):
            return
        checks = []
        for n in sorted(self.plans):
            a, weight, out, _, _ = self.plans[n]
            result = Heuristic()
            code = self.library.lib.cublasLtMatmulAlgoCheck(
                self.handle, self.operation, a, weight, out, out, C.byref(algo), C.byref(result)
            )
            checks.append(
                dict(
                    rows=n,
                    status=code,
                    workspace_bytes=result.workspaceSize,
                    result_state=result.state,
                )
            )
        if any(
            r["status"]
            or not eligible_heuristic(
                r["result_state"],
                r["workspace_bytes"],
                record["reduction"],
                record["numerical_flags"],
            )
            for r in checks
        ):
            self.rejections.append(dict(configuration=record, checks=checks))
            return
        self.candidates.append((Algo.from_buffer_copy(algo), dict(record, origin=origin)))

    def run_configuration(self, source, weight, output, n, algo, record):
        a, b, d, original_algo, original_record = self.plans[n]
        try:
            self.plans[n] = a, b, d, algo, record
            self.execute(source, weight, output, n)
        finally:
            self.plans[n] = a, b, d, original_algo, original_record


class Probe(original.Probe):
    def observe(self, component, module, hidden, output):
        if self.index not in original.WINDOWS or tuple(hidden.shape) != (1, 1, 4096):
            return
        if self.source is None or module.bias is not None or hidden.dtype != torch.bfloat16:
            raise RuntimeError("original bias-free pinned BF16 projection required")
        if not isinstance(self.plans, KernelPlans):
            self.plans.close()
            self.plans = KernelPlans((self.layout["prompt_rows"], self.layout["joint_rows"]))
        n = self.layout["prompt_rows"]
        self.source[n].copy_(hidden.reshape(-1))
        records = []
        for algo, configuration in self.plans.candidates:
            self.plans.run_configuration(
                self.source, module.weight, self.output, n + 1, algo, configuration
            )
            incoming = original.comparison(self.output[n], output.reshape(-1))
            prompt = self.output[:n].clone()
            self.plans.run_configuration(
                self.source, module.weight, self.output, n, algo, configuration
            )
            records.append(
                dict(
                    configuration=configuration,
                    incoming_row=incoming,
                    prompt_rows=original.comparison(prompt, self.output[:n]),
                )
            )
        self.rows.append(dict(step=self.index, component=component, kernels=records))


def collect(adapter, call, folder):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    with Probe(adapter).installed() as state:
        measured = call()
        if (
            state.steps != 127
            or not measured["logits_bitwise"]
            or not measured["no_FullKV_fallback"]
        ):
            raise RuntimeError("complete unchanged 128-vector parent trajectory required")
        return dict(
            comparisons=state.rows,
            common_exact_configurations=common_exact(state.rows),
            layout=state.layout,
            source_model_unchanged=True,
            parent_logits_bitwise=True,
            no_FullKV_fallback=True,
            vendor_ids_considered=state.plans.ids,
            bounded_rejected_configurations=state.plans.rejections,
            maximum_executed_configurations=MAX_PLANS,
            explicit_diagnostic_fixed_workspace_bytes=backing_bytes((state.source, state.output)),
            vendor_algorithm_workspace_bytes=0,
            whole_request_GPU_peak_certified=False,
            native_ABI=state.plans.record(),
            registered_enums=ENUMS,
            candidate_forward_not_integrated=True,
            no_quality_or_latency_claim=True,
            independent_quality_N=0,
        )
