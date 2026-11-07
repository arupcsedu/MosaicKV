"""128-row source + output buffers; zero-workspace FP32-accumulating cuBLASLt."""

import time

import torch
import triton
import triton.language as tl

from mosaickv.evaluation.shared_source_kernel import _read_coherent


@triton.jit(do_not_specialize=["ROWS", "FIRST", "TAKE"])
def _decode(
    CODES,
    MINIMUM,
    SCALE,
    MAP,
    VALUES,
    OUT,
    ROWS,
    FIRST,
    TAKE,
    ROW_MAP,
    WIDTH: tl.constexpr,
    HAS_COLUMNS: tl.constexpr,
    SOURCE_ROWS: tl.constexpr = 0,
    HAS_ROW_MAP: tl.constexpr = False,
):
    local = tl.program_id(0) * 16 + tl.arange(0, 16)
    start = tl.program_id(1) * 64
    x = _read_coherent(
        CODES,
        MINIMUM,
        SCALE,
        MAP,
        VALUES,
        FIRST + local,
        start,
        ROWS,
        WIDTH,
        3,
        HAS_COLUMNS,
        False,
        False,
        ROW_MAP,
        SOURCE_ROWS,
        HAS_ROW_MAP,
    )
    offset = local[:, None] * WIDTH + start + tl.arange(0, 64)[None, :]
    tl.store(OUT + offset, x, local[:, None] < TAKE)


def decode_slice(packet, target, first, take):
    from mosaickv.evaluation.source_channel_exceptions import kernel_arguments

    n, width = packet.source.shape[2:]
    if (
        type(first) is not int
        or type(take) is not int
        or not 0 <= first < first + take <= n
        or take > 128
        or target.shape != (128, width)
        or target.dtype != torch.bfloat16
        or not target.is_contiguous()
        or target.device != packet.source.codes.device
    ):
        raise ValueError("bounded private 128-row original source chunk required")
    mapping, values, columns = kernel_arguments(packet)
    from mosaickv.evaluation.source_replacement import arguments as source_arguments

    row_map, source_rows, has_row_map = source_arguments(packet.source)
    compiled = _decode[(triton.cdiv(take, 16), width // 64)](
        packet.source.codes,
        packet.source.minimum,
        packet.source.scale,
        mapping,
        values,
        target,
        n,
        first,
        take,
        row_map,
        WIDTH=width,
        HAS_COLUMNS=columns,
        SOURCE_ROWS=source_rows,
        HAS_ROW_MAP=has_row_map,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if compiled.metadata.global_scratch_size != 0 or compiled.n_spills != 0:
        raise RuntimeError("source chunk decoder introduced scratch or spills")


class Engine:
    def __init__(self, packet, *, plans_factory=None):
        from mosaickv.evaluation.zero_workspace_lt import Plans

        started = time.perf_counter()
        _, heads, n, dim = packet.shape
        if dim != 128 or heads * dim != 4096:
            raise ValueError("registered LLaVA 4096-wide geometry required")
        rows = tuple(min(128, n - first) for first in range(0, n, 128))
        self.plans = Plans(rows, 4096) if plans_factory is None else plans_factory(rows, 4096)
        try:
            self.source = torch.empty(
                (128, 4096), device=packet.source.codes.device, dtype=torch.bfloat16
            )
            self.output = torch.empty_like(self.source)
        except BaseException:
            self.plans.close()
            raise
        self.closed = False
        self.project_calls = self.matmul_calls = 0
        self.construction_wall_seconds = time.perf_counter() - started

    @property
    def workspace_bytes(self):
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

        return backing_bytes((self.source, self.output))

    def check_source(self, packet, reference):
        for first in range(0, packet.shape[2], 128):
            take = min(128, packet.shape[2] - first)
            self.source.view(torch.int16).fill_(32767)
            decode_slice(packet, self.source, first, take)
            if not torch.equal(
                self.source[:take].view(torch.int16),
                reference[0, first : first + take].view(torch.int16),
            ):
                raise RuntimeError("bounded Lt source decoder changed an original source word")

    def project(self, packet, attention, component, target, *, inspect_kernel=False):
        from mosaickv.evaluation.repaired_source_tile import validate
        from mosaickv.evaluation.shared_source_kernel import rotate_native

        if self.closed:
            raise RuntimeError("closed request-private Lt reader")
        validate(packet, attention, component, target)
        _, heads, n, dim = packet.shape
        weight = (attention.k_proj, attention.v_proj)[component].weight
        for first in range(0, n, 128):
            take = min(128, n - first)
            decode_slice(packet, self.source, first, take)
            self.plans.execute(self.source, weight, self.output, take)
            target[0, :, first : first + take, :].copy_(
                self.output[:take].view(take, heads, dim).transpose(0, 1)
            )
            self.matmul_calls += 1
        if component == 0:
            rotate_native(target, packet.cos, packet.sin, 0, n)
        self.project_calls += 1
        if inspect_kernel:
            return dict(
                vendor_GEMM_registers="not_exposed",
                vendor_GEMM_spills="not_exposed",
                algorithm_workspace_bytes=0,
                explicit_workspace_bytes=self.workspace_bytes,
                source_decoder_scratch_and_spills_checked=True,
            )

    def record(self):
        return dict(
            self.plans.record(),
            workspace_bytes=self.workspace_bytes,
            descriptor_and_buffer_construction_wall_seconds=self.construction_wall_seconds,
            project_calls=self.project_calls,
            matmul_calls=self.matmul_calls,
            two_distinct_128_row_BF16_owners=True,
            vendor_internal_resources_not_whole_GPU_peak_evidence=True,
        )

    def close(self):
        if not self.closed:
            torch.cuda.synchronize()
            self.plans.close()
            self.closed = True
