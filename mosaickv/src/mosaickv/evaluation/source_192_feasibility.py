"""One bounded 192-row operator hypothesis, not a row/algorithm search.

Copy the exact existing 128-row and joint-tail algorithm descriptors and
require installed API compatibility at the new shapes. The default forbids
new corrections; a separately registered successor charges exact original
word patches within the unchanged cap. Never relax rounding or modify weights.
"""

import ctypes as C
import random
import statistics
from types import SimpleNamespace

import torch
import triton

from mosaickv.evaluation.frozen_joint_plan import FrozenKernelPlans
from mosaickv.evaluation.joint_incoming_projection import _decode_joint
from mosaickv.evaluation.joint_kernel_compatibility import fingerprint
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lt_source_chunks import _decode
from mosaickv.evaluation.zero_workspace_lt import Algo, Heuristic, eligible_heuristic

CAPACITY = 192
WIDTH = 4096


def ranges(n, capacity=CAPACITY):
    from mosaickv.evaluation.paper_geometry import source_shape_allowed

    if (
        type(n) is not int
        or not source_shape_allowed((1, 32, n, 128))
        or type(capacity) is not int
        or capacity not in (128, CAPACITY)
    ):
        raise ValueError(
            "only the two consumed source geometries and registered capacities supported"
        )
    return tuple((first, min(capacity, n - first)) for first in range(0, n, capacity))


class Plans192(FrozenKernelPlans):
    def __init__(self, n):
        ranges(n)
        old_tail = n % 128
        super().__init__((128, old_tail, old_tail + 1))
        try:
            a, weight, out, _, _ = self.plans[old_tail + 1]
            joint_algo, joint_record = self.candidates[0]
            self.plans[old_tail + 1] = (
                a,
                weight,
                out,
                joint_algo,
                dict(
                    joint_record,
                    rows=old_tail + 1,
                    workspace_bytes=0,
                    compute_type="CUBLAS_COMPUTE_32F",
                ),
            )
            # Preserve the exact original algorithm for every full and tail
            # chunk; choose no new algorithm based on timing or output words.
            for new, old in ((192, 128), (n % 192, old_tail), (n % 192 + 1, old_tail + 1)):
                self.add_original_algorithm_row(new, old)
        except BaseException:
            self.close()
            raise

    def add_original_algorithm_row(self, row, original):
        _, weight, _, old, _ = self.plans[original]
        algo = Algo.from_buffer_copy(old)
        a, out = self.layout(row, WIDTH), self.layout(row, WIDTH)
        result = Heuristic()
        self.library.call(
            "cublasLtMatmulAlgoCheck",
            self.handle,
            self.operation,
            a,
            weight,
            out,
            out,
            C.byref(algo),
            C.byref(result),
        )
        record = self.describe(algo)
        if not eligible_heuristic(
            result.state, result.workspaceSize, record["reduction"], record["numerical_flags"]
        ):
            raise RuntimeError(
                "original algorithm is not zero-workspace FP32 compatible at 192 geometry"
            )
        if fingerprint(record) != fingerprint(self.describe(old)):
            raise RuntimeError("copied original algorithm changed configuration")
        self.plans[row] = (
            a,
            weight,
            out,
            algo,
            dict(
                record,
                rows=row,
                workspace_bytes=0,
                copied_from_original_rows=original,
                selection="same_immutable_original_algorithm_no_search",
                compute_type="CUBLAS_COMPUTE_32F",
            ),
        )


def decode(packet, source, first, take, capacity, incoming=None):
    from mosaickv.evaluation.source_channel_exceptions import kernel_arguments

    n = packet.source.shape[-2]
    if (
        capacity not in (128, CAPACITY)
        or (first, take) not in ranges(n, capacity)
        or source.shape != (capacity, WIDTH)
        or source.dtype != torch.bfloat16
        or not source.is_contiguous()
        or source.device != packet.source.codes.device
        or (incoming is not None and (take == capacity or first + take != n))
    ):
        raise ValueError("explicit bounded original source chunk and tail incoming required")
    mapping, values, columns = kernel_arguments(packet)
    if incoming is None:
        compiled = _decode[(triton.cdiv(take, 16), 64)](
            packet.source.codes,
            packet.source.minimum,
            packet.source.scale,
            mapping,
            values,
            source,
            n,
            first,
            take,
            WIDTH=WIDTH,
            HAS_COLUMNS=columns,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        compiled = _decode_joint[(triton.cdiv(take + 1, 16), 64)](
            packet.source.codes,
            packet.source.minimum,
            packet.source.scale,
            mapping,
            values,
            incoming,
            source,
            n,
            FIRST=first,
            TAKE=take,
            WIDTH=WIDTH,
            HAS_COLUMNS=columns,
            num_warps=4,
            enable_fp_fusion=False,
        )
    if compiled.metadata.global_scratch_size or compiled.n_spills:
        raise RuntimeError("192-row lossless decoder introduced scratch or spills")


@torch.inference_mode()
def study(*, lossless_corrections=False):
    from mosaickv.evaluation.dual_projection_kernel import apply_patch
    from mosaickv.evaluation.exact_storage import SparseBitPatch
    from mosaickv.evaluation.precision_tier import pack
    from mosaickv.evaluation.prompt_graph_reader import allocator_blocks, generator_metadata
    from mosaickv.evaluation.shared_source_kernel import rotate_native

    if type(lossless_corrections) is not bool:
        raise ValueError("explicit separately registered numerical correction policy required")
    torch.manual_seed(20261105)
    torch.use_deterministic_algorithms(True)
    records = []
    for n in (600, 604):
        # Synthetic operator fixture only; no model/dataset or quality result.
        raw = torch.randn((1, 1, n, WIDTH), device="cuda", dtype=torch.bfloat16)
        packet = SimpleNamespace(source=pack(raw, bits=3, axis=2))
        old_tail = n % 128
        plans = Plans192(n)
        try:
            weights = tuple(
                torch.randn(WIDTH, WIDTH, device="cuda", dtype=torch.bfloat16) for _ in range(2)
            )
            incoming = torch.randn(WIDTH, device="cuda", dtype=torch.bfloat16)
            incoming[:2] = torch.tensor([0.0, -0.0], device="cuda", dtype=torch.bfloat16)
            buffers = {
                cap: (
                    torch.empty(cap, WIDTH, device="cuda", dtype=torch.bfloat16),
                    torch.empty(cap, WIDTH, device="cuda", dtype=torch.bfloat16),
                )
                for cap in (128, CAPACITY)
            }
            owners = {
                cap: tuple(
                    torch.full((1, 32, n + 127, 128), -71.0, device="cuda", dtype=torch.bfloat16)
                    for _ in range(2)
                )
                for cap in buffers
            }
            incoming_outputs = {
                cap: tuple(torch.empty_like(incoming) for _ in range(2)) for cap in buffers
            }
            phase = torch.randn(1, n, 64, device="cuda")
            phase = torch.cat((phase, phase), -1)
            cos, sin = phase.cos().bfloat16(), phase.sin().bfloat16()
            corrections = {}

            def call(
                cap,
                *,
                buffers=buffers,
                weights=weights,
                n=n,
                packet=packet,
                incoming=incoming,
                plans=plans,
                owners=owners,
                incoming_outputs=incoming_outputs,
                cos=cos,
                sin=sin,
                corrections=corrections,
            ):
                source, output = buffers[cap]
                for c, weight in enumerate(weights):
                    for first, take in ranges(n, cap):
                        tail = first + take == n
                        decode(packet, source, first, take, cap, incoming if tail else None)
                        plans.execute(source, weight, output, take + int(tail))
                        owners[cap][c][0, :, first : first + take].copy_(
                            output[:take].view(take, 32, 128).transpose(0, 1)
                        )
                        if tail:
                            incoming_outputs[cap][c].copy_(output[take])
                    rotate_native(owners[cap][c][..., :n, :], cos, sin, 0, n) if c == 0 else None
                    if cap in corrections:
                        apply_patch(corrections[cap][c], owners[cap][c][..., :n, :])

            # Old reference rows are part of the exact same private handle.
            if old_tail + 1 not in plans.plans or 128 not in plans.plans:
                raise RuntimeError("original comparison geometry unavailable")
            for cap in buffers:
                call(cap)
            torch.cuda.synchronize()
            different = [
                int((a[..., :n, :].view(torch.int16) != b[..., :n, :].view(torch.int16)).sum())
                for a, b in zip(owners[128], owners[CAPACITY], strict=True)
            ]
            incoming_different = [
                int((a.view(torch.int16) != b.view(torch.int16)).sum())
                for a, b in zip(incoming_outputs[128], incoming_outputs[CAPACITY], strict=True)
            ]
            repair_bytes = 0
            if any(different) and lossless_corrections and not any(incoming_different):
                corrections[CAPACITY] = tuple(
                    SparseBitPatch.build(a[..., :n, :], b[..., :n, :])
                    for a, b in zip(owners[128], owners[CAPACITY], strict=True)
                )
                repair_bytes = backing_bytes(
                    tuple(t for p in corrections[CAPACITY] for t in p.tensors)
                )
                if repair_bytes > 32768:
                    raise RuntimeError(
                        "192-row lossless corrections exceed unchanged 32-KiB layer cap"
                    )
                call(CAPACITY)
                torch.cuda.synchronize()
                if any(
                    not torch.equal(
                        a[..., :n, :].view(torch.int16), b[..., :n, :].view(torch.int16)
                    )
                    for a, b in zip(owners[128], owners[CAPACITY], strict=True)
                ):
                    raise RuntimeError("actual counted word corrections failed exact K/V parity")
            elif any(different + incoming_different):
                records.append(
                    dict(
                        source_length=n,
                        different_prompt_words=different,
                        different_incoming_words=incoming_different,
                        compatible=False,
                    )
                )
                return dict(
                    complete=True,
                    numerically_compatible=False,
                    records=records,
                    no_runtime_integration_authorized=True,
                    independent_quality_N=0,
                )
            if any(not bool((t[..., n:, :] == -71.0).all()) for tt in owners.values() for t in tt):
                raise RuntimeError("192-row projector overwrote unused lease capacity")
            graphs, metadata = {}, []
            for cap in buffers:
                stream = torch.cuda.Stream()
                current = torch.cuda.current_stream()
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    call(cap)
                current.wait_stream(stream)
                torch.cuda.synchronize()
                blocks, before = allocator_blocks(), torch.cuda.memory_allocated()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        call(cap)
                    finally:
                        graph.capture_end()
                current.wait_stream(stream)
                torch.cuda.synchronize()
                extra = torch.cuda.memory_allocated() - before
                if extra:
                    metadata.extend(generator_metadata(blocks, allocator_blocks(), extra))
                graphs[cap] = graph
            timings = {cap: [] for cap in buffers}
            rng = random.Random(20261105)
            try:
                for trial in range(-5, 20):
                    placement = list(buffers)
                    rng.shuffle(placement)
                    for cap in placement:
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        torch.cuda.synchronize()
                        start.record()
                        for _ in range(5):
                            graphs[cap].replay()
                        end.record()
                        torch.cuda.synchronize()
                        if trial >= 0:
                            timings[cap].append(start.elapsed_time(end) / 5)
                for cap in buffers:
                    graphs[cap].replay()
                torch.cuda.synchronize()
                if any(
                    not torch.equal(a.view(torch.int16), b.view(torch.int16))
                    for a, b in zip(owners[128], owners[CAPACITY], strict=True)
                ):
                    raise RuntimeError("graph replay changed exact projector words or sentinels")
            finally:
                for graph in graphs.values():
                    graph.reset()
                graphs.clear()
                del graph
                torch.cuda.synchronize()
            if any(p in allocator_blocks() for p in metadata):
                raise RuntimeError("operator retained generator graph ownership")
            medians = {str(cap): statistics.median(values) for cap, values in timings.items()}
            records.append(
                dict(
                    source_length=n,
                    compatible=True,
                    before_correction_prompt_word_mismatches=different,
                    different_incoming_words=incoming_different,
                    native_operator_median_ms=medians,
                    paired_operator_trials_ms=timings,
                    median_ratio_192_over_128=medians["192"] / medians["128"],
                    old_explicit_workspace_bytes=backing_bytes(buffers[128]),
                    new_explicit_workspace_bytes=backing_bytes(buffers[CAPACITY]),
                    new_additional_workspace_bytes=2**20,
                    actual_added_numerical_correction_bytes=repair_bytes,
                    lossless_corrections_explicitly_registered=lossless_corrections,
                    bitwise_after_actual_counted_correction=True,
                    old_chunks=len(ranges(n, 128)),
                    new_chunks=len(ranges(n)),
                    original_algorithm_fingerprints_preserved=True,
                    no_changes_to_phase_positions=True,
                    incoming_words_cannot_be_corrected_or_relaxed=True,
                    algorithm_workspace_bytes=0,
                    source_decoder_scratch_and_spills=0,
                )
            )
        finally:
            plans.close()
    return dict(
        complete=True,
        numerically_compatible=True,
        records=records,
        all_operator_medians_faster=all(r["median_ratio_192_over_128"] < 1 for r in records),
        no_full_request_latency_claim=True,
        independent_quality_N=0,
    )
