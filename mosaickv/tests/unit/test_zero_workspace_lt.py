import ctypes as C
import threading
from types import SimpleNamespace

import pytest
import torch
from test_prompt_source_subphases import projection_fixture

from mosaickv.evaluation import lt_source_chunks as chunks
from mosaickv.evaluation import repaired_tile_operator as operator
from mosaickv.evaluation import zero_workspace_lt as lt


class FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def invoke(*args, **options):
            self.calls.append((grid, args, options))
            return SimpleNamespace(n_spills=0, metadata=SimpleNamespace(global_scratch_size=0))

        return invoke


def test_installed_ABI_proof_never_queries_a_GPU(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("GPU queried"))
    proof = lt.abi_proof()
    assert (C.sizeof(lt.Algo), C.sizeof(lt.Heuristic)) == (64, 96)
    assert proof["enums_verified"]["CUBLAS_COMPUTE_32F"] == 68
    assert len(proof["header_hashes"]) == 3
    assert proof["no_package_installed_or_modified"]


@pytest.mark.parametrize(
    "state,workspace,reduction,flags,accepted",
    [
        (0, 0, 0, 512, True),
        (0, 0, 2, 513, True),
        (1, 0, 0, 512, False),
        (0, 8, 0, 512, False),
        (0, 0, 4, 512, False),
        (0, 0, 1, 512, False),
        (0, 0, 0, 256, False),
        (False, 0, 0, 512, False),
    ],
)
def test_rejects_nonzero_workspace_and_non_FP32_or_output_type_reduction(
    state, workspace, reduction, flags, accepted
):
    assert lt.eligible_heuristic(state, workspace, reduction, flags) is accepted


@pytest.mark.parametrize(
    "rows,width",
    [([], 4096), ((), 4096), ((True,), 4096), ((129,), 4096), ((0,), 4096), ((128,), 2048)],
)
def test_plan_rejects_unregistered_geometry_before_GPU_or_library_access(rows, width):
    with pytest.raises(ValueError):
        lt.Plans(rows, width)


def test_layout_descriptor_cleanup_is_reverse_order_idempotent_and_fail_closed():
    calls = []
    obj = lt.Plans.__new__(lt.Plans)
    obj.owners = [("handle", 1), ("descriptor", 2)]
    obj.closed = False
    obj.library = SimpleNamespace(call=lambda *args: calls.append(args))
    obj.close()
    obj.close()
    assert calls == [("descriptor", 2), ("handle", 1)] and obj.closed


def test_full_ABI_dispatch_uses_FP32_zero_workspace_and_original_weight_pointer(monkeypatch):
    calls, attrs = [], []

    class FakeLibrary:
        def __init__(self):
            self.abi = {}

        def call(self, name, *args):
            calls.append((name, args))
            if name.endswith("Create"):
                C.cast(args[0], C.POINTER(C.c_void_p))[0] = len(calls) * 256
            elif name.endswith("SetAttribute"):
                kind = (
                    C.c_uint64
                    if name == "cublasLtMatmulPreferenceSetAttribute" and args[1] == 1
                    else C.c_uint32
                )
                attrs.append((name, args[1], C.cast(args[2], C.POINTER(kind))[0]))
            elif name == "cublasLtMatmulAlgoGetHeuristic":
                args[-2][0].workspaceSize = 16
                args[-2][1].workspaceSize = 0
                args[-2][1].algo.data[0] = 7
                C.cast(args[-1], C.POINTER(C.c_int))[0] = 2
            elif name.endswith("GetAttribute"):
                kind = C.c_uint64 if "Cap" in name else C.c_uint32
                value = 512 if "Cap" in name else (7 if args[1] == 0 else 0)
                C.cast(args[2], C.POINTER(kind))[0] = value
                C.cast(args[-1], C.POINTER(C.c_size_t))[0] = C.sizeof(kind)

    monkeypatch.setattr(lt, "library", lambda: FakeLibrary())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0x1234))

    class Tensor:
        dtype = torch.bfloat16
        device = SimpleNamespace(type="cuda")
        shape = (4096, 4096)

        def __init__(self, pointer):
            self.pointer = pointer

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return self.pointer

        def untyped_storage(self):
            return self

        def numel(self):
            return 4096 * 4096

    plans = lt.Plans((128, 88), 4096)
    plans.execute(Tensor(256), Tensor(512), Tensor(768), 88)
    name, args = calls[-1]
    assert name == "cublasLtMatmul" and len(args) == 16
    assert args[3] == 256 and args[5] == 512 and args[10] == 768
    assert args[-3:] == (None, 0, 0x1234)
    assert ("cublasLtMatmulPreferenceSetAttribute", 1, 0) in attrs
    assert ("cublasLtMatmulPreferenceSetAttribute", 3, 2) in attrs
    assert all(
        r["algorithm_id"] == 7 and r["workspace_bytes"] == 0 for r in plans.record()["algorithms"]
    )
    plans.close()


def test_closed_and_cross_thread_requests_cannot_dispatch_native_work():
    obj = lt.Plans.__new__(lt.Plans)
    obj.closed = True
    with pytest.raises(RuntimeError):
        obj.execute(None, None, None, 128)
    obj.closed, obj.owner_thread, obj.plans = False, threading.get_ident() + 1, {128: None}
    with pytest.raises(RuntimeError):
        obj.execute(None, None, None, 128)


def test_decode_window_preserves_original_codes_headers_and_positions(monkeypatch):
    packet, _, _ = projection_fixture()
    target = torch.empty(128, 256, dtype=torch.bfloat16)
    kernel = FakeKernel()
    monkeypatch.setattr(chunks, "_decode", kernel)
    chunks.decode_slice(packet, target, 8, 8)
    grid, args, options = kernel.calls[0]
    assert grid == (1, 4) and args[0] is packet.source.codes
    assert args[1] is packet.source.minimum and args[2] is packet.source.scale
    assert args[5] is target and args[-3:] == (16, 8, 8)
    assert options["WIDTH"] == 256 and options["enable_fp_fusion"] is False
    with pytest.raises(ValueError):
        chunks.decode_slice(packet, target, 8, 129)


def test_two_private_buffers_are_fully_charged_and_not_shared_between_requests(monkeypatch):
    class FakePlans:
        def __init__(self, *args):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(lt, "Plans", FakePlans)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    packet = SimpleNamespace(shape=(1, 32, 17, 128), source=SimpleNamespace(codes=torch.empty(1)))
    a, b = chunks.Engine(packet), chunks.Engine(packet)
    assert a.workspace_bytes == b.workspace_bytes == 2097152
    assert (
        len({x.untyped_storage().data_ptr() for x in (a.source, a.output, b.source, b.output)}) == 4
    )
    a.close()
    a.close()
    assert a.plans.closed and not b.plans.closed
    with pytest.raises(RuntimeError):
        a.project(None, None, 0, None)
    b.close()


def test_failed_operator_retires_private_descriptors_without_swallowing_error(monkeypatch):
    calls = []

    def failed(index, name, owners):
        owners.append(SimpleNamespace(close=lambda: calls.append("closed")))
        raise RuntimeError("observed failure")

    monkeypatch.setattr(operator, "_run", failed)
    with pytest.raises(RuntimeError, match="observed failure"):
        operator.run(0, engine_name="lt128")
    assert calls == ["closed"]
