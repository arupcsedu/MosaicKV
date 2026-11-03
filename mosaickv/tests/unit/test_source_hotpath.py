from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import source_hotpath as policy


class Tensor:
    def __init__(self, pointer=256):
        self.pointer = pointer
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda:0")

    def data_ptr(self):
        return self.pointer


class Compiled:
    def __init__(self, kernel, scratch):
        self.kernel = kernel
        self.metadata = SimpleNamespace(global_scratch_size=scratch)

    def __getitem__(self, grid):
        def runner(*args):
            self.kernel.direct.append((grid, args))

        return runner


class Kernel:
    arg_names = tuple([f"a{i}" for i in range(13)] + ["COMPONENT", "WIDTH"])

    def __init__(self, scratch=0):
        self.calls = []
        self.direct = []
        self.compiled = Compiled(self, scratch)

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return self.compiled

        return run


def args_fixture():
    codes = Tensor()
    target = Tensor(1024)
    return (
        codes,
        Tensor(),
        Tensor(),
        codes,
        codes,
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        target,
        target,
        17,
        24 * 128,
    )


def test_prepared_runtime_destinations_and_strides_are_not_cached():
    kernel, args, attention = Kernel(), args_fixture(), object()
    options = dict(COMPONENT=0, WIDTH=512, num_warps=4)
    with policy.use("prepared") as state:
        policy.source_launch(kernel, (2, 4), args, options, attention)
        other = Tensor(2048)
        current = (*args[:9], other, other, 17, 26 * 128)
        policy.source_launch(kernel, (2, 4), current, options, attention)
        assert state["source_hits"] == state["source_misses"] == 1
        assert kernel.direct[0][1][9] is other and kernel.direct[0][1][12] == 26 * 128
        assert kernel.direct[0][1][-2:] == (0, 512)
        entry = next(iter(state["entries"].values()))
        assert not any(isinstance(x, Tensor) for x in entry[0])
    assert not state["entries"]
    with policy.use("prepared") as other:
        policy.source_launch(kernel, (2, 4), args, options, attention)
        assert other["source_misses"] == 1 and other["source_hits"] == 0


@pytest.mark.parametrize(
    "fault", ["source", "dtype", "target", "alignment", "wide_stride", "options"]
)
def test_mutated_route_or_lease_fails_closed(fault):
    kernel, args, attention = Kernel(), args_fixture(), object()
    options = dict(COMPONENT=0, WIDTH=512, num_warps=4)
    with policy.use("prepared"):
        policy.source_launch(kernel, (2, 4), args, options, attention)
        changed = list(args)
        if fault == "source":
            changed[0] = Tensor(512)
        elif fault == "dtype":
            args[9].dtype = torch.float32
        elif fault == "target":
            changed[10] = Tensor()
        elif fault == "alignment":
            args[9].pointer = 1025
        elif fault == "wide_stride":
            changed[-1] = 2**33
        else:
            options["WIDTH"] = 1024
        with pytest.raises((ValueError, RuntimeError)):
            policy.source_launch(kernel, (2, 4), tuple(changed), options, attention)
    assert not kernel.direct


def test_state_cleanup_nested_and_scratch_checks():
    for invalid in (True, None, "guess"):
        with pytest.raises(ValueError):
            policy.validate(invalid)
    with pytest.raises(RuntimeError, match="test"), policy.use("prepared") as state:
        with pytest.raises(RuntimeError, match="nonnested"), policy.use("restore"):
            pass
        policy.source_launch(
            Kernel(), (2, 4), args_fixture(), dict(COMPONENT=0, WIDTH=512), object()
        )
        raise RuntimeError("test")
    assert not state["entries"]
    with policy.use("prepared"), pytest.raises(RuntimeError, match="scratch"):
        policy.source_launch(
            Kernel(scratch=4), (2, 4), args_fixture(), dict(COMPONENT=0, WIDTH=512), object()
        )


@pytest.mark.parametrize("index", range(4))
def test_lazy_prefix_fill_keeps_all_CPU_bits_storage_and_history(index):
    from test_sequential_kv_lease import example

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    attention, formats = example()
    fmt = formats[index]
    before = backing_bytes(fmt.tensors())
    for component in (0, 1):
        original = torch.full((1, 4, 69, 128), -71, dtype=torch.bfloat16)
        candidate = torch.full_like(original, -71)
        fill_component(fmt, attention, component, original)
        with policy.use("restore") as state:
            fill_component(fmt, attention, component, candidate)
        assert torch.equal(original.view(torch.int16), candidate.view(torch.int16))
        assert backing_bytes(fmt.tensors()) == before
        assert state["fused_restore_calls"] == 0 and not state["entries"]


def test_disjoint_certificate_is_set_only_after_real_constructor_checks(monkeypatch):
    from test_shared_source_format import example

    from mosaickv.evaluation.precision_tier import unpack_cpu
    from mosaickv.evaluation.shared_source_format import (
        construct,
        fit_transport,
        transport_corrections,
    )
    from mosaickv.evaluation.text_source_storage import strip_text_overlays

    x, pair, heads, text, phases, identity, attention = example()
    packet = construct(x, pair, heads, text, phases, identity)
    assert not packet.protected_patches_disjoint
    monkeypatch.setattr(
        "mosaickv.evaluation.shared_source_kernel.decode_source_for_construction",
        lambda p: unpack_cpu(p.source).squeeze(1),
    )
    fitted, _ = fit_transport(packet, attention, componentwise=True)
    assert fitted.protected_patches_disjoint
    full_heads = (tuple(range(33)),) * 2
    stripped = strip_text_overlays(fitted, pair, full_heads, set(text))
    assert stripped.protected_patches_disjoint
    original = torch.ones(1, 1, 3, 128).bfloat16()
    damaged = original.clone()
    damaged[0, 0, 1, 0] = 5
    with pytest.raises(RuntimeError, match="protected"):
        transport_corrections((original,), (damaged,), torch.tensor([1], dtype=torch.int32))
    from mosaickv.evaluation.fused_source_restore import restore

    with pytest.raises(ValueError, match="constructor-verified"):
        restore(replace(fitted, protected_patches_disjoint=False), 0, original)


def test_fusion_is_integer_only_and_runtime_lengths_do_not_recompile():
    from mosaickv.evaluation.fused_source_restore import _restore

    params = {p.name: p for p in _restore.params}
    assert all(
        params[n].do_not_specialize and not params[n].is_constexpr
        for n in ("COUNT", "PATCHES", "N", "HEAD_OUT")
    )
    assert "tl.dot" not in _restore.src and "tl.float" not in _restore.src


def test_registered_modes_and_activity_reject_unadvertised_fallbacks():
    from evicastkv_paired_request_readers import hotpath_activity, hotpath_policy, mode_config

    plan = dict(
        paired_reader_stage="source_hotpath",
        reader_modes=["anchor_hotpath", "candidate_hotpath"],
        anchor_hotpath="original",
        candidate_hotpath="prepared_restore",
        selected_reader="carry_literals",
    )
    assert mode_config(plan, "candidate_hotpath") == ("carry_literals", False)
    assert hotpath_policy(plan, "candidate_hotpath") == "prepared_restore"
    with pytest.raises(ValueError):
        hotpath_policy(plan, "silent_fallback")
    metrics = dict(
        source_launch_hits=100,
        fused_restore_calls=10,
        fused_nonempty_calls=7,
        skipped_prefix_views=20,
    )
    assert hotpath_activity(metrics, "mosaickv_exact", "prepared_restore")
    assert not hotpath_activity(
        dict(metrics, fused_restore_calls=0), "mosaickv_exact", "prepared_restore"
    )
