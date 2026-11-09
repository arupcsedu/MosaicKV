import gc
import weakref
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.fused_history_unpack import bind, launch
from mosaickv.evaluation.history_source_precision import pack_source
from mosaickv.evaluation.lossless_bf16 import pack_cpu
from mosaickv.evaluation.mixed_history_projection import unpack_sources


@dataclass
class Packet:
    shape: tuple
    recent: tuple
    older: object = None


def fixture(*, raw=False, old=0, group=32, count=4):
    packets = tuple(
        SimpleNamespace(
            shape=(1, 32, 16, 128),
            archive=SimpleNamespace(
                source=pack_cpu(torch.randn(1, 1, 16, 4096).bfloat16(), raw_fallback=False)
            ),
        )
        for _ in range(count)
    )
    if raw:
        for p in packets:
            source = torch.randn(1, 1, 16, 4096).bfloat16()
            p.archive.source = replace(p.archive.source, raw=True, buffers=(source,))
    older = (
        SimpleNamespace(
            body=pack_source(torch.randn(1, 1, old, 4096).bfloat16(), group=group),
            shape=(1, 32, old, 128),
        )
        if old
        else None
    )
    return Packet((1, 32, old + count * 16, 128), packets, older)


@pytest.mark.parametrize(
    "raw,old,group,count",
    [
        (False, 0, 32, 1),
        (True, 0, 32, 4),
        (False, 16, 32, 4),
        (True, 32, 128, 4),
        (False, 48, 32, 4),
    ],
)
def test_complete_geometry_and_reference_words(raw, old, group, count):
    packet = fixture(raw=raw, old=old, group=group, count=count)
    target = torch.full((135, 4096), -91, dtype=torch.bfloat16)
    entry = bind(packet, target, device_type="cpu")
    assert len(entry.arguments()) == 24
    assert entry.grid == ((old + count * 16) * 4,)
    assert entry.geometry["OLD_ROWS"] == old
    assert entry.arguments()[-1] is target
    unpack_sources(packet, target)
    assert bool((target[packet.shape[-2] :] == -91).all())
    assert all(isinstance(pointer, weakref.ReferenceType) for pointer in entry.pointers)


def test_binding_retains_no_tensor_owner_and_expires():
    packet = fixture(count=1)
    target = torch.empty(128, 4096, dtype=torch.bfloat16)
    source = weakref.ref(packet.recent[0].archive.source.buffers[0])
    dest = weakref.ref(target)
    entry = bind(packet, target, device_type="cpu")
    del packet, target
    gc.collect()
    assert source() is None and dest() is None
    with pytest.raises(RuntimeError, match="expired"):
        entry.arguments()


@pytest.mark.parametrize(
    "case",
    [
        "recent_count",
        "logical_length",
        "short_workspace",
        "stride",
        "shared_literals",
        "offset_dtype",
        "raw_dtype",
        "older_rows",
        "older_group",
        "older_codebook",
    ],
)
def test_incomplete_or_unsupported_sources_fail_closed(case):
    packet = fixture(old=16)
    target = torch.empty(128, 4096, dtype=torch.bfloat16)
    if case == "recent_count":
        packet.recent = ()
    elif case == "logical_length":
        packet.shape = (1, 32, 81, 128)
    elif case == "short_workspace":
        target = target[:16]
    elif case == "stride":
        target = torch.empty(128, 8192, dtype=torch.bfloat16)[:, ::2]
    elif case == "shared_literals":
        p = packet.recent[0]
        p.archive.source = replace(p.archive.source, shared_literal_window=True)
    elif case == "offset_dtype":
        p = packet.recent[0]
        buffers = list(p.archive.source.buffers)
        buffers[3] = buffers[3].long()
        p.archive.source = replace(p.archive.source, buffers=tuple(buffers))
    elif case == "raw_dtype":
        p = packet.recent[0]
        p.archive.source = replace(p.archive.source, raw=True, buffers=(torch.empty(65536),))
    elif case == "older_rows":
        packet.older.body = replace(packet.older.body, shape=(1, 1, 17, 4096))
    elif case == "older_group":
        packet.older.body = replace(packet.older.body, group=64)
    else:
        packet.older.body = replace(packet.older.body, codebook=torch.empty(4))
    with pytest.raises(ValueError):
        bind(packet, target, device_type="cpu")


def test_cpu_is_never_reported_as_fused_kernel_execution():
    packet = fixture(count=1)
    with pytest.raises(RuntimeError, match="CUDA"):
        launch(packet, torch.empty(128, 4096, dtype=torch.bfloat16))
