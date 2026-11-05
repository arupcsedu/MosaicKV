from dataclasses import replace

import pytest
import torch
from test_full_prompt_source import example

from mosaickv.evaluation import lt_source_format as native
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def fixture():
    attention, base = example(9)
    for patch in base.transport_patches:
        assert not torch.isin(patch.indices // 128, base.row_positions).any()
    return attention, replace(base, protected_patches_disjoint=True)


class Engine:
    """CPU arithmetic oracle for unit tests, not a measured GPU implementation."""

    def __init__(self, packet):
        self.closed = False
        self.calls = 0

    def project(self, base, attention, component, target):
        from mosaickv.evaluation.sequential_kv_lease import fill_component

        assert not self.closed
        fill_component(base, attention, component, target)
        positions = [
            i for i in range(base.shape[1] * base.shape[2]) if i not in base.row_positions.tolist()
        ]
        if positions:
            row = positions[0]
            target[0, row // base.shape[2], row % base.shape[2], 7] += 1.0
        self.calls += 1

    def record(self):
        return dict(mock_CPU_only=True, calls=self.calls)

    def close(self):
        self.closed = True


def test_fixed_repairs_preserve_every_parent_word_and_padded_history():
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    attention, base = fixture()
    original = tuple(t.clone() for t in base.tensors())
    engine = Engine(base)
    value = native.fit(base, attention, engine)
    assert value.base is base and native.source_format(value) is value
    assert backing_bytes(value.private()) >= backing_bytes(base.private())
    with native.use(engine):
        for tail in (0, 1, 64, 127):
            for component in (0, 1):
                a = torch.full((1, 2, 9 + tail, 128), -71.0, dtype=torch.bfloat16)
                b = torch.full_like(a, -71.0)
                fill_component(base, attention, component, a)
                fill_component(value, attention, component, b)
                assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(original, base.tensors(), strict=True)
    )
    with pytest.raises(RuntimeError, match="bound private"):
        value.fill_component(attention, 0, torch.empty(base.shape, dtype=torch.bfloat16))


def test_conversion_retires_its_engine_in_success_and_failure(monkeypatch):
    attention, base = fixture()
    owners = []

    def factory(packet):
        owners.append(Engine(packet))
        return owners[-1]

    values, info = native.convert((base,), (attention,), engine_factory=factory)
    assert info["eligible_layers"] == 1 and info["parent_KV_bitwise"]
    assert owners[0].closed and values[0].base is base
    monkeypatch.setattr(native, "fit", lambda *a: (_ for _ in ()).throw(RuntimeError("failure")))
    with pytest.raises(RuntimeError, match="failure"):
        native.convert((base,), (attention,), engine_factory=factory)
    assert owners[-1].closed


def test_incompatible_baseline_is_truthfully_unchanged_and_allocates_nothing():
    baseline = object()

    def forbidden(*args):
        pytest.fail("incompatible baseline allocated a vendor plan")

    packets, info = native.convert((baseline,), (object(),), engine_factory=forbidden)
    assert packets[0] is baseline and info["eligible_layers"] == 0
    assert info["engine"] is None and info["added_patch_bytes"] == 0


def test_replay_rebinds_only_original_source_bodies_and_keeps_exact_patch_objects():
    from mosaickv.evaluation.layer_code_delta import map_bodies

    attention, base = fixture()
    value = native.fit(base, attention, Engine(base))
    seen = []

    def replace_codes(role, body):
        seen.append(role)
        return replace(body, codes=body.codes.clone())

    replay = map_bodies(value, replace_codes)
    assert seen == ["source"] and replay.repairs is value.repairs
    assert replay.base.source.minimum is value.base.source.minimum
    assert replay.base.row_positions is value.base.row_positions
    assert replay.base.source.codes.data_ptr() != value.base.source.codes.data_ptr()


def test_private_engine_context_is_nonnested_and_exception_safe():
    engine = object()
    with pytest.raises(RuntimeError, match="abort"), native.use(engine):
        with pytest.raises(RuntimeError, match="nonnested"), native.use(engine):
            pass
        raise RuntimeError("abort")
    with native.use(engine):
        pass


def test_runtime_requires_explicit_graph_reader_before_prefill_or_model_access():
    from mosaickv.evaluation.common_streaming_timing import request

    for value in (True, 1, "true"):
        with pytest.raises(ValueError, match="private graph"):
            request(None, None, None, "mosaickv_exact", lt_source_reader=value)
