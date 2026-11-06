from dataclasses import replace

import pytest
import torch
from test_lt_source_format import fixture

from mosaickv.evaluation import verified_constructor as mod
from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.sequential_kv_lease import fill_component


def corrected_fixture():
    from mosaickv.evaluation.exact_storage import decoded, encode

    attention, base = fixture()
    base = replace(
        base,
        row_positions=base.row_positions[1:].clone(),
        exact=tuple(encode(decoded(body)[:, :, 1:, :].contiguous()) for body in base.exact),
    )
    unprotected = [
        i for i in range(base.shape[1] * base.shape[2]) if i not in base.row_positions.tolist()
    ]
    assert unprotected
    patches = tuple(
        SparseBitPatch(
            torch.tensor([unprotected[0] * 128 + 7], dtype=torch.int32),
            torch.tensor([1.0], dtype=torch.bfloat16),
        )
        for _ in range(2)
    )
    return attention, replace(base, transport_patches=patches)


@pytest.mark.parametrize("audit", [False, True])
def test_same_buffer_correction_matches_original_full_second_projection(audit):
    attention, base = corrected_fixture()
    bare = replace(base, transport_patches=())
    for component in (0, 1):
        actual, expected = (torch.empty(base.shape, dtype=torch.bfloat16) for _ in range(2))
        fill_component(bare, attention, component, actual)
        fill_component(base, attention, component, expected)
        patch = base.transport_patches[component]
        output = mod.correct_same_buffer(bare, attention, component, patch, actual, audit=audit)
        assert output is actual
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


def test_same_buffer_audit_rejects_changed_second_projection(monkeypatch):
    attention, base = fixture()
    actual = torch.empty(base.shape, dtype=torch.bfloat16)
    fill_component(base, attention, 0, actual)
    from mosaickv.evaluation import sequential_kv_lease

    monkeypatch.setattr(sequential_kv_lease, "fill_component", lambda *args: args[-1].zero_())
    empty = SparseBitPatch(torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="second projection"):
        mod.correct_same_buffer(base, attention, 0, empty, actual, audit=True)


def test_invalid_or_nested_scope_cannot_silently_enable_a_new_constructor():
    @mod.managed
    def call(*args, **kwargs):
        return None, None, {}

    assert call(None, None, None, "prefixkv_reimpl")[2]["verified_constructor"]["eligible"] is False
    for mode in (True, 1, "other"):
        with pytest.raises(ValueError, match="explicit"):
            call(None, None, None, "mosaickv_exact", verified_constructor=mode)
    with pytest.raises(NotImplementedError):
        call(None, None, None, "mosaickv_exact", verified_constructor="handoff")
    token = mod._scope.set(object())
    try:
        with pytest.raises(ValueError, match="nest"):
            call(None, None, None, "prefixkv_reimpl")
    finally:
        mod._scope.reset(token)


def test_handoff_keeps_compact_text_not_a_full_reference_owner():
    attention, base = corrected_fixture()
    verified = torch.empty(base.shape, dtype=torch.bfloat16)
    fill_component(base, attention, 0, verified)

    class Engine:
        def project(self, packet, attention, component, target):
            fill_component(packet, attention, component, target)

        def close(self):
            pass

    state = mod.State("handoff", [("resident", "resident")], audit=False)
    state.engine = Engine()
    flat = torch.tensor([0, 9], dtype=torch.int32)
    state.layers.append(dict(shape=base.shape, text=(0,), flat=flat, bodies=[], repairs=[]))
    for c in (0, 1):
        fill_component(base, attention, c, verified)
        state.capture(base, attention, c, base.transport_patches[c], verified)
    compact = state.layers[0]["bodies"]
    assert len(compact) == 2
    assert all(t.shape == (1, 1, 2, 128) for t in compact)
    assert all(
        t.untyped_storage().data_ptr() != verified.untyped_storage().data_ptr() for t in compact
    )
    assert state.record()["full_reference_tensors_retained_across_components"] == 0
    state.close()
    assert state.closed and state.engine is None and state.layers == []


def test_new_model_geometries_fail_closed_before_native_engine_creation():
    from types import SimpleNamespace

    state = mod.State("handoff", [("resident", "resident")], audit=False)
    with pytest.raises(NotImplementedError, match="consumed"):
        state.begin(0, SimpleNamespace(shape=(1, 2, 9, 128)), (0,))


def test_scope_time_is_charged_even_for_an_ineligible_baseline(monkeypatch):
    from types import SimpleNamespace

    ticks = iter((100.0, 100.1))
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

    @mod.managed
    def call(*args, **kwargs):
        return None, None, dict(total_wall_seconds=0.01)

    value = call(None, None, None, "prefixkv_reimpl")[2]
    assert value["original_core_wall_seconds"] == 0.01
    assert value["total_wall_seconds"] == pytest.approx(0.1)
    assert value["verified_constructor_scope_overhead_seconds"] == pytest.approx(0.09)
