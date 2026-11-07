from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.storage_six_overlay import LayerOverlay, OverlayPolicy, plan_layer


def overlay(phases=None):
    phases = phases or (torch.ones(3, 8), torch.zeros(3, 8))
    return LayerOverlay(
        head=1,
        positions=(2, 3),
        exact_ids=(4,),
        exact_pair=(torch.full((1, 8), 7.0), torch.full((1, 8), 8.0)),
        prototype_pair=(torch.full((1, 8), 2.0), torch.full((1, 8), 3.0)),
        phases=phases,
        residual_pair=(torch.full((2, 8), 4.0), torch.full((2, 8), 5.0)),
        anchor=2,
        dispersion=0.1,
        evidence={},
    )


def test_policy_strict():
    for kwargs in (
        {"layers": (31, 1)},
        {"layers": (32,)},
        {"extra_exact_positions": 0},
        {"force_repair": 1},
        {"prototype_members": 1},
    ):
        with pytest.raises(ValueError):
            OverlayPolicy(**kwargs)


def test_real_prototype_exact_payload_and_no_unrelated_changes():
    o = overlay()
    value = torch.zeros(1, 2, 8, 8)
    o.fill(1, value)
    assert (value[0, 1, 2:4] == 3).all()
    assert (value[0, 1, 4] == 8).all()
    assert not value[0, 0].any()
    assert not value[0, 1, 5:].any()
    assert not o.promoted.item()
    assert not o.promotion[0].any()  # no hidden original GPU residual


def test_real_residual_promotion_keeps_addresses_and_bytes():
    o = overlay()
    from mosaickv.evaluation.storage_six_overlay import _bytes

    before = _bytes(o.gpu_tensors())
    addresses = [t.data_ptr() for t in o.gpu_tensors()]
    event = o.promote()
    value = torch.zeros(1, 2, 8, 8)
    o.fill(0, value)
    assert (value[0, 1, 2:4] == 4).all()
    assert event["promoted_bitwise_original"]
    assert addresses == [t.data_ptr() for t in o.gpu_tensors()]
    assert before == _bytes(o.gpu_tensors())
    with pytest.raises(RuntimeError):
        o.promote()


def test_original_member_rotary_phases_not_anchor_phase_replicated():
    cos, sin = torch.ones(3, 8), torch.zeros(3, 8)
    cos[0].zero_()
    sin[0].fill_(1)
    o = overlay((cos, sin))
    value = torch.zeros(1, 2, 8, 8)
    o.fill(0, value)
    assert (value[0, 1, 2, :4] == -2).all()
    assert (value[0, 1, 2, 4:] == 2).all()
    assert (value[0, 1, 3] == 2).all()


def test_core_planner_executes_on_real_tensors_without_labels():
    from mosaickv.adapters.huggingface.llava import Llava15Adapter
    from mosaickv.cache_state import Modality, ModalitySpan

    torch.manual_seed(9)
    n, h, d = 16, 2, 128
    attention = SimpleNamespace(q_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16())
    prepared = SimpleNamespace(
        modality_map=(
            ModalitySpan(0, 12, Modality.IMAGE, image_index=0),
            ModalitySpan(12, 16, Modality.TEXT),
        )
    )
    original = tuple(torch.randn(1, h, n, d).bfloat16() for _ in range(2))
    hidden = torch.randn(1, n, h * d).bfloat16()
    phases = (torch.ones(1, n, d).bfloat16(), torch.zeros(1, n, d).bfloat16())
    p = torch.softmax(torch.randn(1, h, n, n), -1).bfloat16()
    adapter = SimpleNamespace(capabilities=Llava15Adapter.capabilities)
    o = plan_layer(
        adapter,
        prepared,
        attention,
        original,
        hidden,
        phases,
        p,
        ((*range(7), 15),) * h,
        OverlayPolicy(extra_exact_positions=2),
        pin_residuals=False,
    )
    assert o.evidence["forecast_mode"] == "prompt_window"
    assert o.evidence["graph_edges"] > 0
    assert o.evidence["value_contribution_max"] > 0
    assert len(o.evidence["prototype_members"]) >= 2
    assert not o.evidence["benchmark_labels_accessed"]
    assert not set(o.evidence["prototype_members"]) & set(
        o.evidence["original_protected_positions"]
    )


def test_scope_restores_original_functions_even_on_failure():
    from mosaickv.evaluation import storage_runs, streaming_source_constructor
    from mosaickv.evaluation.storage_six_overlay import integrated_storage

    original = storage_runs.run
    encoder = streaming_source_constructor.StreamingDirectSource
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        integrated_storage(None, OverlayPolicy(enabled=False)),
    ):
        assert storage_runs.run is not original
        raise RuntimeError("deliberate")
    assert storage_runs.run is original
    assert streaming_source_constructor.StreamingDirectSource is encoder
