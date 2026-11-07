from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.inline_first_step_repair import InlineRepair, hidden_after_attention
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def test_inline_repair_rejects_different_policy_and_restores_class_patches():
    from mosaickv.evaluation.post_attention_segment import Transport
    from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer
    from mosaickv.evaluation.uniform_core_transport import Binding

    original = (Transport.run, SequentialSourceHistoryLayer.materialize_component, Binding.scores)
    with pytest.raises(NotImplementedError):
        InlineRepair(None, None, OverlayPolicy(layers=(30,)), {})
    report = {}
    with (
        pytest.raises(RuntimeError, match="stop"),
        InlineRepair(None, None, OverlayPolicy(), report),
    ):
        assert Transport.run is not original[0]
        raise RuntimeError("stop")
    assert (
        Transport.run,
        SequentialSourceHistoryLayer.materialize_component,
        Binding.scores,
    ) == original
    assert report["inline_first_step_repair"]["retained_private_tensor_owners"] == 0


def test_actual_frozen_hidden_suffix_does_not_modify_input_residual():
    torch.manual_seed(84)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=torch.nn.Linear(8, 8).eval()),
        post_attention_layernorm=torch.nn.LayerNorm(8).eval(),
        mlp=torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU()).eval(),
    )
    residual, outputs = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    original = residual.clone()
    hidden = residual + layer.self_attn.o_proj(outputs)
    expected = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    assert torch.equal(expected, hidden_after_attention(layer, residual, outputs))
    assert torch.equal(residual, original)


def test_no_repair_returns_the_original_hidden_object_and_retires_observers():
    adapter = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(norm=torch.nn.Identity())),
            get_output_embeddings=lambda: lambda t: torch.tensor([[[20.0, 0.0, -20.0]]]),
        )
    )
    report = dict(repair_events=[])
    repair = InlineRepair(adapter, None, OverlayPolicy(), report)
    repair.key = torch.zeros(1, 32, 4, 128).bfloat16()
    repair.value = repair.key.clone()
    repair.query = torch.zeros(1, 32, 1, 128).bfloat16()
    repair.heads = torch.zeros(1, 1, 32, 128).bfloat16()
    repair.scaling = 128**-0.5
    output = torch.zeros(1, 1, 4096).bfloat16()
    layer = SimpleNamespace(self_attn=SimpleNamespace(layer_idx=31))
    observed = []
    transport = SimpleNamespace(
        cache=SimpleNamespace(observe=lambda values: observed.append(values))
    )
    result = repair.finish(transport, layer, output.clone(), output)
    assert result is output and repair.done
    assert not report["repair_events"] and repair.key is repair.query is None
    assert repair.evidence["suffix_recomputations"] == 0 and observed


def test_forced_real_restoration_recomputes_only_final_suffix_and_preserves_other_heads():
    from mosaickv.evaluation.storage_six_overlay import LayerOverlay

    pair = (torch.zeros(2, 128).bfloat16(), torch.ones(2, 128).bfloat16())
    overlay = LayerOverlay(
        head=0,
        positions=(0, 1),
        exact_ids=(2,),
        exact_pair=tuple(torch.zeros(1, 128).bfloat16() for _ in range(2)),
        prototype_pair=tuple(torch.zeros(1, 128).bfloat16() for _ in range(2)),
        phases=(torch.ones(3, 128).bfloat16(), torch.zeros(3, 128).bfloat16()),
        residual_pair=pair,
        anchor=0,
        dispersion=1.0,
        evidence={},
    )
    adapter = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(norm=torch.nn.Identity())),
            get_output_embeddings=lambda: lambda t: t[..., :3],
        )
    )
    report = dict(repair_events=[])
    repair = InlineRepair(adapter, overlay, OverlayPolicy(force_repair=True), report)
    repair.key = torch.zeros(1, 32, 4, 128).bfloat16()
    repair.value = repair.key.clone()
    value = repair.value
    repair.query = torch.zeros(1, 32, 1, 128).bfloat16()
    repair.heads = torch.zeros(1, 1, 32, 128).bfloat16()
    repair.scaling = 128**-0.5
    output = torch.zeros(1, 1, 4096).bfloat16()
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(layer_idx=31, o_proj=torch.nn.Identity()),
        post_attention_layernorm=torch.nn.Identity(),
        mlp=torch.zeros_like,
    )
    transport = SimpleNamespace(cache=SimpleNamespace(observe=lambda values: None))
    result = repair.finish(transport, layer, output.clone(), output)
    assert (result[..., :128] == 0.5).all() and not result[..., 128:].any()
    assert not value[:, 1:].any() and not value[0, 0, 2:].any()
    assert overlay.repair_count == 1 and repair.evidence["suffix_recomputations"] == 1
    assert report["repair_events"][0]["stress_control"] is True
    assert report["repair_events"][0]["whole_model_redecode_performed"] is False
