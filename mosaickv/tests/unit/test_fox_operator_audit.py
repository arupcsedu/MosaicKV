from pathlib import Path

import pytest
import torch

from mosaickv.evaluation.fox_operator_audit import (
    OperatorCapture,
    boundary_localization,
    compare_captures,
    difference,
    padding_localization,
)


def tiny(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify_hf_adapters import tiny_adapters

    from mosaickv.evaluation.messages import build_multimodal_messages

    _, adapter = tiny_adapters()[0]
    return adapter, adapter.prepare_inputs(build_multimodal_messages("tiny diagnostic prompt"))


def test_capture_hooks_are_read_only_and_cleaned_up(monkeypatch):
    adapter, prepared = tiny(monkeypatch)
    before = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    cap = OperatorCapture(adapter)
    with cap.install():
        observed = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    assert torch.equal(before.logits, observed.logits)
    assert cap.values and all(
        t.shape[1] == 1 and t.device.type == "cpu" for t in cap.values.values()
    )
    assert "layer.0.query_after_rope" in cap.values
    assert "layer.0.valid_attention_probabilities" in cap.values
    assert "layer.0.attention_context" in cap.values
    cap.clear()
    adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    assert not cap.values
    with pytest.raises(RuntimeError), cap.install():
        raise RuntimeError("cleanup")
    assert not any(
        m._forward_hooks for layer in adapter._language_layers() for m in layer.modules()
    )


def test_localization_diagnoses_both_geometries_without_reclassifying(monkeypatch):
    adapter, prepared = tiny(monkeypatch)
    out = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    rows = padding_localization(adapter, out.state, int(out.next_token_id.item()), steps=2)
    assert len(rows) == 2 and all(
        r["padded_physical_length"] > r["full_physical_length"] for r in rows
    )
    b = boundary_localization(adapter, prepared)
    assert b["operators"] and b["logits"]["absolute_linf"] >= 0


def test_difference_and_order_are_strict():
    a, b = OperatorCapture(None), OperatorCapture(None)
    a.save("x", torch.ones(1, 2, 3))
    b.save("x", torch.zeros(1, 2, 3))
    assert compare_captures(a, b)["first_nonidentical_operator"] == "x"
    with pytest.raises(ValueError, match="more than once"):
        a.save("x", torch.ones(1, 1, 3))
    b.clear()
    with pytest.raises(ValueError, match="execution paths"):
        compare_captures(a, b)
    with pytest.raises(ValueError, match="nonfinite"):
        difference(torch.ones(1), torch.tensor([float("nan")]))
