from dataclasses import replace

import pytest
import torch

from mosaickv.evaluation.storage_component_gate import (
    logit_diagnostics,
    stage_effect,
    storage_match,
)
from mosaickv.evaluation.storage_six_overlay import COMPONENT_MODES, OverlayPolicy


def test_component_modes_strict_and_old_default_unchanged():
    assert OverlayPolicy().component_mode == "full"
    assert not OverlayPolicy().record_phase_timings
    for mode in COMPONENT_MODES:
        assert replace(OverlayPolicy(), component_mode=mode).component_mode == mode
    for kw in (
        {"component_mode": "unknown"},
        {"record_phase_timings": 1},
        {"enabled": False, "component_mode": "no_value"},
    ):
        with pytest.raises(ValueError):
            OverlayPolicy(**kw)


def test_first_divergence_included_but_later_logits_not_shared_prefix():
    ref = torch.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    actual = ref.clone()
    actual[1] = actual[1].flip(0)
    result = logit_diagnostics(ref, actual, horizon=4)
    assert result["first_divergent_prediction"] == 1
    assert result["shared_prefix_prediction_count"] == 2
    assert not result["fixed_prefix_for_entire_horizon"]
    assert result["token_agreement"] == 0.75
    assert stage_effect(result, result)["ablated_minus_full_shared_prefix_mean_kl"] is None


def test_bitwise_readout_and_malformed_inputs():
    ref = torch.randn(32, 16)
    result = logit_diagnostics(ref, ref)
    assert result["maximum_kl"] == 0
    assert result["first_divergent_prediction"] is None
    assert stage_effect(result, result)["ablated_minus_full_shared_prefix_mean_kl"] == 0
    for actual in (ref[:16], ref * float("nan")):
        with pytest.raises(ValueError):
            logit_diagnostics(ref, actual)


def test_actual_backing_boundaries_not_nominal_cap():
    row = dict(
        status="completed",
        detail=dict(
            initial_encoded_bytes=80, final_encoded_bytes=100, reported_counter_peak_bytes=120
        ),
    )
    assert storage_match([row, row])["all_boundaries_matched"]
    other = dict(status="completed", detail=dict(row["detail"], final_encoded_bytes=101))
    result = storage_match([row, other])
    assert not result["all_boundaries_matched"]
    assert result["boundaries"]["final_encoded_bytes"]["spread_bytes"] == 1
    for bad in ([], [dict(row, status="failed")], [dict(status="completed", detail={})]):
        with pytest.raises(ValueError):
            storage_match(bad)


def test_same_real_allocations_for_inactive_no_exact_and_real_prototype_reads():
    from test_storage_six_overlay import overlay

    from mosaickv.evaluation.storage_six_overlay import _bytes

    o = overlay()
    before = _bytes(o.gpu_tensors())
    for mode in COMPONENT_MODES:
        o.component_mode = mode
        value = torch.zeros(1, 2, 8, 8)
        o.fill(1, value)
        assert _bytes(o.gpu_tensors()) == before
        if mode == "parent_reserved":
            assert not value.any()
        elif mode == "no_extra_exact":
            assert not value[0, 1, 4].any()
            assert (value[0, 1, 2:4] == 3).all()
        else:
            assert (value[0, 1, 4] == 8).all()


@pytest.mark.parametrize("mode", COMPONENT_MODES)
def test_all_core_ablation_plans_have_identical_actual_allocations(mode):
    from types import SimpleNamespace

    from mosaickv.adapters.huggingface.llava import Llava15Adapter
    from mosaickv.cache_state import Modality, ModalitySpan
    from mosaickv.evaluation.storage_six_overlay import _bytes, plan_layer

    torch.manual_seed(9)
    n, h, d = 64, 2, 128
    attention = SimpleNamespace(q_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16())
    prepared = SimpleNamespace(
        modality_map=(
            ModalitySpan(0, 56, Modality.IMAGE, image_index=0),
            ModalitySpan(56, 64, Modality.TEXT),
        )
    )
    original = tuple(torch.randn(1, h, n, d).bfloat16() for _ in range(2))
    hidden = torch.randn(1, n, h * d).bfloat16()
    phases = (torch.ones(1, n, d).bfloat16(), torch.zeros(1, n, d).bfloat16())
    probs = torch.softmax(torch.randn(1, h, n, n), -1).bfloat16()
    args = (
        SimpleNamespace(capabilities=Llava15Adapter.capabilities),
        prepared,
        attention,
        original,
        hidden,
        phases,
        probs,
        ((*range(7), n - 1),) * h,
    )
    common = dict(extra_exact_positions=2, record_phase_timings=True)
    base = plan_layer(*args, OverlayPolicy(**common), pin_residuals=False)
    actual = plan_layer(*args, OverlayPolicy(**common, component_mode=mode), pin_residuals=False)
    assert _bytes(actual.gpu_tensors()) == _bytes(base.gpu_tensors())
    assert actual.cpu_bytes == base.cpu_bytes
    assert actual.evidence["component_mode"] == mode
    assert actual.evidence["construction_phase_seconds"]
    assert (
        actual.evidence["original_protected_positions"]
        == base.evidence["original_protected_positions"]
    )
    if mode == "no_graph":
        assert actual.evidence["graph_edges"] == 0
    if mode == "no_forecast":
        assert actual.evidence["forecast_mode"] == "observed_prompt_attention_no_forecast"
    if mode == "exact_members":
        assert actual.promoted.item()
        assert not actual.repair_count
        assert actual.evidence["exact_member_control_preloaded"]


def test_no_answer_or_reference_inputs_in_core_planner():
    import inspect

    from mosaickv.evaluation.storage_six_overlay import plan_layer

    parameters = inspect.signature(plan_layer).parameters
    assert not {"answer", "references", "reference", "task_score", "labels"} & set(parameters)


def test_precision_decomposition_preserves_input_and_marks_non_quality_evidence():
    from mosaickv.evaluation.storage_component_gate import precision_evidence

    original = (torch.zeros(8, 8), torch.ones(8, 8))
    parent = tuple(t.clone() for t in original)
    parent[1][4].add_(1)
    payload = dict(
        original_pair=original,
        parent_pair=parent,
        prompt_queries=torch.ones(4, 8),
        evidence=dict(selected_extra=[4], prototype_members=[2, 3]),
        overlay=dict(member_keys=torch.zeros(2, 8), prototype_value=torch.ones(1, 8)),
        evaluation_only=True,
        labels_accessed=False,
    )
    result = precision_evidence(payload)
    assert result["variants"]["parent"]["mean_squared_output_error"] > 0
    assert result["variants"]["extra_exact_only"]["mean_squared_output_error"] == 0
    assert not result["generated_answer_quality_test"]
    assert not result["prototype_retires_parent_source_storage"]
    assert not result["replacement_gpu_authorized"]
    assert (parent[1][4] == 2).all()
    with pytest.raises(ValueError):
        precision_evidence(dict(payload, labels_accessed=True))


def test_cpu_analyzer_handles_native_fullkv_and_structured_caption_records(tmp_path, monkeypatch):
    import json

    import storage_six_ablation as script

    from mosaickv.evaluation import coco_shared_bytes

    # Pure synthetic CPU fixture; never a measured result or official scorer.
    class FixtureScorer:
        def __init__(self, samples):
            assert all(isinstance(s.prediction, str) for s in samples)

        def __call__(self, samples):
            return 0.0

    monkeypatch.setattr(coco_shared_bytes, "TokenizedCider", FixtureScorer)
    monkeypatch.setattr(script, "checked", lambda *args: {"source_sha": "fixture"})
    monkeypatch.setattr(script, "PARENT", tmp_path / "parent")
    script.PARENT.mkdir()
    (script.PARENT / "ledger.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"sample_id": str(i), "references": ["fixture caption"]} for i in range(2)
                ]
            }
        )
    )
    (tmp_path / "inputs.json").write_text(json.dumps([{"sample_id": str(i)} for i in range(2)]))
    for i in range(2):
        folder = tmp_path / f"workers/{i}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps({"status": "completed"}))
        (folder / "storage_matching.json").write_text(json.dumps({"all_boundaries_matched": True}))
        logits = torch.zeros(128, 8)
        logits[:, 0] = 1
        for j, arm in enumerate(script.ARMS):
            path = folder / f"{arm}.pt"
            torch.save(logits, path)
            details = dict(answer={"answer": "fixture caption"}, end_to_end_seconds=1.0)
            if arm == "full_kv":
                details["storage"] = dict(
                    initial_active_kv_storage_bytes=120, final_active_kv_storage_bytes=140
                )
                details["decode_owned_peak_bytes"] = None
            else:
                details.update(
                    initial_encoded_bytes=80,
                    final_encoded_bytes=100,
                    reported_counter_peak_bytes=120,
                )
            row = dict(
                arm=arm,
                sample_id=str(i),
                status="completed",
                detail=details,
                logits_path=str(path),
                logits_sha256=script.sha(path),
            )
            if arm != "full_kv":
                row["fullkv_readout"] = logit_diagnostics(logits, logits)
            (folder / f"{j:02}-{arm}.json").write_text(json.dumps(row))
    script.analyze(tmp_path, "fixture")
    result = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert result["execution_passed"]
    assert result["actual_rows"] == 24
    assert not result["replacement_gpu_authorized"]
    assert not result["superiority_established"]
    assert result["common_prefix_effects"]["0"]["common_fullkv_prefix_predictions"] == 32
    assert "not_certified" in (tmp_path / "analysis/report.md").read_text()
