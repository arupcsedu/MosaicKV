"""Tiny random-weight CPU feasibility; no benchmark evidence or downloads."""

import pytest
import torch
from test_hf_tiny_models import _llava

from mosaickv.adapters.huggingface.bounded_history_cache import install_bounded_history
from mosaickv.backends.hf_downstream_sensitivity import DownstreamCapture


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def capture(*, half=False):
    torch.manual_seed(91)
    adapter = _llava()
    if half:
        adapter.model.half()
    prefill = adapter.prefill(adapter.prepare_inputs([dict(role="user", content="tiny prompt")]))
    state = prefill.state
    state.model_state["mosaickv_packed_all_valid"] = True
    install_bounded_history(adapter, state, generated_capacity=8, sink_tokens=1, cap_bytes=1536)
    observer = DownstreamCapture(adapter)
    anchors = {}
    token = prefill.next_token_id
    for step in range(1, 5):
        with observer.capture("anchor", step):
            out = adapter.decode_one_token(token, state, capture_queries=False)
        token = out.next_token_id
        anchors[step] = out.logits.clone()
    return adapter, state, observer, anchors


@pytest.mark.parametrize("half", [False, True])
def test_isolated_anchor_replay_vjp_and_jvp_preserve_weights_cache_and_flags(half):
    adapter, state, observer, anchors = capture(half=half)
    assert set(observer.snapshots) == {1, 4}
    cache = [(x.keys.clone(), x.values.clone()) for x in state.past_key_values.layers]
    weights = [p.detach().clone() for p in adapter.model.parameters()]
    flags = [p.requires_grad for p in adapter.model.parameters()]
    references = {
        step: z + torch.linspace(-0.05, 0.05, z.numel()).reshape_as(z)
        for step, z in anchors.items()
    }
    direction = torch.randn(2, 16) * 0.01
    gradients, report = observer.gradients(references, anchors, jvp_direction=direction)
    assert set(gradients) == {1, 4}
    assert all(g.shape == (2, 16) and torch.isfinite(g).all() for g in gradients.values())
    assert report["prefixes"][0]["jvp_crosscheck_passed"]
    assert all(r["primal_bitwise_replayed"] for r in report["prefixes"])
    assert not report["parameters_updated"] and not report["candidate_forward_required"]
    assert not report["full_autoregressive_Jacobian"]
    assert flags == [p.requires_grad for p in adapter.model.parameters()]
    assert all(
        torch.equal(p, w) and p.grad is None
        for p, w in zip(adapter.model.parameters(), weights, strict=True)
    )
    assert all(
        torch.equal(x.keys, k) and torch.equal(x.values, v)
        for x, (k, v) in zip(state.past_key_values.layers, cache, strict=True)
    )
    assert all(
        not row["keys"].is_inference() for rows in observer.snapshots.values() for row in rows
    )


def test_changed_anchor_logit_fails_closed_and_restores_parameter_flags():
    adapter, _, observer, anchors = capture()
    flags = [p.requires_grad for p in adapter.model.parameters()]
    anchors[1] = anchors[1] + 0.01
    with pytest.raises(RuntimeError, match="bitwise replay"):
        observer.gradients(anchors, anchors)
    assert flags == [p.requires_grad for p in adapter.model.parameters()]
    assert all(p.grad is None for p in adapter.model.parameters())


def test_capture_rejects_duplicate_and_does_not_record_candidate_activations():
    _, _, observer, _ = capture()
    with observer.capture("candidate", 1):
        pass
    assert set(observer.snapshots) == {1, 4}
    with pytest.raises(ValueError, match="duplicate"), observer.capture("anchor", 1):
        pass


def test_complete_consumed_input_diagnostic_keeps_original_guard_and_exact_storage(tmp_path):
    from dataclasses import replace

    from test_bounded_history_runtime import _config
    from test_hf_tiny_models import _runtime_request

    from mosaickv.backends.hf_allocation_guard import packed_positions_sha
    from mosaickv.config import BoundedDecodeConfig
    from mosaickv.evaluation.downstream_head_diagnostic import DownstreamHeadDiagnostic
    from mosaickv.evaluation.head_substitution_followup import HeadSubstitutionFollowup
    from mosaickv.selection.head_substitution import substitution_pool
    from mosaickv.types import MosaicKVMethod

    torch.manual_seed(37)
    adapter = _llava()
    base = _config(MosaicKVMethod.MOSAICKV_EXACT)
    cfg = replace(
        base,
        cache=replace(base.cache, budget_value=768),
        bounded_decode=BoundedDecodeConfig(generated_capacity=4, sink_tokens=1),
    )

    class Reference(HeadSubstitutionFollowup):
        def _observe_exact_anchor(self, prefill, full_state, anchor, attention, budget):
            self.followup["anchor_positions_sha256"] = packed_positions_sha(anchor)
            self.jvp_proposal = substitution_pool(
                full_state, attention, anchor, prompt_bytes=budget.prompt_budget_bytes, limit=1
            )[0]
            return super()._observe_exact_anchor(prefill, full_state, anchor, attention, budget)

    reference = Reference(
        adapter,
        cfg,
        trace_directory=tmp_path / "reference",
        followup=dict(anchor_positions_sha256="synthetic-only", survivors=[]),
    )
    assert reference.diagnose(_runtime_request("reference"))["controls_passed"]
    runtime = DownstreamHeadDiagnostic(
        adapter,
        cfg,
        trace_directory=tmp_path / "downstream",
        anchor_positions_sha256=reference.followup["anchor_positions_sha256"],
        reference_logits={k: reference.saved_logits[k] for k in ("fullkv", "anchor")},
        jvp_proposal=reference.jvp_proposal,
    )
    result = runtime.diagnose(_runtime_request("downstream"))
    assert (
        result["controls_passed"] and result["original_reference_and_anchor_32_prefixes_replayed"]
    )
    assert result["original_guard_unchanged"] and result["policy"]["guard"]["horizon"] == 32
    assert result["active_cap_bytes"] == 768 and result["padding_bytes"] == 0
    assert result["measured_source_cache_unchanged"] and result["logical_positions_unchanged"]
    assert result["live_private_states_after_diagnostic"] == 0
    assert result["downstream_sensitivity"]["prefixes"][0]["jvp_crosscheck_passed"]
    assert result["downstream_sensitivity"]["candidate_forward_calls_for_ranking"] == 0
    assert all(
        r["ranking_objective"] == "frozen_downstream_directional_KL" for r in result["pool_trace"]
    )
    assert not any(
        result[k]
        for k in (
            "benchmark_labels_accessed",
            "fresh_cohort_authorized",
            "calibration_authorized",
            "paper_campaign_authorized",
        )
    )
