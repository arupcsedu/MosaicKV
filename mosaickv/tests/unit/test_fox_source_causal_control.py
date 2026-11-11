"""Counterfactual prompt-component construction, no pretrained weights/downloads."""

import evicastkv_source_causal_control as study
import pytest
import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


class Packet:
    shape = (1, 2, 3, 4)

    def __init__(self):
        self.body = torch.zeros(1, dtype=torch.bfloat16)
        self.calls = 0

    def private(self):
        return (self.body,)

    def shared(self):
        return ()

    def fill(self, attention, destinations, *, direct):
        assert attention is None and direct is True
        self.calls += 1
        destinations[0][..., :3, :].fill_(-7)
        destinations[1][..., :3, :].fill_(-8)


def original():
    return tuple(torch.full(Packet.shape, value, dtype=torch.bfloat16) for value in (1, 2))


@pytest.mark.parametrize("arm", list(study.ARMS))
def test_factorial_only_changes_prompt_component_and_preserves_history(arm):
    packet, raw = Packet(), original()
    before = tuple(t.clone() for t in raw)
    wrapper = study.PromptCounterfactual(packet, raw, study.ARMS[arm])
    destinations = tuple(torch.full((1, 2, 8, 4), -99, dtype=torch.bfloat16) for _ in range(2))
    for _ in range(127):
        wrapper.fill(None, destinations, direct=True)
    for index, (target, native, use_raw) in enumerate(
        zip(destinations, raw, study.ARMS[arm], strict=True)
    ):
        wanted = native if use_raw else torch.full_like(native, -7 - index)
        assert torch.equal(target[..., :3, :], wanted)
        assert bool((target[..., 3:, :] == -99).all())
        assert torch.equal(native, before[index])
    assert wrapper.fill_calls == packet.calls == 127
    assert wrapper.verified_raw_copies == [2 if v else 0 for v in study.ARMS[arm]]


def test_original_backing_is_retained_and_charged_for_every_arm():
    packet, raw = Packet(), original()
    expected = backing_bytes((packet.body, *raw))
    for flags in study.ARMS.values():
        wrapper = study.PromptCounterfactual(packet, raw, flags)
        assert wrapper.component_bytes == expected
        assert wrapper.component_bytes > backing_bytes((packet.body,))
        assert tuple(map(study.owner, wrapper.tensors())) == tuple(
            map(study.owner, (packet.body, *raw))
        )


def test_aliasing_originals_or_both_destinations_is_rejected_before_write():
    packet, raw = Packet(), original()
    wrapper = study.PromptCounterfactual(packet, raw, (True, False))
    with pytest.raises(ValueError, match="private disjoint"):
        wrapper.fill(None, raw, direct=True)
    target = torch.zeros(Packet.shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="private disjoint"):
        wrapper.fill(None, (target, target), direct=True)
    assert packet.calls == wrapper.fill_calls == 0


@pytest.mark.parametrize("flags", [(1, False), (True,), (True, False, True)])
def test_component_configuration_is_strict(flags):
    with pytest.raises(ValueError, match="component flags"):
        study.PromptCounterfactual(Packet(), original(), flags)


def test_incomplete_or_wrong_precision_originals_fail_closed():
    with pytest.raises(ValueError, match="complete batch-one"):
        study.PromptCounterfactual(Packet(), original()[:1], (True, True))
    with pytest.raises(ValueError, match="complete batch-one"):
        study.PromptCounterfactual(Packet(), tuple(t.float() for t in original()), (True, True))


def test_causal_reference_contract_contains_no_tuned_token_or_prefix():
    assert study.POLICY["guard"] == {
        "tokens": 32,
        "token_agreement": 1.0,
        "maximum_KL_FP32": 0.02,
        "logit_difference": 1.0,
    }
    assert study.POLICY["selector_prefix_access"] is False
    assert "tie_probe_ids" not in study.POLICY
    assert "tie_probe_prediction_zero_based" not in study.POLICY
    assert study.POLICY["no_shared_realized_byte_claim"] is True


def test_missing_outputs_are_a_preserved_negative_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "validate", lambda _: {"source_sha": "source"})
    monkeypatch.setattr(study.fixed, "analysis_cpu_controls", lambda: {})
    gate = study.analyze(tmp_path)
    assert gate["status"] == "blocked" and gate["failures"]
    assert gate["whole_request_KV_peak_certified"] is False
    assert gate["superiority_established"] is False
    assert gate["all_lossy_arms_pass_guard"] is False
    assert (tmp_path / "GATE_source_causal_control.json").exists()


def test_tensor_digest_binds_dtype_shape_and_values():
    tensor = torch.ones((1, 2), dtype=torch.bfloat16)
    assert study.tensor_sha(tensor) == study.tensor_sha(tensor.clone())
    assert study.tensor_sha(tensor) != study.tensor_sha(tensor.float())
    with pytest.raises(ValueError, match="finite original"):
        study.tensor_sha(torch.full((1, 2), float("nan")))
