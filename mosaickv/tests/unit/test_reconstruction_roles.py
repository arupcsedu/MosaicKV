from contextvars import copy_context
from types import SimpleNamespace

import pytest

from mosaickv.evaluation import reconstruction_policy as policy
from mosaickv.evaluation.reconstruction_roles import choose_order, packet_role


def test_request_policy_nested_exception_and_context_isolation():
    assert policy.current() == "original"
    with policy.use("reuse_headers"):
        assert policy.reuse_headers() and not policy.carry_literals()
        fork = copy_context()
        with pytest.raises(RuntimeError), policy.use("reuse_both"):
            assert policy.carry_literals()
            raise RuntimeError()
        assert policy.current() == "reuse_headers"
    assert policy.current() == "original"
    assert fork.run(policy.current) == "reuse_headers"
    for invalid in (True, None, "approximate", 1):
        with pytest.raises(ValueError):
            policy.validate(invalid)


def test_role_uses_request_ownership_not_shapes_or_tensor_values():
    prompt, history = object(), object()
    layer = SimpleNamespace(representation=prompt, source_sealed=[history])
    assert packet_role(prompt, layer) == "prompt_source"
    assert packet_role(history, layer) == "generated_history"
    assert packet_role(prompt, None) == "history_construction"
    with pytest.raises(ValueError, match="unowned"):
        packet_role(object(), layer)


def test_reader_order_follows_operation_cost_and_requires_both_inputs():
    def groups(a, b, c):
        return [
            dict(
                method="mosaickv_exact",
                input_index=i,
                phases={
                    role + ".projection": dict(mean_inclusive_cuda_stream_ms=value)
                    for role, value in zip(
                        ("prompt_source", "text_overlay", "generated_history"),
                        (a, b, c),
                        strict=True,
                    )
                },
            )
            for i in (0, 1)
        ]

    assert choose_order(groups(4, 2, 3))[0] == ["reuse_headers", "carry_literals"]
    assert choose_order(groups(2, 3, 4))[0] == ["carry_literals", "reuse_headers"]
    with pytest.raises(ValueError):
        choose_order(groups(4, 2, 3)[:1])
    with pytest.raises(ValueError):
        choose_order(groups(float("nan"), 2, 3))


def test_role_scopes_restore_on_error_without_changing_results(monkeypatch):
    from mosaickv.evaluation import encoded_projection_kernel as encoded
    from mosaickv.evaluation import sequential_source_history as history
    from mosaickv.evaluation import shared_source_kernel as source
    from mosaickv.evaluation.construction_phase_timing import PhaseTimer
    from mosaickv.evaluation.reconstruction_roles import instrument

    prompt, old = object(), object()
    seen = []
    monkeypatch.setattr(source, "project", lambda *a: seen.append("source"))
    monkeypatch.setattr(encoded, "project", lambda *a: seen.append("encoded"))

    def fill(packet, *args):
        if packet is prompt:
            source.project()
            encoded.project()
        else:
            encoded.project()

    monkeypatch.setattr(history, "fill_component", fill)

    def materialize(self, *args):
        history.fill_component(self.representation)
        history.fill_component(self.source_sealed[0])
        return "unchanged"

    monkeypatch.setattr(history.SequentialSourceHistoryLayer, "materialize_component", materialize)
    timer = PhaseTimer()
    layer = SimpleNamespace(representation=prompt, source_sealed=[old])
    with instrument(timer):
        assert history.SequentialSourceHistoryLayer.materialize_component(layer) == "unchanged"
    assert history.fill_component is fill
    assert seen == ["source", "encoded", "encoded"]
    assert {r["phase"] for r in timer.resolve()} == {
        "prompt_source.reconstruction",
        "generated_history.reconstruction",
        "prompt_source.projection",
        "text_overlay.projection",
        "generated_history.projection",
    }
