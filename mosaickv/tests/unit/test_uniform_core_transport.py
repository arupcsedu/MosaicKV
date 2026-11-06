from types import SimpleNamespace

import pytest

from mosaickv.evaluation import uniform_core_transport as transport


def cache(lengths):
    return SimpleNamespace(
        layers=[SimpleNamespace(prompt_length=n, maximum_decode_updates=128) for n in lengths],
        audit_workspace=True,
    )


def test_default_off_and_request_scope_isolation():
    @transport.managed
    def run():
        assert transport.bind(cache((600,) * 32)) is None
        return 1, 2, {}

    assert run()[2] == dict(uniform_attention_core=dict(requested=False))
    assert transport.bind(cache((600,) * 32)) is None


def test_adaptive_binding_is_exact_original_without_allocations():
    @transport.managed
    def run(**kwargs):
        state = transport.bind(cache(tuple(range(300, 332))))
        assert not state.eligible and state.core is None
        state.close()
        assert state.cache is None and state.core is None
        return 1, 2, {}

    info = run(uniform_attention_core=True, prompt_graph_reader=True)[2]["uniform_attention_core"]
    assert info["strategy"] == "original_eager_nonuniform"
    assert info["captures"] == info["workspace_bytes"] == info["calls"] == 0
    assert info["graphs_retired"] and not info["aborted"]


def test_failed_request_retires_and_resets_context():
    states = []

    @transport.managed
    def run(**kwargs):
        states.append(transport.bind(cache((600,) * 32)))
        raise RuntimeError("request failed")

    with pytest.raises(RuntimeError, match="request failed"):
        run(uniform_attention_core=True, prompt_graph_reader=True)
    assert states[0].closed and states[0].result["aborted"]
    assert transport.bind(cache((600,) * 32)) is None


@pytest.mark.parametrize("value", [1, "yes", None])
def test_implicit_boolean_is_rejected(value):
    @transport.managed
    def run(**kwargs):
        pytest.fail("invalid scope entered")

    with pytest.raises(ValueError, match="explicit"):
        run(uniform_attention_core=value)


def test_missing_binding_or_missing_retirement_is_not_success():
    @transport.managed
    def run(**kwargs):
        return 1, 2, {}

    with pytest.raises(RuntimeError, match="bound and retired"):
        run(uniform_attention_core=True, prompt_graph_reader=True)
