"""Physical-budget regression fixtures, not experiment observations."""

from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.initial_byte_budget import (
    InitialByteBudgetExceeded,
    inspect_initial_budget,
    validate_cap,
)


def cache():
    # A short view retains all 64 backing bytes; the shared alias is counted
    # once. Another four-byte owner models decode binding's auxiliary storage.
    source = torch.zeros(16)
    extra = torch.zeros(1)
    return SimpleNamespace(
        shared_tensors=(source[:1],),
        initial_bytes=68,
        layers=[SimpleNamespace(owned_tensors=lambda: (source[2:4], extra))],
    )


def test_bound_auxiliary_and_alias_backing_are_not_packet_payload():
    result = inspect_initial_budget(cache(), 64)
    assert result["actual_bytes"] == 68
    assert not result["within_cap"] and result["excess_bytes"] == 4
    result = inspect_initial_budget(cache(), 68)
    assert result["within_cap"] and result["shortfall_bytes"] == 0
    assert not result["largest_feasible_allocation_certified"]


@pytest.mark.parametrize("cap", [True, 0, -1, 64.0, float("nan")])
def test_invalid_caps_fail(cap):
    with pytest.raises(ValueError):
        validate_cap(cap)


def test_initial_budget_fails_before_decode_and_retires_private_graphs(monkeypatch):
    from mosaickv.evaluation import storage_runs

    state = cache()
    retired = []
    state.prompt_graph_pool = SimpleNamespace(close=lambda: retired.append(True))
    monkeypatch.setattr(storage_runs, "cache_from_formats", lambda *a, **k: state)
    # Adapter/prefill deliberately have no usable GPU/decode APIs. Entering
    # generation instead of rejecting the budget would fail this regression.
    with pytest.raises(InitialByteBudgetExceeded) as error:
        storage_runs.run(
            None, None, (), tokens=2, initial_physical_byte_cap=64, prompt_graph_reader=True
        )
    assert error.value.evidence["actual_bytes"] == 68
    assert error.value.evidence["before_first_decode"] and retired == [True]


def test_counter_mismatch_is_not_silently_treated_as_feasible(monkeypatch):
    from mosaickv.evaluation import storage_runs

    state = cache()
    state.initial_bytes = 64
    retired = []
    state.prompt_graph_pool = SimpleNamespace(close=lambda: retired.append(True))
    monkeypatch.setattr(storage_runs, "cache_from_formats", lambda *a, **k: state)
    with pytest.raises(RuntimeError, match="independent backing"):
        storage_runs.run(
            None, None, (), tokens=2, initial_physical_byte_cap=68, prompt_graph_reader=True
        )
    assert retired == [True]


def test_controller_executes_apply_mode_instead_of_inert_dry_run(monkeypatch):
    import sys

    import evicastkv_paper_boundaries as script

    captured = []
    monkeypatch.setattr(script, "check", lambda *a: None)
    monkeypatch.setattr(
        script.subprocess, "run", lambda command, **kwargs: captured.append(command)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["boundaries", "audit_controller", "--root", "/scratch/run", "--plan-sha", "a" * 64],
    )
    script.main()
    assert len(captured) == 1
    assert "--apply" in captured[0] and "--watch" in captured[0]


def test_cap_canary_scope_cannot_expand_to_other_methods_or_ids(tmp_path, monkeypatch):
    import evicastkv_paper_boundaries as script

    monkeypatch.setattr(script, "clean_source", lambda *a: "a" * 40)

    def prepare(root, *args, **kwargs):
        root.mkdir()
        script.write_once(root / "plan.json", dict(source_sha="a" * 40))

    monkeypatch.setattr(script.campaign, "prepare", prepare)
    monkeypatch.setattr(script, "submit", lambda *a, **k: "not_submitted")
    root = tmp_path / "cap"
    script.prepare(root, verify_initial_cap=True)
    plan = script.campaign.read(root / "plan.json")
    monkeypatch.setattr(script.campaign, "checked", lambda *a: plan)
    _, scope = script.check(root, script.sha(root / "plan.json"))
    assert scope["methods"] == ["mosaickv_exact", "prompt_attention_topk"]
    assert scope["consumed_sample_indices"] == [2]
    scope["consumed_sample_indices"] = [2, 3]
    (root / "scope.json").write_text(__import__("json").dumps(scope))
    with pytest.raises(ValueError, match="scope/source"):
        script.check(root, script.sha(root / "plan.json"))
