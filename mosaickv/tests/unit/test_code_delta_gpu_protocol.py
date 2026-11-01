import pytest


@pytest.mark.parametrize(
    "complete,failures,identity", [(False, [], False), (True, ["bad"], True), (True, [], False)]
)
def test_lifetime_followup_cannot_repeat_failed_replay(
    complete, failures, identity, monkeypatch, tmp_path
):
    import evicastkv_code_delta_gpu as module

    records = []
    monkeypatch.setattr(
        module.previous.common, "inputs", lambda *a: {"constructor_lifetime_audit": True}
    )
    monkeypatch.setattr(
        module.previous,
        "read",
        lambda *a: dict(complete=complete, failures=failures, all_five_methods_bitwise=identity),
    )
    monkeypatch.setattr(module, "sha", lambda *a: "bound-parent-outcome")
    monkeypatch.setattr(module, "write_once", lambda path, record: records.append(record))
    monkeypatch.setattr(
        module.common.control.parent, "submit", lambda *a: pytest.fail("unsafe GPU retry")
    )
    module.submit(tmp_path, "plan")
    assert len(records) == 1
    assert records[0]["GPU_submitted"] is False
    assert records[0]["parent_decision_sha256"] == "bound-parent-outcome"


def test_only_explicit_lifetime_scope_registers_followup(monkeypatch, tmp_path):
    import evicastkv_code_delta_gpu as module

    monkeypatch.setattr(module.previous.common, "inputs", lambda *a: {})
    with pytest.raises(ValueError, match="only the registered"):
        module.register_lifetimes(tmp_path, "plan")
