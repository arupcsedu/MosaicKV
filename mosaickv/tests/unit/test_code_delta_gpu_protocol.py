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


def test_complete_rows_require_all_methods_identities_bits_and_accounting():
    from evicastkv_code_delta_gpu import rows_complete

    from mosaickv.evaluation.common_streaming_timing import METHODS

    rows = [
        dict(
            status="completed",
            task=f"synthetic-task-{i}",
            sample_id=str(i),
            method=m,
            decoded_KV_bitwise=True,
            generation_128_logits_bitwise=True,
            complete_teacher_logits_bitwise=True,
            initial_active_bytes=100,
            generation_storage={"initial_active_kv_storage_bytes": 100},
            constructed_code_payload={"net_active_bytes": 100},
        )
        for i in range(2)
        for m in METHODS
    ]
    assert rows_complete(rows)
    assert not rows_complete([*rows[:-1], rows[0]])
    for key, value in (
        ("decoded_KV_bitwise", False),
        ("initial_active_bytes", 99),
        ("status", "failed"),
        ("method", "lookm_official"),
    ):
        broken = [dict(r) for r in rows]
        broken[0][key] = value
        assert not rows_complete(broken)
