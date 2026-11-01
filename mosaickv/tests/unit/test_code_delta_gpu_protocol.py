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


def test_memory_audit_and_timing_policies_remain_separate(monkeypatch):
    from evicastkv_code_delta_gpu import decode_run

    calls = []
    monkeypatch.setattr(
        "mosaickv.evaluation.storage_runs.run", lambda *args, **kwargs: calls.append(kwargs)
    )
    decode_run(None, None, (), audit=True)
    decode_run(None, None, ())
    assert [r["audit"] for r in calls] == [True, False]
    assert all(r["tokens"] == 128 and r["sequential_source_history"] for r in calls)


def test_teacher_hash_comes_from_bound_original_not_derived_storage_schema():
    from evicastkv_code_delta_gpu import teacher_reference_hash

    # This fixture reproduces the actual two producer SCHEMAS, not their scores.
    fields = dict(
        nll_sum=1.0,
        per_token_nll=[1.0],
        conditional_greedy_tokens=[1],
        evaluated_token_count=1,
        target_ids=[1],
        target_mask="fixture",
        no_EOS_stop=True,
        no_truncation=True,
    )
    current = dict(fields, storage={"source_history": True})
    original = dict(
        fields,
        storage={
            "raw_history_control": dict(
                complete_teacher_logits_bitwise=True,
                maximum_logit_difference=0,
                source_history_logit_sha256="a" * 64,
                original_logit_sha256="a" * 64,
            )
        },
    )
    assert teacher_reference_hash(current, original) == "a" * 64
    with pytest.raises(ValueError):
        teacher_reference_hash(dict(current, nll_sum=2.0), original)
    with pytest.raises(ValueError):
        teacher_reference_hash(current, dict(original, storage={}))
