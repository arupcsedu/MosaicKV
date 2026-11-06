import copy

import evicastkv_prefill_catalog as control
import pytest


def test_every_arm_requires_a_real_independent_owner_check(monkeypatch):
    info = dict(
        policy="immutable_layer_storage_segments_v1",
        tensor_owners_retained=False,
        independent_full_enumerations=1,
    )
    value = dict(prefill_storage=dict(ownership_catalog=info))
    monkeypatch.setattr(control, "OLD_EXECUTE", lambda *a, **k: value)
    assert control.execute() is value
    info["independent_full_enumerations"] = 0
    with pytest.raises(RuntimeError, match="reconciliation"):
        control.execute()


def test_metadata_optimization_cannot_change_any_frozen_storage_boundary(monkeypatch):
    value = dict(
        prefill_storage=dict(logical_lengths=[600] * 32, ownership_catalog={}),
        **dict.fromkeys(control.driver.STORAGE, 10),
    )
    audited = {arm: copy.deepcopy(value) for arm in control.ARMS}
    monkeypatch.setattr(control.driver, "read", lambda path: value)
    monkeypatch.setattr(control.base, "ORIGINAL_SUMMARY", lambda rows, audited: {})
    assert control.summary([], audited)[
        "all_frozen_initial_final_decode_owned_peak_bytes_identical"
    ]
    audited[control.ARMS[-1]][control.driver.STORAGE[-1]] += 1
    with pytest.raises(RuntimeError, match="backing storage"):
        control.summary([], audited)


def test_faster_constructor_is_frozen_and_common_policy_is_truthfully_labeled(monkeypatch):
    for name in ("CANDIDATE_ARM", "VARIANT", "JOB_NAME"):
        monkeypatch.setattr(control.base.control, name, None)
    monkeypatch.setattr(control, "OLD_MAIN", lambda: None)
    control.finish_configuration()
    assert control.base.control.CANDIDATE_ARM == "joint-same-buffer"
    assert control.base.control.VARIANT == "consumed_common_prefill_owner_catalog"
