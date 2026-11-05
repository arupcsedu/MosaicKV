import evicastkv_prefill_native_overlap as control
import pytest


def test_frozen_policy_is_fail_closed_and_not_a_quality_gate():
    with pytest.raises(ValueError):
        control.frozen_policy(dict(complete=False, failures=[], rows=[]))
    row = dict(
        own_initial_final_peak_identical=True,
        means=[
            dict(method="mosaickv_exact", arm="joint-frozen", total_wall_seconds=2),
            dict(method="mosaickv_exact", arm="joint-previous", total_wall_seconds=3),
        ],
    )
    assert control.frozen_policy(dict(complete=True, failures=[], rows=[row, row]))
    row["means"][0]["total_wall_seconds"] = 4
    assert not control.frozen_policy(dict(complete=True, failures=[], rows=[row, row]))


def test_common_policy_and_baseline_truthful_noop(monkeypatch):
    monkeypatch.setattr(
        control, "POLICIES", dict(segment=True, rotary=True, copy=False, frozen=True)
    )
    records = dict(
        mosaickv_exact={"graph-lt": {}}, prefixkv_reimpl={"graph": {}, "graph-original": {}}
    )
    configs = control.configurations(records)
    for key, record in configs.items():
        assert record["post_attention_segment"] and record["exact_rotary_dispatch"]
        assert record["frozen_joint_plan"]
        assert record["overlap_native_fit"] is (
            key[1] == "joint-overlap" or key[0] != "mosaickv_exact"
        )
    assert not configs["mosaickv_exact", "joint-overlap"]["reuse_source_input_copy"]
