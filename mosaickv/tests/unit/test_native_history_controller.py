import evicastkv_native_history_reader as control
import pytest


def test_all_readers_get_both_history_policies(monkeypatch):
    monkeypatch.setattr(
        control, "POLICIES", dict(segment=True, rotary=True, copy=False, frozen=True, overlap=False)
    )
    records = dict(
        mosaickv_exact={"graph-lt": {}}, prefixkv_reimpl={"graph": {}, "graph-original": {}}
    )
    configs = control.configurations(records)
    assert len(configs) == 6
    for (method, arm), record in configs.items():
        assert record["native_history_reader"] is arm.endswith("-history")
        assert record["post_attention_segment"] and record["exact_rotary_dispatch"]
        assert record["overlap_native_fit"] is (method != "mosaickv_exact")


def test_incomplete_predecessor_cannot_be_silently_bypassed():
    with pytest.raises(ValueError):
        control.overlap_policy(dict(complete=False, failures=["synthetic failure"], rows=[]))
    row = dict(
        own_initial_final_peak_identical=True,
        means=[
            dict(method="mosaickv_exact", arm="joint-overlap", total_wall_seconds=2),
            dict(method="mosaickv_exact", arm="joint-previous", total_wall_seconds=3),
        ],
    )
    assert control.overlap_policy(dict(complete=True, failures=[], rows=[row, row]))
    row["means"][0]["total_wall_seconds"] = 4
    assert not control.overlap_policy(dict(complete=True, failures=[], rows=[row, row]))
