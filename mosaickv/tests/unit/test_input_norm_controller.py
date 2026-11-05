import evicastkv_input_norm_segment as control
import pytest


def row():
    return dict(
        means=[
            dict(method=method, arm=arm, total_wall_seconds=time)
            for method, old, new in (
                ("mosaickv_exact", "joint-previous", "joint-history"),
                ("prefixkv_reimpl", "graph", "graph-history"),
                ("prefixkv_reimpl", "graph-original", "graph-original-history"),
            )
            for arm, time in ((old, 2), (new, 1))
        ]
    )


def test_history_is_only_promoted_if_every_own_mean_improves():
    r = row()
    decision = dict(complete=True, failures=[], rows=[r, row()])
    assert control.history_policy(decision)
    r["means"][-1]["total_wall_seconds"] = 3
    assert not control.history_policy(decision)
    with pytest.raises(ValueError, match="complete"):
        control.history_policy(dict(complete=False, failures=[], rows=[]))


def test_both_baseline_readers_receive_identical_optimization_controls(monkeypatch):
    monkeypatch.setattr(
        control, "POLICIES", dict(segment=True, rotary=True, copy=False, frozen=True, overlap=True)
    )
    monkeypatch.setattr(control, "HISTORY", True)
    records = dict(
        mosaickv_exact={"graph-lt": {}}, prefixkv_reimpl={"graph": {}, "graph-original": {}}
    )
    configs = control.configurations(records)
    assert len(configs) == 6
    for (_method, arm), record in configs.items():
        assert record["input_norm_segment"] is arm.endswith("-norm")
        assert record["native_history_reader"] is True
        assert record["post_attention_segment"] is True
        assert record["exact_rotary_dispatch"] is True
