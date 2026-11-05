import evicastkv_norm_input_reuse as control
import pytest


def test_failed_or_slower_norm_is_never_silently_promoted():
    with pytest.raises(ValueError, match="complete"):
        control.norm_policy(dict(complete=False, failures=[], rows=[]))
    row = dict(
        means=[
            dict(method=method, arm=arm, total_wall_seconds=time)
            for method, old, new in (
                ("mosaickv_exact", "joint-previous", "joint-norm"),
                ("prefixkv_reimpl", "graph", "graph-norm"),
                ("prefixkv_reimpl", "graph-original", "graph-original-norm"),
            )
            for arm, time in ((old, 2), (new, 1))
        ]
    )
    decision = dict(complete=True, failures=[], rows=[row, row])
    assert control.norm_policy(decision)
    row["means"][-1]["total_wall_seconds"] = 3
    assert not control.norm_policy(decision)


def test_common_new_variant_does_not_withhold_optimization_from_baselines(monkeypatch):
    monkeypatch.setattr(
        control.prior,
        "POLICIES",
        dict(segment=True, rotary=True, copy=False, frozen=True, overlap=True),
    )
    monkeypatch.setattr(control.prior, "HISTORY", False)
    monkeypatch.setattr(control, "NORM", False)
    records = dict(
        mosaickv_exact={"graph-lt": {}}, prefixkv_reimpl={"graph": {}, "graph-original": {}}
    )
    for (_method, arm), record in control.configurations(records).items():
        assert record["reuse_norm_input"] is arm.endswith("-reuse")
        assert record["input_norm_segment"] is arm.endswith("-reuse")
        assert record["native_history_reader"] is False
