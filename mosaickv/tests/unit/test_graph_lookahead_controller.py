import evicastkv_graph_lookahead as control
import pytest


def records():
    return {
        "mosaickv_exact": {"graph": {}, "graph-lt": {"lt_source_reader": True}},
        "prefixkv_reimpl": {"graph": {}, "graph-original": {}},
    }


def test_all_control_readers_receive_lookahead_and_originals_remain():
    original = records()
    proposed = control.variants(original)
    assert len(control.ARMS) == 8 and len(proposed) == 4
    assert sum(method == "prefixkv_reimpl" for method, _, _ in proposed) == 2
    assert all(row["graph_lookahead"] is True for _, _, row in proposed)
    assert all(
        "graph_lookahead" not in row for group in original.values() for row in group.values()
    )


def test_screen_never_grants_statistical_latency_acceptance():
    rows = [
        dict(
            method=method,
            arm=arm,
            trial=trial,
            warmup=trial < 0,
            measurement=dict(
                total_wall_seconds=0.9 if method == "mosaickv_exact" else 1.0,
                logits_bitwise=True,
                no_FullKV_fallback=True,
            ),
        )
        for method, arm in control.ARMS
        for trial in range(-5, 3)
    ]
    summary = control.summary(rows)
    assert summary["four_percent_POINT_only"] and not summary["formal_latency_acceptance"]
    assert len(summary["candidate_mean_ratios_vs_all_PrefixKV"]) == 2
    assert all(
        len(ratios) == 4 for ratios in summary["candidate_mean_ratios_vs_all_PrefixKV"].values()
    )
    with pytest.raises(ValueError):
        control.summary(rows[:-1])


def test_engagement_uses_all_same_token_routes_and_optimized_common_policies(monkeypatch):
    seen = []
    proof = dict(
        enabled=True,
        launches=3906,
        hits=3906,
        first_capture_cold_misses=31,
        skipped_token_boundaries=127,
        extra_tensor_workspace_bytes=0,
        prompt_copies=0,
        pending=False,
        retired=True,
    )

    def execute(*args, **kwargs):
        seen.append(args[-1])
        return dict(prompt_graph_reader=dict(lookahead=proof))

    monkeypatch.setattr(control, "ORIGINAL_EXECUTE", execute)
    row = dict(lt_source_reader=True, graph_lookahead=True)
    args = (None, None, None, None, None, "mosaickv_exact", row)
    control.execute(*args)
    assert seen[-1]["graph_accounting"] == "event_sum"
    assert seen[-1]["lt_fit_policy"] == "direct" and "lt_fit_policy" not in row
    proof["hits"] -= 1
    with pytest.raises(RuntimeError, match="31x126"):
        control.execute(*args)
