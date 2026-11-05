import evicastkv_prepared_history_dispatch as control
import pytest


def test_every_baseline_reader_receives_same_optional_optimization():
    original = {
        "mosaickv_exact": {"graph": {}, "graph-lt": {"lt_source_reader": True}},
        "prefixkv_reimpl": {"graph": {}, "graph-original": {}},
    }
    variants = control.variants(original)
    assert len(control.ARMS) == 8 and len(variants) == 4
    assert sum(method == "prefixkv_reimpl" for method, _, _ in variants) == 2
    assert all(row["reuse_patch_dispatch"] for _, _, row in variants)
    assert all(
        "reuse_patch_dispatch" not in row for group in original.values() for row in group.values()
    )


def test_screen_checks_all_optimized_controls_without_claiming_acceptance():
    rows = [
        dict(
            method=method,
            arm=arm,
            trial=t,
            warmup=t < 0,
            measurement=dict(
                total_wall_seconds=0.9 if method == "mosaickv_exact" else 1.0,
                logits_bitwise=True,
                no_FullKV_fallback=True,
            ),
        )
        for method, arm in control.ARMS
        for t in range(-5, 3)
    ]
    result = control.summary(rows)
    assert result["four_percent_POINT_only"] and not result["formal_latency_acceptance"]
    assert len(result["candidate_mean_ratios_vs_all_PrefixKV"]) == 2
    with pytest.raises(ValueError):
        control.summary(rows[:-1])


def test_actual_reuse_and_no_additional_tensor_owners_required(monkeypatch):
    proof = dict(
        enabled=True, hits=6600, misses=448, tensor_owners_retained=0, GPU_workspace_bytes=0
    )
    configured = []

    def execute(*args, **kwargs):
        configured.append(args[-1])
        return dict(prepared_patch_dispatch=proof)

    monkeypatch.setattr(control, "ORIGINAL_EXECUTE", execute)
    record = dict(reuse_patch_dispatch=True, lt_source_reader=True)
    args = (None, None, None, None, None, "mosaickv_exact", record)
    control.execute(*args)
    assert configured[-1]["graph_accounting"] == "event_sum"
    assert configured[-1]["lt_fit_policy"] == "direct"
    assert "graph_accounting" not in record
    proof["tensor_owners_retained"] = 1
    with pytest.raises(RuntimeError, match="owners"):
        control.execute(*args)


@pytest.mark.parametrize("fault", [None, "incomplete", "source", "plan", "failures"])
def test_preceding_GPU_screen_must_complete_from_registered_source(monkeypatch, tmp_path, fault):
    plan = dict(predecessor_source_sha="source", predecessor_plan_sha256="plan")
    gate = dict(
        complete=fault != "incomplete",
        failures=["failed"] if fault == "failures" else [],
        source_sha="foreign" if fault == "source" else "source",
        plan_sha256="foreign" if fault == "plan" else "plan",
    )
    monkeypatch.setattr(
        control.controller.driver, "read", lambda path: plan if path.name == "plan.json" else gate
    )
    monkeypatch.setattr(control, "sha", lambda path: "hash")
    if fault:
        with pytest.raises(ValueError, match="preceding"):
            control.review_predecessor(tmp_path)
    else:
        assert control.review_predecessor(tmp_path)["sha256"] == "hash"
