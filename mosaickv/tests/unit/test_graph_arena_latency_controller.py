import copy

import evicastkv_graph_arena_latency as controller
import pytest


def fixture(ratio=0.95, final=False):
    return [
        dict(
            method=m,
            arm=a,
            trial=t,
            warmup=t < 0,
            measurement=dict(
                total_wall_seconds=ratio if (m, a) == ("mosaickv_exact", "graph") else 1.0,
                logits_bitwise=True,
                no_FullKV_fallback=True,
            ),
        )
        for m, a in controller.required(final)
        for t in range(-5, 20)
    ]


def test_retains_all_baseline_readers_and_rejects_any_faster_control():
    rows = fixture()
    assert controller.summarize(rows, 3)["four_percent_latency_gate"]
    for arm in (
        "direct",
        "graph-original",
        "graph",
        "whole-6291456",
        "whole-graph-original",
        "whole-graph",
    ):
        bad = copy.deepcopy(rows)
        for r in bad:
            if (r["method"], r["arm"]) == ("prefixkv_reimpl", arm):
                r["measurement"]["total_wall_seconds"] = 0.90
        assert not controller.summarize(bad, 3)["four_percent_latency_gate"]


@pytest.mark.parametrize(
    "ratio,passed", [(0.95, True), (0.961, False), (1.0, False), (1.04, False)]
)
def test_only_adjusted_four_percent_is_a_pass(ratio, passed):
    assert controller.summarize(fixture(ratio), 3)["four_percent_latency_gate"] is passed


def test_complete_five_method_extension_and_unknown_arm_rejection():
    assert controller.summarize(fixture(final=True), 3, final=True)["four_percent_latency_gate"]
    with pytest.raises(ValueError):
        controller.summarize(fixture(), 3, final=True)
    rows = fixture()
    rows[0]["arm"] = "arbitrary"
    with pytest.raises(ValueError):
        controller.summarize(rows, 3)


def test_policy_is_common_and_source_parent_unmodified():
    parent = dict(
        records={m: dict(direct={}, **{"whole-6291456": {}}) for m in controller.previous.METHODS}
    )
    original = copy.deepcopy(parent)
    records = controller.make_records(parent, controller.previous.METHODS)
    assert original == parent
    for arms in records.values():
        assert "prompt_graph_policy" not in arms["graph-original"]
        assert arms["graph"]["prompt_graph_policy"] == controller.POLICY
        assert arms["graph"]["prompt_graph_policy"]["row_major_codes"] is False
    assert records["prefixkv_reimpl"]["whole-graph"]["prompt_graph_policy"] == controller.POLICY


@pytest.mark.parametrize("fault", [None, "borrow", "miss", "row4", "codes", "retired"])
def test_new_policy_must_actually_engage_without_enabling_negative_layout(monkeypatch, fault):
    info = dict(
        optimized_policy=controller.POLICY,
        history_borrows=192,
        history_borrow_misses=0,
        history_borrow_maximum_bytes=1024,
        row4_workspace_peak_bytes=0,
        row4_replays=0,
        retired=True,
        captured_code_replays=4064,
        captured_code_kernel_dispatches=3937,
    )
    if fault is not None:
        key, value = {
            "borrow": ("history_borrows", 0),
            "miss": ("history_borrow_misses", 1),
            "row4": ("row4_replays", 1),
            "codes": ("captured_code_kernel_dispatches", 0),
            "retired": ("retired", False),
        }[fault]
        info[key] = value
    monkeypatch.setattr(
        controller, "ORIGINAL_EXECUTE", lambda *a, **k: dict(prompt_graph_reader=info)
    )

    def invoke():
        return controller.execute(
            None,
            None,
            None,
            None,
            dict(code_delta_methods=["mosaickv_exact"]),
            "mosaickv_exact",
            dict(prompt_graph_policy=controller.POLICY),
        )

    if fault is None:
        assert invoke()["prompt_graph_reader"] is info
    else:
        with pytest.raises(RuntimeError):
            invoke()
