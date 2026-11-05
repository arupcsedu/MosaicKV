import copy
import json

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
        history_borrow_maximum_bytes=3670016,
        maximum_workspace_bytes=5955584,
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


@pytest.mark.parametrize(
    "capacity,borrows,misses,maximum",
    [
        (5955584, 192, 0, 3670016),
        (8192 * 128, 32, 160, 1048576),
        (8192 * 256, 96, 96, 2097152),
        (8192 * 448, 192, 0, 3670016),
    ],
)
def test_arena_fallback_is_required_only_when_physical_capacity_cannot_fit(
    capacity, borrows, misses, maximum
):
    assert controller.expected_arena(capacity) == dict(
        borrows=borrows, misses=misses, maximum=maximum
    )


@pytest.mark.parametrize(
    "changed",
    [
        ["mosaickv/src/mosaickv/evaluation/prompt_graph_reader.py"],
        ["mosaickv/src/mosaickv/evaluation/shared_source_kernel.py"],
        ["AGENTS.md"],
    ],
)
def test_positive_control_reuse_rejects_any_runtime_or_algorithm_source_change(changed):
    with pytest.raises(ValueError, match="runtime/algorithm"):
        controller.compatible_source("new", changed=changed)
    controller.compatible_source("new", changed=sorted(controller.APPROVED_GUARD_PATHS))


@pytest.mark.parametrize("fault", [None, "bytes", "policy", "parity", "failure_kind"])
def test_control_reuse_accepts_only_complete_hash_bound_identical_positive_controls(
    tmp_path, monkeypatch, fault
):
    monkeypatch.setattr(controller, "RESUME", tmp_path)
    worker = tmp_path / "workers/0"
    worker.mkdir(parents=True)
    record = dict(
        prompt_graph_reader=False,
        packet_sha256="a" * 64,
        audited=dict(logits_bitwise=True, no_FullKV_fallback=True),
    )
    if fault == "policy":
        record["prompt_graph_reader"] = True
    if fault == "parity":
        record["audited"]["logits_bitwise"] = False
    name = "control-mosaickv_exact-direct.json"
    path = worker / name
    path.write_text(json.dumps(record))
    result = dict(
        source_sha=controller.RESUME_SHA,
        status="failed",
        failures=["RuntimeError: complete charged graph/arena policy engagement required"],
        outputs={name: controller.sha(path)},
    )
    if fault == "bytes":
        path.write_text(path.read_text() + " ")
    if fault == "failure_kind":
        result["failures"] = ["unrelated numerical error"]
    (worker / "result.json").write_text(json.dumps(result))
    expected = dict(mosaickv_exact=dict(direct=dict(prompt_graph_reader=False)))
    if fault is None:
        controls = controller.reused_controls(0, expected)
        assert set(controls) == {("mosaickv_exact", "direct")}
        assert controls["mosaickv_exact", "direct"][1]["source_sha"] == controller.RESUME_SHA
    else:
        with pytest.raises(ValueError):
            controller.reused_controls(0, expected)
