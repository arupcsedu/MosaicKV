from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_review_parallel as controller
import pytest


def test_one_array_encodes_every_lane_and_leaves_blocked_lanes_out():
    plan = {
        name: dict(
            status="prepared", workers=2, root=f"/scratch/djy8hg/test/{name}", spec_sha="a" * 64
        )
        for name in controller.LANES
    }
    plan["source"]["status"] = "blocked_preparation"
    tasks = controller.array_tasks(plan)
    assert len(tasks) == 8
    assert {t["lane"] for t in tasks} == set(controller.LANES) - {"source"}
    assert all(t["module"] == controller.LANES[t["lane"]] for t in tasks)


@pytest.mark.parametrize("count", [0, -1, 17, True, 1.5])
def test_reject_unbounded_or_ambiguous_worker_counts(count):
    with pytest.raises(ValueError):
        controller.array_tasks(dict(terms=dict(status="prepared", workers=count)))


@pytest.mark.parametrize("identity", ["1_2", "a", "123:456", "", "1;bad\n9"])
def test_bad_job_ids(identity):
    with pytest.raises(ValueError):
        controller.job_id(identity)


def test_gpu_submission_has_one_gpu_twelve_cpus_two_hours_and_global_six(tmp_path):
    controller.write_once(tmp_path / "tasks.json", dict(tasks=[]))
    with patch.object(controller.subprocess, "check_output", return_value="12345\n") as call:
        result = controller.submit(tmp_path, "gpu", "worker", ["100"], array="0-12%6")
    command = call.call_args.args[0]
    assert result == "12345"
    for argument in (
        "--array=0-12%6",
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--time=02:00:00",
        "--reservation=bi_fox_dgx",
        "--dependency=afterany:100",
    ):
        assert argument in command
    with pytest.raises(FileExistsError):
        controller.submit(tmp_path, "gpu", "worker", array="0-12%6")


@pytest.mark.parametrize("array", ["0-2%8", "0-2%4", "1-2%6", "0-4", "0-2%6;echo"])
def test_array_throttle_cannot_be_accidentally_broadened(tmp_path, array):
    with pytest.raises(ValueError):
        controller.submit(tmp_path, "bad", "worker", array=array)


def test_scratch_output_constraint():
    with pytest.raises(ValueError):
        controller.safe_root(Path("/tmp/not-authorized"))


def test_ownership_inventory_flattens_without_cloning_or_retaining_extra_buffers():
    from evicastkv_review_lifetimes import flatten_pairs

    a, b, c = object(), object(), object()
    assert flatten_pairs(((a, b), (c, a))) == (a, b, c, a)


def test_pooling_factorial_uses_variant_namespaced_warmups():
    import evicastkv_pooling_index_study as study

    assert study.STAGE == "optimization"
    namespaces = [f"{method}-{cap}-{variant}" for method, cap, variant in study.ARMS]
    assert len(set(namespaces)) == len(study.ARMS)


def test_pooling_orders_balance_each_implementation_pair_over_frozen_grid():
    import evicastkv_pooling_index_study as study

    arms = [tuple(a) for a in study.ARMS]
    before = {method: 0 for method in ("prompt_attention_topk", "mosaickv_exact")}
    for sample in range(4):
        for repeat in range(3):
            order = study.balanced_order(arms, sample, repeat)
            assert set(order) == set(arms)
            for method in before:
                before[method] += order.index((method, 40, "pool-reference")) < order.index(
                    (method, 40, "pool-index")
                )
    assert set(before.values()) == {6}


def test_evaluation_pooling_overlay_does_not_mutate_runtime_source():
    import mosaickv.graph.builder as builder
    from mosaickv.evaluation.fox_pooling_index import pooling_index

    original = builder.pool_prompt_attention_coactivation
    with pytest.raises(RuntimeError), pooling_index():
        raise RuntimeError("diagnostic failure")
    assert builder.pool_prompt_attention_coactivation is original


def test_preparation_uses_one_thirteen_task_array_not_five_separate_arrays(tmp_path):
    registration = dict(
        source_sha="a" * 40,
        parent="/scratch/djy8hg/parent",
        mechanisms="/scratch/djy8hg/mechanisms",
        prior="/scratch/djy8hg/prior",
    )
    counts = dict(boundary=1, terms=4, pooling=4, ownership=2, source=2)

    def preparer(lane):
        def prepare(*args, **kwargs):
            target = args[1] if lane == "boundary" else args[0]
            name = (
                "plan.json"
                if lane == "source"
                else "protocol.json"
                if lane == "boundary"
                else "protocol_lock.json"
            )
            controller.write_once(target / name, dict(scope="mock"))
            return dict(workers=counts[lane], spec_sha256="b" * 64)

        return prepare

    modules = {
        module: SimpleNamespace(prepare=preparer(lane)) for lane, module in controller.LANES.items()
    }
    with (
        patch.object(controller, "load", return_value=registration),
        patch.object(controller, "cpu_gate", return_value=dict(status="passed")),
        patch.object(controller.importlib, "import_module", side_effect=lambda name: modules[name]),
        patch.object(
            controller, "gpu_snapshot", return_value=dict(active=[], pending=[], active_gpus=0)
        ),
        patch.object(controller, "submit", return_value="123") as submissions,
    ):
        result = controller.prepare(tmp_path)
    assert result["tasks"] == 13
    calls = submissions.call_args_list
    gpu_calls = [call for call in calls if call.kwargs.get("array")]
    assert len(gpu_calls) == 1
    assert gpu_calls[0].kwargs["array"] == "0-12%6"
    assert [call.args[1] for call in calls] == ["freshness", "gpu", "analyze"]


def test_worker_refuses_excess_aggregate_gpu_occupancy_before_model_import(tmp_path):
    with (
        patch.object(controller, "load", return_value={}),
        patch.object(controller, "gpu_snapshot", return_value=dict(active_gpus=7)),
        patch.object(controller.importlib, "import_module") as imports,
        pytest.raises(RuntimeError, match="exceeds six"),
    ):
        controller.worker(tmp_path, 0)
    imports.assert_not_called()
