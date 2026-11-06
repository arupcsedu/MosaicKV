"""Synthetic controls; no checkpoints or dataset downloads."""

from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import direct_text_construction as direct
from mosaickv.evaluation.paper_geometry import (
    enabled,
    odd_length_allowed,
    registered_geometry,
    source_shape_allowed,
)
from mosaickv.evaluation.streaming_baselines import source_capacities


def test_geometry_extension_is_explicit_nonnested_and_restored():
    assert not source_shape_allowed((1, 32, 603, 128))
    assert source_shape_allowed((1, 32, 600, 128))
    with registered_geometry(603):
        assert enabled() and odd_length_allowed(603)
        assert source_shape_allowed((1, 32, 603, 128))
        assert not source_shape_allowed((2, 32, 603, 128))
        assert not source_shape_allowed((1, 32, 605, 128))
        with pytest.raises(ValueError, match="isolated"), registered_geometry(603):
            pass
    assert not enabled() and not odd_length_allowed(603)
    with pytest.raises(ValueError), registered_geometry(605):
        pass


def test_odd_capacity_uses_exact_global_half_slots_no_prompt_padding():
    from mosaickv.selection.profile_lifetime import allocate_profile_lifetime

    profile = (0.5,) * 32
    with pytest.raises(NotImplementedError):
        source_capacities(profile, 603)
    with registered_geometry(603):
        capacity = source_capacities(profile, 603)
        expected = allocate_profile_lifetime(
            profile,
            (603,) * 32,
            (1,) * 32,
            (16384,) * 32,
            cap_bytes=603 * 32 * 16384 // 2,
            generated_capacity=0,
        )
        assert capacity == expected.prompt_positions
        assert expected.charged_ceiling_bytes <= expected.cap_bytes
        assert not expected.largest_single_increment_feasible
        assert all(1 <= n <= 603 for n in capacity)
    assert max(source_capacities(profile, 604)) <= 604


def test_dynamic_accounting_cannot_escape_scope_or_allowance():
    policy = dict(
        modes=[["source", "source"]],
        maximum_added_bytes=0,
        accounting_mode="request_local_base_plus_allowance",
    )
    packet = SimpleNamespace(private=lambda: (torch.ones(8),))
    with pytest.raises(ValueError, match="bounded"):
        direct.Construction(policy)
    with registered_geometry(603):
        state = direct.Construction(policy)
        actual, _ = state.optimize(packet, (), (), (), None)
        assert state.finish((actual,))["private_bytes"] == 32
        too_large = SimpleNamespace(private=lambda: (torch.ones(9),))
        with pytest.raises(RuntimeError, match="allowance"):
            state.finish((too_large,))
        with pytest.raises(ValueError):
            direct.Construction(dict(policy, answer="label"))


def test_id_ordering_uses_only_identity_and_seed():
    from evicastkv_paper_precision import select_ids

    rows = [("a", "ref-do-not-read", 1), ("b", "other", 2), ("c", "third", 3)]
    first = select_ids(rows, {"b"}, 2, 42)
    assert {r[0] for r in first} == {"a", "c"}
    assert select_ids(list(reversed(rows)), {"b"}, 2, 42) == first
    with pytest.raises(ValueError, match="duplicate"):
        select_ids([*rows, rows[0]], set(), 1, 42)
    with pytest.raises(ValueError, match="insufficient"):
        select_ids(rows, {"a", "b", "c"}, 1, 42)


def test_paired_timing_rejects_fullkv_duplicates_and_missing_rows():
    from evicastkv_paper_precision import timing_comparisons

    rows = [
        dict(sample_index=s, method=m, trial=t, end_to_end_seconds=v)
        for s in range(6)
        for m, v in (("mosaickv_exact", 0.9), ("prefixkv_reimpl", 1.0))
        for t in range(20)
    ]
    result = timing_comparisons(rows, 42)
    assert result[0]["mean_ratio"] == pytest.approx(0.9)
    assert result[0]["adjusted_ratio_interval"] == pytest.approx([0.9, 0.9])
    with pytest.raises(ValueError, match="complete"):
        timing_comparisons(rows[:-1], 42)
    with pytest.raises(ValueError, match="complete"):
        timing_comparisons(rows + rows[:1], 42)
    with pytest.raises(ValueError, match="FullKV"):
        timing_comparisons(
            [dict(r, method="full_kv") if r["method"] == "prefixkv_reimpl" else r for r in rows], 42
        )


def test_request_options_copy_frozen_math_not_saved_bytes():
    from evicastkv_paper_precision import request_options

    policy = dict(
        modes=[["source", "source"]] * 32,
        maximum_added_bytes=6 * 2**20,
        base_private_bytes=123,
        expected_private_bytes=456,
    )
    record = dict(
        direct_policy=policy,
        uniform_attention_core=True,
        packet_sha256="historical-only",
        source_tile_rows=192,
        joint_incoming_projection="kv",
        verified_constructor="async_handoff",
    )
    plan = dict(runtime_parent=dict(code_delta_methods=["mosaickv_exact"]))
    options = request_options(record, plan, "mosaickv_exact", False)
    assert "base_private_bytes" not in options["direct_text_policy"]
    assert "expected_private_bytes" not in options["direct_text_policy"]
    assert options["direct_text_policy"]["modes"] == policy["modes"]
    assert options["source_tile_rows"] == 192 and options["code_delta"]
    assert "packet_sha256" not in options


def test_real_launch_plan_recovery_without_gpu_or_submission(tmp_path, monkeypatch):
    import evicastkv_paper_precision as campaign

    monkeypatch.setattr(campaign, "clean_source", lambda *args: "a" * 40)
    monkeypatch.setattr(campaign, "submit", lambda *args, **kwargs: "CPU_NOT_SUBMITTED")
    root = tmp_path / "prepare"
    campaign.prepare(root)
    plan = campaign.read(root / "plan.json")
    assert set(plan["options"]) == set(plan["protocol"]["methods"]) - {"full_kv"}
    for method, record in plan["options"].items():
        options = campaign.request_options(record, plan, method, True)
        assert options["uniform_attention_core"]
        assert options["componentwise_fitting"]
        assert len(options["direct_text_policy"]["modes"]) == 32
    source = plan["options"]["mosaickv_exact"]
    assert source["source_tile_rows"] == 192
    assert source["joint_incoming_projection"] == "kv"
    assert source["reuse_source_input_copy"] is False
    assert source["verified_constructor"] == "async_handoff"


def test_only_our_new_held_array_is_adopted(tmp_path, monkeypatch):
    import evicastkv_paper_precision as campaign

    receipts = iter(("900", "901", "902"))
    monkeypatch.setattr(campaign, "submit", lambda *args, **kwargs: next(receipts))
    campaign.launch_phase(tmp_path, "canary", [dict(method="full_kv", samples=[0])])
    policy = campaign.read(tmp_path / "canary/scheduler/policy.json")
    assert policy["adopt_existing_holds"] == policy["arrays"] == ["900"]
    assert policy["maximum_a100s"] == 6


def test_ledger_reuse_is_bound_and_policy_checked(tmp_path, monkeypatch):
    import evicastkv_paper_precision as campaign

    monkeypatch.setattr(campaign, "clean_source", lambda *args: "a" * 40)
    monkeypatch.setattr(campaign, "submit", lambda *args, **kwargs: "CPU_NOT_SUBMITTED")
    parent = tmp_path / "parent"
    campaign.prepare(parent)
    campaign.write_once(parent / "ledger.json", dict(samples=[]))
    campaign.write_once(
        parent / "ledger_seal.json", dict(ledger_sha256=campaign.sha(parent / "ledger.json"))
    )
    retry = tmp_path / "retry"
    campaign.prepare(retry, parent)
    plan = campaign.read(retry / "plan.json")
    assert plan["reused_ledger_parent"] == str(parent)
    assert plan["bound_files"][str(parent / "ledger.json")] == campaign.sha(parent / "ledger.json")
    changed = dict(campaign.read(campaign.CONFIG), output_tokens=64)
    path = tmp_path / "changed.json"
    campaign.write_once(path, changed)
    monkeypatch.setattr(campaign, "CONFIG", path)
    with pytest.raises(ValueError, match="identical registered"):
        campaign.prepare(tmp_path / "reject", parent)


def test_failed_producer_rows_remain_in_attempted_denominator(tmp_path, monkeypatch):
    import json

    import evicastkv_paper_precision as campaign

    plan = dict(source_sha="a" * 40, protocol=dict(methods=["full_kv", "mosaickv_exact"]))
    monkeypatch.setattr(campaign, "checked", lambda *args: plan)
    jobs = [dict(method=m, samples=[0]) for m in plan["protocol"]["methods"]]
    campaign.write_once(tmp_path / "quality/jobs.json", jobs)
    for i, job in enumerate(jobs):
        folder = tmp_path / f"quality/workers/{i}"
        campaign.write_once(folder / "manifest.json", dict(synthetic_test_fixture=True))
        row = dict(
            sample_index=0,
            sample_id="toy",
            method=job["method"],
            trial=0,
            warmup=False,
            status="completed" if i == 0 else "failed",
            generated_token_ids=[1] * 128 if i == 0 else [],
            used_fullkv_fallback=False,
        )
        (folder / "raw.jsonl").write_text(json.dumps(row) + "\n")
        campaign.write_once(
            folder / "result.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256="p",
                job=job,
                status=row["status"],
                failures=[] if i == 0 else ["synthetic numerical failure"],
                raw_sha256=campaign.sha(folder / "raw.jsonl"),
                manifest_sha256=campaign.sha(folder / "manifest.json"),
            ),
        )
    campaign.analyze(tmp_path, "p", "quality")
    decision = campaign.read(tmp_path / "quality/analysis/decision.json")
    assert not decision["complete"]
    assert decision["attempted"] == decision["expected"] == 2
    assert decision["successful"] == decision["failed_or_missing"] == 1


def test_registered_tail_ranges_and_frozen_plan_frontend(monkeypatch):
    from mosaickv.evaluation.frozen_joint_plan import FrozenKernelPlans, Plans
    from mosaickv.evaluation.source_192_feasibility import ranges

    with pytest.raises(ValueError):
        ranges(603)
    with pytest.raises(NotImplementedError):
        FrozenKernelPlans((128, 91, 92))

    class ReachedLibraryBoundary(Exception):
        pass

    def stop_before_cuda(*args):
        raise ReachedLibraryBoundary

    monkeypatch.setattr(Plans, "__init__", stop_before_cuda)
    with registered_geometry(603):
        assert ranges(603) == ((0, 192), (192, 192), (384, 192), (576, 27))
        assert ranges(603, 128)[-1] == (512, 91)
        with pytest.raises(ReachedLibraryBoundary):
            FrozenKernelPlans((128, 91, 92))
        with pytest.raises(NotImplementedError):
            FrozenKernelPlans((128, 90, 91))
        with pytest.raises(ValueError):
            ranges(605)


def test_geometry_crosses_worker_boundary_without_scope_or_owner_leaks():
    from concurrent.futures import ThreadPoolExecutor

    from mosaickv.evaluation.paper_geometry import current_length, inherit_worker_geometry
    from mosaickv.evaluation.source_192_feasibility import ranges

    @inherit_worker_geometry
    def task(owner):
        return ranges(603), current_length()

    with registered_geometry(603):
        owner = SimpleNamespace(paper_prompt_length=current_length())
    assert not enabled()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result, length = executor.submit(task, owner).result()
        assert result[-1] == (576, 27) and length == 603
        assert executor.submit(enabled).result() is False
        with pytest.raises(ValueError):
            executor.submit(task, SimpleNamespace(paper_prompt_length=None)).result()
        assert executor.submit(enabled).result() is False
