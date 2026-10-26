"""CPU-only plan/schema checks; no data access, jobs, or model downloads."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from mosaickv.config import BoundedDecodeConfig, RunConfig, canonical_config, config_sha256


def tooling():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "bounded_pilot", root / "mosaickv/scripts/bounded_history_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submission_tooling():
    import sys

    module = tooling()
    spec = importlib.util.spec_from_file_location(
        "bounded_submit_v3", module.SOURCE / "mosaickv/scripts/submit_bounded_history_memory_v3.py"
    )
    submitted = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("bounded_history_pilot")
    sys.modules["bounded_history_pilot"] = module
    try:
        spec.loader.exec_module(submitted)
    finally:
        if previous is None:
            sys.modules.pop("bounded_history_pilot", None)
        else:
            sys.modules["bounded_history_pilot"] = previous
    return submitted


@pytest.mark.parametrize("task", ["llava_description", "mmvet_prefixkv"])
@pytest.mark.parametrize("cap", [40, 80])
def test_all_core_baselines_receive_same_lifetime_cap_and_generation(task, cap, tmp_path):
    module = tooling()
    route = {"dataset_id": "synthetic/native", "dataset_revision": "b" * 40, "split": "test"}
    configs = module.configurations(task, cap, route, tmp_path / "profile.json")
    reference = configs.pop("full_kv")
    assert reference.bounded_decode is None
    assert reference.cache.retention_ratio == 1.0
    for cfg in configs.values():
        assert cfg.cache.budget_value == cap * 2**20
        assert cfg.bounded_decode == BoundedDecodeConfig(32, 4)
        assert cfg.generation == reference.generation
        assert cfg.execution == reference.execution
        assert cfg.model == reference.model
        assert cfg.dataset == reference.dataset
        assert (
            cfg.cache.physical_budget_mandatory_policy
            == reference.cache.physical_budget_mandatory_policy
        )
        assert config_sha256(RunConfig.from_mapping(canonical_config(cfg))) == config_sha256(cfg)
    assert configs["prefixkv_reimpl"].prefixkv.decode_policy == "append_only"
    assert configs["prefixkv_reimpl"].prefixkv.hf_cache_transport == "ragged_llama_eager_v1"


def test_pilot_resources_and_scope_are_explicit():
    module = tooling()
    protocol = yaml.safe_load(module.PROTOCOL.read_text())
    assert protocol["max_global_a100s"] == 4
    assert protocol["cpus_per_worker"] == 12
    assert protocol["time_limit_seconds"] == 7200
    assert protocol["paper_campaign_authorized"] is False
    assert protocol["heldout_claim"] is False
    assert protocol["cross_budget_direct_comparisons"] == "forbidden"


def test_numpy_scalars_serialize_without_leaving_partial_invalid_files(tmp_path):
    import json

    import numpy as np

    module = tooling()
    path = tmp_path / "valid.json"
    module.write_once(path, dict(passed=np.bool_(False), n=np.int64(8), metric=np.float32(0.5)))
    assert json.loads(path.read_text()) == dict(passed=False, n=8, metric=0.5)
    with pytest.raises(FileExistsError):
        module.write_once(path, dict(passed=True))
    for value in (float("nan"), object()):
        failed = tmp_path / "invalid.json"
        with pytest.raises((ValueError, TypeError)):
            module.write_once(failed, dict(value=value))
        assert not failed.exists()


@pytest.mark.parametrize("version,offset", [(5, 22), (6, 30)])
def test_draft_protocol_is_disjoint_bounded_and_not_a_paper_promotion(version, offset):
    import copy

    module = tooling()
    submit = submission_tooling()
    protocol = yaml.safe_load(
        (
            module.SOURCE / f"mosaickv/configs/experiments/bounded_history_memory_v{version}.yaml"
        ).read_text()
    )
    assert protocol["caps_mib"] == [40] and protocol["sample_offset"] == offset
    assert all(len(protocol["development_exclusions"][t]) == version - 1 for t in protocol["tasks"])
    assert protocol["draft_guard"]["checkpoints"] == [1, 4, 16, 32]
    jobs = []
    for task in protocol["tasks"]:
        jobs.extend(
            dict(index=len(jobs) + i, method=method, task=task)
            for i, method in enumerate(["full_kv", *protocol["methods"]])
        )
    plan = dict(
        protocol=protocol,
        jobs=jobs,
        source_sha="a" * 40,
        bound_files={"fixture-cpu.json": "b" * 64},
        development_disjointness=[
            dict(
                task=t,
                selected_count=8,
                excluded_count=offset,
                overlap_count=0,
                selection_uses_answers_or_scores=False,
            )
            for t in protocol["tasks"]
        ],
        cpu_feasibility=dict(
            passed=True,
            source_sha="a" * 40,
            path="fixture-cpu.json",
            sha256="b" * 64,
            classification="synthetic_CPU_feasibility_not_pretrained_quality_evidence",
            **(
                dict(
                    policy="bounded_single_equal_byte_draft_logit_v6",
                    exact_bytes_and_mandatory=True,
                )
                if version == 6
                else {}
            ),
        ),
    )
    refs, compressed = submit.validate_scope(plan)
    assert len(refs) == 2 and len(compressed) == 6
    altered = copy.deepcopy(plan)
    altered["protocol"]["draft_guard"]["horizon"] = 4
    with pytest.raises(ValueError, match="frozen guard"):
        submit.validate_scope(altered)
    altered = copy.deepcopy(plan)
    altered["protocol"]["caps_mib"] = [32, 40, 80]
    with pytest.raises(ValueError, match="limited"):
        submit.validate_scope(altered)
    summaries = [
        dict(
            task=t,
            method=m,
            samples=8,
            cap_mib=40,
            rouge_l_f1=0.5,
            corpus_ppl=4.0,
            draft_guard_acceptance_fraction=1.0,
        )
        for t in protocol["tasks"]
        for m in (protocol["candidate_arm"], "prompt_attention_topk", "mosaickv_exact")
    ]
    screen = module.v5_preliminary_screen(summaries, protocol["tasks"], protocol["candidate_arm"])
    assert screen["preliminary_point_estimate_screen_passed"]
    assert not screen["frozen_candidate_eligible"] and not screen["paper_campaign_authorized"]
    summaries[0]["corpus_ppl"] = 5.0
    assert not module.v5_preliminary_screen(
        summaries, protocol["tasks"], protocol["candidate_arm"]
    )["preliminary_point_estimate_screen_passed"]


def test_memory_successor_has_a_common_grid_and_no_automatic_paper_promotion():
    module = tooling()
    protocol = yaml.safe_load(
        (module.SOURCE / "mosaickv/configs/experiments/bounded_history_memory_v2.yaml").read_text()
    )
    assert protocol["caps_mib"] == [32, 40, 80]
    assert protocol["max_global_a100s"] == 6
    assert protocol["sample_offset"] == 2
    assert protocol["samples_per_task"] == 4
    assert protocol["history_fraction"] == 0.2
    assert len(protocol["methods"]) == 6
    assert not protocol["promotion"]["automatic_paper_campaign"]
    assert protocol["cross_budget_direct_comparisons"] == "forbidden"


def test_guarded_successor_keeps_controls_and_requires_disjoint_development_ids():
    module = tooling()
    protocol = yaml.safe_load(
        (module.SOURCE / "mosaickv/configs/experiments/bounded_history_memory_v3.yaml").read_text()
    )
    assert protocol["caps_mib"] == [32, 40, 80]
    assert protocol["samples_per_task"] == 8
    assert protocol["sample_offset"] == 6
    assert protocol["history_fraction"] == 0.2
    assert protocol["output_tokens"] == 1024
    assert protocol["max_global_a100s"] == 6
    assert len(protocol["methods"]) == 4
    assert not protocol["promotion"]["automatic_paper_campaign"]
    assert not protocol["promotion"]["powered_validation_authorized"]
    for task in protocol["tasks"]:
        assert len(protocol["development_exclusions"][task]) == 2
        assert all(len(row["sha256"]) == 64 for row in protocol["development_exclusions"][task])


def test_direct_error_successor_is_small_disjoint_and_has_unchanged_numerical_guard():
    from mosaickv.selection.direct_error_exchange import MAX_EXCHANGES, MAX_VERIFIED_PAIRS
    from mosaickv.selection.evidence_layer_allocation import (
        RECONSTRUCTION_ATOL,
        RECONSTRUCTION_RTOL,
    )

    module = tooling()
    p = yaml.safe_load(
        (module.SOURCE / "mosaickv/configs/experiments/bounded_history_memory_v4.yaml").read_text()
    )
    assert p["caps_mib"] == [32, 40, 80]
    assert p["sample_offset"] == 14
    assert p["samples_per_task"] == 8
    assert len(p["methods"]) == 3
    assert p["cpu_feasibility_required"]
    assert p["candidate_allocation"]["max_exchanges"] == MAX_EXCHANGES
    assert p["candidate_allocation"]["max_verified_pairs"] == MAX_VERIFIED_PAIRS
    assert p["candidate_allocation"]["absolute_tolerance"] == RECONSTRUCTION_ATOL
    assert p["candidate_allocation"]["relative_tolerance"] == RECONSTRUCTION_RTOL
    assert not p["paper_campaign_authorized"]
    assert not p["promotion"]["powered_validation_authorized"]
    assert all(len(p["development_exclusions"][task]) == 3 for task in p["tasks"])


def test_direct_error_launcher_requires_clean_source_bound_positive_cpu_feasibility():
    module = tooling()
    submit = submission_tooling()
    p = yaml.safe_load(
        (module.SOURCE / "mosaickv/configs/experiments/bounded_history_memory_v4.yaml").read_text()
    )
    jobs = []
    for task in p["tasks"]:
        jobs.append(dict(index=len(jobs), method="full_kv", task=task))
        for cap in p["caps_mib"]:
            for method in p["methods"]:
                jobs.append(dict(index=len(jobs), method=method, task=task, cap_mib=cap))
    plan = dict(
        protocol=p,
        jobs=jobs,
        source_sha="a" * 40,
        bound_files={"synthetic-cpu.json": "b" * 64},
        development_disjointness=[
            dict(
                task=t,
                selected_count=8,
                excluded_count=14,
                overlap_count=0,
                selection_uses_answers_or_scores=False,
            )
            for t in p["tasks"]
        ],
    )
    with pytest.raises(ValueError, match="CPU feasibility"):
        submit.validate_scope(plan)
    plan["cpu_feasibility"] = dict(
        passed=True,
        source_sha="a" * 40,
        path="synthetic-cpu.json",
        sha256="b" * 64,
        classification="synthetic_CPU_feasibility_not_pretrained_quality_evidence",
    )
    refs, compressed = submit.validate_scope(plan)
    assert len(refs) == 2 and len(compressed) == 18
    plan["cpu_feasibility"]["source_sha"] = "c" * 40
    with pytest.raises(ValueError, match="CPU feasibility"):
        submit.validate_scope(plan)


def test_guarded_launcher_refuses_campaign_expansion_or_missing_disjointness():
    import copy

    module = tooling()
    submit = submission_tooling()
    protocol = yaml.safe_load(
        (module.SOURCE / "mosaickv/configs/experiments/bounded_history_memory_v3.yaml").read_text()
    )
    jobs = []
    for task in protocol["tasks"]:
        jobs.append(dict(index=len(jobs), method="full_kv", task=task))
        for cap in protocol["caps_mib"]:
            for method in protocol["methods"]:
                jobs.append(dict(index=len(jobs), method=method, task=task, cap_mib=cap))
    plan = dict(
        protocol=protocol,
        jobs=jobs,
        development_disjointness=[
            dict(
                task=task,
                selected_count=8,
                excluded_count=6,
                overlap_count=0,
                selection_uses_answers_or_scores=False,
            )
            for task in protocol["tasks"]
        ],
    )
    references, compressed = submit.validate_scope(plan)
    assert len(references) == 2
    assert len(compressed) == 24
    for field, value in [
        ("samples_per_task", 128),
        ("paper_campaign_authorized", True),
        ("caps_mib", [10, 20, 40, 80]),
        ("max_global_a100s", 8),
    ]:
        expanded = copy.deepcopy(plan)
        expanded["protocol"][field] = value
        with pytest.raises(ValueError, match="limited"):
            submit.validate_scope(expanded)
    plan["development_disjointness"][0]["overlap_count"] = 1
    with pytest.raises(ValueError, match="disjointness"):
        submit.validate_scope(plan)


def test_development_disjointness_hashes_and_task_binding_fail_closed(tmp_path):
    module = tooling()
    ledger = tmp_path / "consumed.json"
    module.write_once(ledger, dict(task="llava_description", samples=[dict(sample_id="old")]))
    protocol = dict(
        candidate_arm="mosaickv_output_impact_guarded_v3",
        development_exclusions={
            "llava_description": [dict(path=str(ledger), sha256=module.digest(ledger))]
        },
    )
    bound, check = module.development_exclusions(protocol, "llava_description", ["new"])
    assert bound == {str(ledger): module.digest(ledger)}
    assert check["excluded_count"] == 1
    assert check["overlap_count"] == 0
    for ids, message in [(["old"], "overlaps"), (["new", "new"], "duplicate")]:
        with pytest.raises(ValueError, match=message):
            module.development_exclusions(protocol, "llava_description", ids)
    with pytest.raises(ValueError, match="explicit exclusion"):
        module.development_exclusions(protocol, "mmvet_prefixkv", ["new"])
    protocol["development_exclusions"]["mmvet_prefixkv"] = protocol["development_exclusions"][
        "llava_description"
    ]
    with pytest.raises(ValueError, match="task mismatch"):
        module.development_exclusions(protocol, "mmvet_prefixkv", ["new"])
    ledger.write_text("changed synthetic fixture")
    with pytest.raises(ValueError, match="hash changed"):
        module.development_exclusions(protocol, "llava_description", ["new"])


@pytest.mark.parametrize("same_budget", [True, False])
def test_guard_trace_validation_requires_both_label_independent_phases(tmp_path, same_budget):
    import copy

    from mosaickv.selection.evidence_layer_allocation import reconstruction_guard

    module = tooling()
    guard = dict(
        reconstruction_guard(((1.0,),), ((0.5,),)),
        proposed_positions_sha256="a" * 64,
        anchor_positions_sha256="b" * 64,
        host_wall_seconds_including_device_transfers=0.01,
    )
    trace = dict(
        sample_id="synthetic-ci",
        teacher_forced_evaluation=dict(enabled=False),
        evidence_layer_allocation=dict(
            reconstruction_guard=guard,
            selected_positions_sha256="a" * 64,
            padding_bytes=0,
            benchmark_labels_accessed=False,
            prompt_bytes=80,
        ),
        bounded_decode=dict(budget_reservation=dict(reserved_generated_bytes=20)),
    )
    module.write_once(tmp_path / "generated.json", trace)
    with pytest.raises(RuntimeError, match="missing reconstruction"):
        module.reconstruction_trace_checks(tmp_path, ["synthetic-ci"], 100)
    forced = copy.deepcopy(trace)
    forced["teacher_forced_evaluation"]["enabled"] = True
    if not same_budget:
        # Existing common policy reserves fewer history slots for short PPL
        # targets, so it legitimately changes the available prompt capacity.
        forced["evidence_layer_allocation"]["prompt_bytes"] = 90
        forced["bounded_decode"]["budget_reservation"]["reserved_generated_bytes"] = 10
        forced["evidence_layer_allocation"]["selected_positions_sha256"] = "c" * 64
        forced["evidence_layer_allocation"]["reconstruction_guard"]["proposed_positions_sha256"] = (
            "c" * 64
        )
    module.write_once(tmp_path / "forced.json", forced)
    checked = module.reconstruction_trace_checks(tmp_path, ["synthetic-ci"], 100)
    assert checked["synthetic-ci"]["validated_trace_count"] == 2
    assert checked["synthetic-ci"]["teacher_forced_selection_identical"] == same_budget
    assert checked["synthetic-ci"]["teacher_forced_budget_identical"] == same_budget
    with pytest.raises(RuntimeError, match="accounting mismatch"):
        module.reconstruction_trace_checks(tmp_path, ["synthetic-ci"], 99)


def test_retry_reuses_original_execution_sha_and_rejects_changed_artifacts(tmp_path, monkeypatch):
    import json

    module = tooling()
    parent = tmp_path / "parent"
    parent.mkdir()
    original = dict(
        source_sha="a" * 40,
        jobs=[
            dict(index=i, method="vl_cache_reimpl" if i in [4, 9, 15, 20] else "full_kv")
            for i in range(22)
        ],
        bound_files={},
        protocol={},
    )
    module.write_once(parent / "plan.json", original)
    parent_sha = module.digest(parent / "plan.json")
    monkeypatch.setattr(module, "clean_source", lambda *_: "b" * 40)
    monkeypatch.setattr(module, "inputs", lambda *_a, **_k: original)
    artifact = parent / "result.json"
    module.write_once(artifact, dict(classification="synthetic_CI_fixture_not_measured"))
    for job in original["jobs"]:
        if job["method"] == "vl_cache_reimpl":
            continue
        module.write_once(
            parent / "workers" / f"{job['index']:02d}.json",
            dict(
                source_sha="a" * 40,
                plan_sha256=parent_sha,
                index=job["index"],
                artifacts={str(artifact): module.digest(artifact)},
                status="passed",
            ),
        )
    root = tmp_path / "retry"
    module.prepare_retry(root, parent, parent_sha)
    plan = json.loads((root / "plan.json").read_text())
    assert plan["source_sha"] == "b" * 40
    assert plan["retry_indices"] == [4, 9, 15, 20]
    assert len(list((root / "workers").glob("*.json"))) == 18
    reused = json.loads((root / "workers/00.json").read_text())
    assert reused["source_sha"] == reused["reused_from"]["source_sha"] == "a" * 40
    assert reused["plan_sha256"] == module.digest(root / "plan.json")
    artifact.write_text("changed synthetic fixture")
    with pytest.raises(ValueError, match="parent result changed"):
        module.prepare_retry(tmp_path / "bad-retry", parent, parent_sha)


@pytest.mark.parametrize("matched", [True, False])
def test_analysis_reduces_corpus_ppl_and_excludes_unequal_storage(tmp_path, monkeypatch, matched):
    """Synthetic CI records only: never experimental measurements."""
    import json
    import math

    import pandas as pd

    module = tooling()
    task = "llava_description"
    methods = ["full_kv", "prompt_attention_topk", "mosaickv_exact"]
    plan = dict(
        jobs=[
            dict(index=i, task=task, method=m, cap_mib=None if i == 0 else 40)
            for i, m in enumerate(methods)
        ],
        protocol=dict(
            samples_per_task=2,
            seed=7,
            tasks=[task],
            caps_mib=[40],
            methods=methods[1:],
            bootstrap_replicates=100,
        ),
    )
    monkeypatch.setattr(module, "inputs", lambda *_args: plan)
    for job in plan["jobs"]:
        observations, rows = [], []
        for sample_id, nll, tokens in [("ci-a", 2.0, 1), ("ci-b", 8.0, 7)]:
            for metric in ["teacher_forced_ppl", "rouge_l_f1"]:
                observations.append(
                    dict(
                        sample_id=sample_id,
                        metric_name=metric,
                        metric_scope="sample",
                        numerator=nll,
                        denominator=tokens,
                        metric_value=0.5,
                        method=job["method"],
                    )
                )
            rows.append(
                dict(
                    sample_id=sample_id,
                    end_to_end_time=1.0,
                    initial_active_kv_storage_bytes=40 * 2**20,
                    final_active_kv_storage_bytes=40 * 2**20,
                    peak_active_kv_storage_bytes=40 * 2**20
                    - (int(job["index"] == 2 and not matched)),
                )
            )
        path = tmp_path / f"observations-{job['index']}.parquet"
        pd.DataFrame(observations).to_parquet(path, index=False)
        module.write_once(
            tmp_path / "workers" / f"{job['index']:02d}.json",
            dict(
                plan_sha256="synthetic-ci-not-a-result",
                index=job["index"],
                artifacts={},
                record=dict(rows=rows, metric_observations_parquet_output=str(path)),
            ),
        )
    module.analyze(tmp_path, "synthetic-ci-not-a-result")
    report = json.loads((tmp_path / "analysis/report.json").read_text())
    summary = pd.read_parquet(tmp_path / "analysis/summary.parquet")
    assert summary.corpus_ppl.to_list() == pytest.approx([math.exp(10 / 8)] * 3)
    assert bool(report["comparisons"]) == matched
    assert bool(report["failures"]) != matched
