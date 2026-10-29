"""Synthetic numerical-equivalence gate tests, not experiment measurements."""

import json
from copy import deepcopy

import evicastkv_zero_draft_clone as module
import pytest

from mosaickv.evaluation.fox_campaign import sha


def fixture():
    return dict(
        answer="fixture",
        nll_sum=2.0,
        evaluated_tokens=2,
        teacher_forced_ppl=2.0,
        rouge_l_f1=0.5,
        config_sha="fixture",
        audited_request_owned_peak_bytes=600,
        audited_decode_transaction_peak_bytes=200,
        metrics={
            **dict(zip(module.common.STORAGE, (100, 150, 200), strict=True)),
            "end_to_end_time": 1.0,
        },
    )


def test_equivalence_gate_separates_quality_storage_ownership_and_latency():
    old = fixture()
    new = deepcopy(old)
    new["audited_request_owned_peak_bytes"] = 450
    new["metrics"]["end_to_end_time"] = 2.0
    result = module.compare_rows(new, old)
    assert result["exact_quality_and_config_equivalence"]
    assert result["exact_live_storage_equivalence"]
    assert result["registered_owner_peak_reduced"]
    assert result["new_end_to_end_seconds"] > result["old_end_to_end_seconds"]


@pytest.mark.parametrize(
    "field", ["answer", "nll_sum", "evaluated_tokens", "rouge_l_f1", "config_sha"]
)
def test_quality_change_cannot_be_hidden_by_memory_reduction(field):
    old = fixture()
    new = deepcopy(old)
    new[field] = "different"
    new["audited_request_owned_peak_bytes"] = 450
    assert not module.compare_rows(new, old)["exact_quality_and_config_equivalence"]


def test_allocator_optimization_is_not_called_a_kv_capacity_reduction():
    old = fixture()
    new = deepcopy(old)
    old["pytorch_request_peak_allocated_bytes"] = 1000
    new["pytorch_request_peak_allocated_bytes"] = 900
    result = module.compare_rows(new, old, "cuda_allocator")
    assert result["cuda_allocator_peak_reduced"]
    assert result["registered_owner_peak_unchanged"]
    assert not result["registered_owner_peak_reduced"]


@pytest.mark.parametrize("missing_ppl", [False, True])
def test_allocator_analyzer_requires_all_phase_tokens_and_does_not_claim_kv_gain(
    tmp_path, monkeypatch, missing_ppl
):
    jobs = []
    for index in range(4):
        worker = tmp_path / f"workers/{index}"
        (worker / "rows").mkdir(parents=True)
        traces = {}
        for phase in ("generation",) if missing_ppl else ("generation", "teacher_forced"):
            p = worker / f"{phase}.json"
            p.write_text(
                json.dumps(
                    dict(execution_phase=phase, generated_token_ids=[1, 2], selected_blocks=[0])
                )
            )
            traces[str(p)] = sha(p)
        old = fixture() | dict(
            status="completed",
            traces=traces,
            memory_audit_traces={},
            pytorch_request_peak_allocated_bytes=1000,
        )
        prior = worker / "prior.json"
        prior.write_text(json.dumps(old))
        new = deepcopy(old)
        new["pytorch_request_peak_allocated_bytes"] = 900
        (worker / "rows/new.json").write_text(json.dumps(new))
        (worker / "result.json").write_text(json.dumps(dict(failures=[])))
        jobs.append(
            dict(
                index=index,
                arms=[dict(label="new")],
                prior_equivalence_row=str(prior),
                task="fixture",
                diagnostic_history_capacity=128,
            )
        )
    monkeypatch.setattr(
        module.common,
        "inputs",
        lambda *_: dict(
            jobs=jobs, source_sha="fixture", equivalence_target_metric="cuda_allocator"
        ),
    )
    module.analyze(tmp_path, "fixture")
    gate = module.common.read(tmp_path / "analysis/GATE.json")
    assert gate["status"] == (
        "failed" if missing_ppl else "passed_equivalence_and_allocator_reduction"
    )
    assert all(not c["registered_owner_peak_reduced"] for c in gate["checks"])
    assert not gate["superiority_established"] and not gate["whole_request_KV_peak_certified"]
