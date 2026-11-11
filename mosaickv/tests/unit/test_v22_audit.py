"""CPU-only census equivalence/provenance fixtures; never model evidence."""

import copy
import json

import evicastkv_v22_audit as study
import pytest
import torch


def fixture(layers=2, length=5):
    scores = {i: torch.linspace(-2, 2, length) for i in range(layers)}
    reference = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    parent = reference + 0.125
    report = dict(
        kind="synthetic_four_prefix_census_fixture",
        horizon=4,
        KL_normalization=0.02,
        decision_margin_normalization=1.0,
        source_shape=[1, 32, length, 128],
        reference_logits_sha256=study.tensor_sha(reference),
        parent_logits_sha256=study.tensor_sha(parent),
        objective_value=0.5,
        draft_input_token_ids=[1, 2, 3, 4],
        draft_predicted_token_ids=[2, 3, 4, 5],
        reference_logits_shape=[4, 6],
        parent_logits_shape=[4, 6],
        reference_KL_per_prefix=[0.1] * 4,
        reference_greedy_margins=[0.2] * 4,
        parent_reference_token_margins=[0.1] * 4,
        margin_shortfalls=[0.1] * 4,
        positive_row_gains=4,
        negative_row_gains=4,
        zero_row_gains=2,
        nonzero_gradient_scalars=100,
        prompt_leaf_KV_backing_bytes=1000,
        autograd_saved_nonweight_peak_backing_bytes=5000,
        autograd_saved_tensor_observations=49,
        no_second_prefill=True,
        original_cache_unchanged=True,
        model_weights_unchanged=True,
        model_parameter_gradients_unchanged=True,
        private_cache_only=True,
        requires_grad_flags_unchanged=True,
        returns_cpu_scores_only=True,
        total_wall_seconds=2.0,
    )
    saved = dict(
        old_scores=scores,
        new_scores={i: t.clone() for i, t in scores.items()},
        old_FullKV4=reference,
        new_FullKV4=reference.clone(),
        old_parent4=parent,
        new_parent4=parent.clone(),
    )
    new = copy.deepcopy(report)
    new.update(total_wall_seconds=1.0, census_implementation="incremental_weak_owner_counter")
    return saved, report, new


def compare(saved, old, new):
    return study.compare_forecasts(
        saved["old_scores"],
        saved["new_scores"],
        old,
        new,
        layers=len(saved["old_scores"]),
        length=old["source_shape"][2],
    )


def test_counter_metadata_can_change_but_all_signed_scores_and_semantics_must_match():
    saved, old, new = fixture()
    assert study.verify_four_logits(saved, {"old": old, "new": new})
    result = compare(saved, old, new)
    assert result["pass_exact_counter_and_forecast_control"]
    assert result["all_signed_score_vectors_bitwise_equal"]
    assert result["old_score_sha256"] == result["new_score_sha256"]
    assert result["diagnostic_only_not_full_request_latency"]
    assert result["old_total_forecast_seconds"] == 2
    assert result["new_total_forecast_seconds"] == 1


@pytest.mark.parametrize(
    "field",
    [
        "autograd_saved_nonweight_peak_backing_bytes",
        "autograd_saved_tensor_observations",
        "objective_value",
        "horizon",
    ],
)
def test_saved_peak_pack_count_and_objective_changes_cannot_pass(field):
    saved, old, new = fixture()
    new[field] += 1
    result = compare(saved, old, new)
    assert not result["pass_exact_counter_and_forecast_control"]
    assert field in result["semantic_report_mismatches"]


def test_signed_zero_difference_is_not_bitwise_score_equivalence():
    saved, old, new = fixture()
    saved["new_scores"][0][2] = -0.0
    assert torch.equal(saved["old_scores"][0], saved["new_scores"][0])
    assert not compare(saved, old, new)["pass_exact_counter_and_forecast_control"]


@pytest.mark.parametrize("invalid", ["missing", "dtype", "nan", "shape", "bool_key"])
def test_partial_nonfinite_or_nonfp32_scores_fail_closed(invalid):
    saved, old, new = fixture()
    if invalid == "missing":
        saved["new_scores"].pop(0)
    elif invalid == "dtype":
        saved["new_scores"][0] = saved["new_scores"][0].double()
    elif invalid == "nan":
        saved["new_scores"][0][0] = float("nan")
    elif invalid == "shape":
        saved["new_scores"][0] = saved["new_scores"][0][None]
    else:
        saved["new_scores"][False] = saved["new_scores"].pop(0)
    with pytest.raises(ValueError, match="complete finite"):
        compare(saved, old, new)


def test_agreeing_but_failed_isolation_controls_cannot_pass():
    saved, old, new = fixture()
    old["original_cache_unchanged"] = new["original_cache_unchanged"] = False
    assert not compare(saved, old, new)["pass_exact_counter_and_forecast_control"]


def test_saved_four_prefix_tensors_are_independently_digest_checked():
    saved, old, new = fixture()
    saved["new_FullKV4"][0, 0] = -0.0
    with pytest.raises(ValueError, match="digest mismatch"):
        study.verify_four_logits(saved, {"old": old, "new": new})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def historical_fixture(root):
    write(
        root / "plan.json",
        dict(
            source_sha=study.PRODUCER_SHA,
            policy_sha256=study.digest(study.previous.POLICY),
            bindings={},
            jobs=[],
        ),
    )
    write(
        root / "analysis/GATE_exact_input_v21.json",
        dict(
            complete=True,
            failures=[],
            bounded_numerical_repair_all_inputs_pass=False,
        ),
    )
    for index in range(2):
        out = root / f"workers/{index}"
        write(out / "raw.jsonl", {"synthetic": True})
        write(
            out / "result.json",
            dict(
                status="completed",
                failures=[],
                actual_rows=3,
                source_sha=study.PRODUCER_SHA,
                plan_sha256=study.sha(root / "plan.json"),
                outputs={"raw.jsonl": study.sha(out / "raw.jsonl")},
            ),
        )


def test_historical_inventory_preserves_negative_filename_and_original_producer(tmp_path):
    historical_fixture(tmp_path)
    old, bindings = study._historical(tmp_path)
    assert old["source_sha"] == study.PRODUCER_SHA
    assert str(tmp_path / "analysis/GATE_exact_input_v21.json") in bindings
    assert len(bindings) == 6
    write(tmp_path / "workers/0/raw.jsonl", {"synthetic": "changed"})
    with pytest.raises(ValueError, match="artifact changed"):
        study._historical(tmp_path)


def test_historical_positive_or_misattributed_results_are_not_reused(tmp_path):
    historical_fixture(tmp_path)
    path = tmp_path / "analysis/GATE_exact_input_v21.json"
    gate = json.loads(path.read_text())
    gate["bounded_numerical_repair_all_inputs_pass"] = True
    write(path, gate)
    with pytest.raises(ValueError, match="negative V22"):
        study._historical(tmp_path)


def test_corrected_analysis_uses_old_source_and_new_derived_directory(monkeypatch):
    calls = []
    monkeypatch.setattr(
        study.previous, "analyze", lambda *a, **k: calls.append((a, k)) or {"complete": True}
    )
    assert study.regenerate_historical_analysis()["complete"]
    assert calls == [
        (
            (study.OLDROOT,),
            dict(
                producer_source=study.PRODUCER_SOURCE,
                output_dir=study.OLDROOT / "analysis-v22-corrected",
            ),
        )
    ]


def test_missing_prepare_gets_failure_receipt_without_touching_historical_analysis(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(study.previous.parent.fixed, "analysis_cpu_controls", lambda: None)
    monkeypatch.setattr(
        study, "regenerate_historical_analysis", lambda: pytest.fail("not after both workers")
    )
    result = study.analyze(tmp_path)
    assert not result["complete"]
    assert result["failures"]
    assert result["historical_quality_producer_sha"] == study.PRODUCER_SHA
    assert result["source_family_peak_matching_unsupported"]
    assert not result["paper_or_fresh_campaign_authorized"]
    assert not result["new_128_decode_performed"]


def audit_fixture(root, oldroot, monkeypatch, *, wrong_representation=False):
    plan = dict(source_sha="a" * 40)
    write(root / "plan.json", plan)
    monkeypatch.setattr(study, "load", lambda _: plan)
    monkeypatch.setattr(study, "OLDROOT", oldroot)
    monkeypatch.setattr(study.previous.parent.fixed, "analysis_cpu_controls", lambda: None)
    for index in range(2):
        out = root / f"workers/{index}"
        saved, old, new = fixture(layers=32)
        write(out / "old_draft.json", old)
        write(out / "new_draft.json", new)
        torch.save(saved, out / "forecast_tensors.pt")
        packet_sha = "c" * 64
        write(
            oldroot / f"workers/{index}/raw.jsonl",
            dict(
                arm=study.previous.ARMS[2],
                representation_sha256=packet_sha,
            ),
        )
        summary = dict(
            comparison=compare(saved, old, new),
            representation_sha256=packet_sha,
            quality_producer_sha=study.PRODUCER_SHA,
            old_quality_guard128=False,
        )
        if wrong_representation and index == 0:
            summary["representation_sha256"] = "d" * 64
        write(out / "raw.jsonl", summary)
        outputs = {p.name: study.sha(p) for p in out.iterdir()}
        write(
            out / "result.json",
            dict(
                status="completed",
                failures=[],
                worker=index,
                source_sha=plan["source_sha"],
                plan_sha256=study.sha(root / "plan.json"),
                outputs=outputs,
            ),
        )


def test_completed_paired_audit_reuses_but_never_relabels_negative_128_predictions(
    tmp_path, monkeypatch
):
    audit_fixture(tmp_path / "audit", tmp_path / "old", monkeypatch)
    calls = []
    monkeypatch.setattr(
        study,
        "regenerate_historical_analysis",
        lambda: (
            calls.append(True)
            or dict(
                complete=True,
                failures=[],
                producer_source_sha=study.PRODUCER_SHA,
                analysis_source_sha="a" * 40,
                bounded_numerical_repair_all_inputs_pass=False,
            )
        ),
    )
    gate = study.analyze(tmp_path / "audit")
    assert gate["complete"] and gate["paired_counter_optimization_verified"]
    assert gate["historical_negative_128_result_preserved"]
    assert not gate["historical_quality_analysis"]["bounded_numerical_repair_all_inputs_pass"]
    assert not gate["whole_request_KV_peak_certified"]
    assert calls == [True]


def test_packet_hash_mismatch_blocks_historical_quality_reuse(tmp_path, monkeypatch):
    audit_fixture(tmp_path / "audit", tmp_path / "old", monkeypatch, wrong_representation=True)
    monkeypatch.setattr(
        study,
        "regenerate_historical_analysis",
        lambda: dict(
            complete=True,
            failures=[],
            producer_source_sha=study.PRODUCER_SHA,
            analysis_source_sha="a" * 40,
            bounded_numerical_repair_all_inputs_pass=False,
        ),
    )
    gate = study.analyze(tmp_path / "audit")
    assert not gate["complete"]
    assert not gate["paired_counter_optimization_verified"]
    assert any("representation/producer-bound" in f["error"] for f in gate["failures"])
