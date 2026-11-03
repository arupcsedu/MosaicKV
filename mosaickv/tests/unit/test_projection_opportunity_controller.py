import json

import evicastkv_projection_opportunity as driver


def test_immutable_ledger_string_paths_and_path_objects_share_the_same_reader(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(dict(task="consumed", sample_id="not_fresh")))
    assert driver.read(str(path)) == driver.read(path) == dict(
        task="consumed", sample_id="not_fresh"
    )


def test_analysis_requires_both_inputs_not_only_mean_or_text_union(monkeypatch, tmp_path):
    plan = dict(source_sha="s", minimum_skip_fraction_on_each_input=0.10)
    monkeypatch.setattr(driver.pair.common.previous.common, "inputs", lambda *a: plan)
    for i in (0, 1):
        out = tmp_path / f"workers/{i}"
        out.mkdir(parents=True)
        evidence = dict(
            packet_bits_identical=True,
            no_decode_executed=True,
            layers=[
                dict(
                    protected_only=dict(
                        total_programs=100, wholly_overwritten_programs=30 if i == 0 else 9
                    ),
                    protected_plus_text=dict(total_programs=100, wholly_overwritten_programs=40),
                    correction_scalars_per_component=[0, 0],
                )
            ],
        )
        (out / "opportunity.json").write_text(json.dumps(evidence))
        result = dict(
            source_sha="s",
            plan_sha256="p",
            status="completed",
            outputs={"opportunity.json": driver.sha(out / "opportunity.json")},
        )
        (out / "result.json").write_text(json.dumps(result))
    driver.analyze(tmp_path, "p")
    result = driver.read(tmp_path / "analysis/decision.json")
    assert result["complete"]
    assert not result["protected_only_kernel_feasibility_gate"]
    assert result["summaries"][1]["nonempty_correction_components"] == 0
    assert not result["measured_latency_gain"]
    assert not result["further_jobs_submitted"]


def test_missing_producer_is_preserved_as_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        driver.pair.common.previous.common,
        "inputs",
        lambda *a: dict(source_sha="s", minimum_skip_fraction_on_each_input=0.10),
    )
    driver.analyze(tmp_path, "p")
    result = driver.read(tmp_path / "analysis/decision.json")
    assert not result["complete"] and len(result["failures"]) == 2
    assert not result["protected_only_kernel_feasibility_gate"]
