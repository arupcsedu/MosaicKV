from pathlib import Path

import pytest


def packet_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_packet

    return evicastkv_packet


def test_packet_text_is_write_once(tmp_path, monkeypatch):
    module = packet_module(monkeypatch)
    path = tmp_path / "nested/report.md"
    module.text_once(path, "negative\n")
    with pytest.raises(FileExistsError):
        module.text_once(path, "positive\n")
    assert path.read_text() == "negative\n"


def test_no_stage_is_inferred_from_missing_artifacts(tmp_path, monkeypatch):
    module = packet_module(monkeypatch)
    assert module.stages(tmp_path) == {}
    assert module.all_records(tmp_path) == ([], [], {})


def test_secondary_timing_family_is_declared_separately():
    from mosaickv.evaluation.fox_statistics import paired_scalar_statistics

    result = paired_scalar_statistics(
        dict(a=2.0, b=4.0),
        dict(a=1.0, b=3.0),
        dict(a=("a",), b=("b",)),
        dict(a="all", b="all"),
        resamples=7,
        family_size=8,
    )
    assert result["family_size"] == 8
    assert result["simultaneous_coverage"] == 1 - 0.05 / 8
    assert result["speedup_ratio_of_means"] == 1.5


def test_no_plot_dependency_is_loaded_without_metric_rows(tmp_path, monkeypatch):
    module = packet_module(monkeypatch)
    assert module.render_plots(tmp_path, tmp_path, []) == []


def test_vector_plots_preserve_measured_values_without_installs(tmp_path, monkeypatch):
    module = packet_module(monkeypatch)
    from mosaickv.evaluation.fox_campaign import write_once

    write_once(
        tmp_path / "analysis/scoring/native/native_metrics.json",
        [dict(workload="coco", scores={"mosaickv_exact:1048576": 2.5})],
    )
    rows = [
        dict(
            stage="scoring",
            workload="coco",
            method="mosaickv_exact",
            budget_bytes=1048576,
            initial_kv_bytes=1048576,
            status="completed",
        )
    ]
    names = module.render_plots(tmp_path, tmp_path, rows)
    assert names == ["coco-cider.svg"]
    assert (tmp_path / "coco-cider.tex").exists()
    assert module.read(tmp_path / "coco-cider.json")[0]["score"] == 2.5


def test_incomplete_packet_keeps_missing_stages_missing(tmp_path, monkeypatch):
    module = packet_module(monkeypatch)
    from mosaickv.evaluation.fox_campaign import write_once

    monkeypatch.setattr(module, "clean_source", lambda *a: "test-fixture-not-measured")
    root = tmp_path / "run"
    write_once(
        root / "protocol_lock.json", dict(source_root=str(root), source_sha="fixture", repeats=3)
    )
    output = tmp_path / "derived"
    module.packet(root, output)
    text = (output / "final_writing_packet.md").read_text()
    assert "NOT_RUN" in text
    assert "confirmation" in text
    assert module.read(output / "analysis_manifest.json")["raw_modified"] is False
    assert module.read(output / "GATE_analysis.json")["status"] == "partial"
    assert (output / "claim_evidence.md").exists()
    assert module.read(output / "quality_claims.json") == []


def test_native_tables_do_not_call_zero_crossing_equivalence(tmp_path, monkeypatch):
    module = packet_module(monkeypatch)
    from mosaickv.evaluation.fox_campaign import write_once

    write_once(
        tmp_path / "analysis/scoring/native/native_metrics.json",
        [
            dict(
                workload="coco",
                metric="cider",
                unique_inputs=8,
                scores={"mosaickv_exact:1048576": 2.5},
                contrasts=[
                    dict(
                        reference="prompt_attention_topk:1048576",
                        candidate="mosaickv_exact:1048576",
                        difference=0.1,
                        ordinary_95_ci=[-0.2, 0.3],
                        simultaneous_ci=[-0.4, 0.5],
                    )
                ],
            )
        ],
    )
    output = tmp_path / "derived"
    output.mkdir()
    module.native_tables(tmp_path, output)
    result = module.read(output / "coco-cider-quality-contrasts.json")
    assert result[0]["claim"] == "inconclusive_not_equivalent"
    assert "reimpl" not in (output / "coco-cider-quality.tex").read_text()


def test_interval_plot_uses_actual_adjusted_endpoints(tmp_path, monkeypatch):
    m = packet_module(monkeypatch)
    report = dict(
        workload="coco",
        metric="cider",
        contrasts=[
            dict(
                candidate="mosaickv_exact:20971520",
                difference=0.25,
                ordinary_95_ci=[-0.2, 0.4],
                simultaneous_ci=[-0.3, 0.6],
            )
        ],
    )
    name = m.contrast_plot(report, tmp_path)
    assert name.endswith("paired-difference.svg")
    assert m.read(tmp_path / name.replace(".svg", ".json"))[0]["adjusted"] == [-0.3, 0.6]


def test_legacy_numerical_record_without_status_is_not_certified(tmp_path, monkeypatch):
    m = packet_module(monkeypatch)
    from mosaickv.evaluation.fox_campaign import write_once

    root = tmp_path / "run"
    write_once(root / "outputs/correctness/000/old/correctness/x.json", dict(max_logit=0.01))
    output = tmp_path / "derived"
    output.mkdir()
    m.numerical_summary(root, output)
    assert m.read(output / "numerical_error_summary.json")["legacy_without_explicit_status"] == 1


def test_mechanism_export_keeps_corpus_cider_and_all_failed_denominators(tmp_path, monkeypatch):
    m = packet_module(monkeypatch)
    from mosaickv.evaluation.fox_campaign import write_once

    write_once(
        tmp_path / "analysis/mechanisms/native/mechanism_metrics.json",
        [dict(workload="coco", metric="cider", scores={"route-0:20971520": 41.25})],
    )
    rows = [
        dict(
            stage="mechanisms",
            workload="coco",
            variant="route-0",
            budget_bytes=20971520,
            sample_id="a",
            status="completed",
            initial_kv_bytes=20971520,
            e2e_wall_seconds=2,
        ),
        dict(
            stage="mechanisms",
            workload="coco",
            variant="route-0",
            budget_bytes=20971520,
            sample_id="b",
            status="failed",
        ),
    ]
    output = tmp_path / "exports"
    output.mkdir()
    m.mechanism_exports(tmp_path, output, rows)
    result = m.read(output / "mechanism_summaries.json")[0]
    assert result["native_quality"] == 41.25
    assert (result["attempted"], result["completed"], result["failed"]) == (2, 1, 1)
    assert "NO_CONFIG_FROZEN" in (output / "development_decision.md").read_text()
    assert len((output / "development_predictions.jsonl").read_text().splitlines()) == 2
