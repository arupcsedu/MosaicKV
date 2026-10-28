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
