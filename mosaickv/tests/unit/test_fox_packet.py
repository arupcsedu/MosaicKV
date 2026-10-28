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
