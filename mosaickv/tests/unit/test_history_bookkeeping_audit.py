import pytest
import torch

from mosaickv.evaluation.history_bookkeeping_audit import profile


def test_metadata_profile_is_cpu_only_and_preserves_the_intended_transaction(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    report = profile(tmp_path / "new-profile", transactions=3, warmups=1)
    assert report["GPU_or_model_work_executed"] is False
    assert report["candidate_combined_or_promoted"] is False
    assert not torch.cuda.is_initialized()
    original, changed = report["results"]
    assert original["counters"]["mock_source_unpacks"] == 6
    assert changed["counters"]["mock_source_unpacks"] == 3
    assert original["counters"]["mock_projections"] == changed["counters"]["mock_projections"] == 6
    assert changed["measured_early_values"] == changed["measured_skips"] == 3
    assert any(f["function"] == "signature" for f in changed["functions"])
    with pytest.raises(FileExistsError):
        profile(tmp_path / "new-profile", transactions=3, warmups=1)


def test_cpu_profile_refuses_to_run_with_visible_cuda_scope(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="CPU-only"):
        profile(tmp_path / "bad")
