from pathlib import Path

import pytest
import torch


def test_precision_control_is_process_local_and_restored_even_on_failure(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from evicastkv_reduction_diagnostic import reduction_policy

    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    with pytest.raises(RuntimeError), reduction_policy(not previous):
        assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is not previous
        raise RuntimeError("controlled fixture")
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is previous
    with pytest.raises(ValueError, match="explicit boolean"), reduction_policy("false"):
        pass


def test_mismatched_or_failed_diagnostic_cannot_submit_gpu_successor(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_reduction_diagnostic as module

    from mosaickv.evaluation.fox_campaign import write_once

    monkeypatch.setattr(module, "clean_source", lambda *a: "fixture")
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **kw: pytest.fail("no submission")
    )
    write_once(tmp_path / "diagnostic_protocol.json", dict(source_sha="fixture"))
    module.analyze(tmp_path)
    gate = __import__("json").loads((tmp_path / "GATE_reduction_diagnostic.json").read_text())
    assert gate["status"] == "blocked"
    assert gate["thresholds_unchanged"]
