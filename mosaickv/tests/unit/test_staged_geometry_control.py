from dataclasses import replace
from types import SimpleNamespace

import evicastkv_staged_geometry as workflow
import pytest
import torch
from test_source_row_geometry import packet

from mosaickv.evaluation.source_row_geometry import paired_resource_controls


def test_unsafe_upstream_blocks_without_gpu_submission(monkeypatch, tmp_path):
    writes = []
    monkeypatch.setattr(workflow.driver, "clean_source", lambda _: "revision")
    monkeypatch.setattr(
        workflow.driver.common.previous,
        "read",
        lambda _: dict(complete=False, failures=["missing"]),
    )
    monkeypatch.setattr(workflow.driver, "write_once", lambda p, d: writes.append((p, d)))
    monkeypatch.setattr(
        workflow, "schedule", lambda *args: pytest.fail("unsafe upstream must block")
    )
    workflow.defer(tmp_path / "run", "revision", "choose")
    assert writes[0][1]["GPU_submitted"] is False


def test_no_geometry_alternative_uses_original_decode_tiling_not_new_policy(monkeypatch, tmp_path):
    writes = []
    calls = []
    monkeypatch.setattr(workflow.driver, "clean_source", lambda _: "revision")
    monkeypatch.setattr(
        workflow.driver.common.previous,
        "read",
        lambda p: (
            dict(complete=True, failures=[], selected_geometry=None)
            if p.name == "decision.json"
            else dict(cpu_analysis="123")
        ),
    )
    monkeypatch.setattr(workflow.driver, "sha", lambda _: "hash")
    monkeypatch.setattr(workflow.driver, "write_once", lambda p, d: writes.append((p, d)))
    monkeypatch.setattr(workflow, "schedule", lambda *args: calls.append(args))
    workflow.defer(tmp_path / "run", "revision", "choose")
    assert writes[0][1]["geometry"] == [16, 4]
    assert writes[0][1]["parent"] == str(workflow.ASYNC)
    assert calls[0][2:] == ("construct", "123")


@pytest.mark.parametrize("bad_spills", [None, 1])
def test_paired_spill_control_preserves_original_one_head_constructor(monkeypatch, bad_spills):
    from mosaickv.evaluation import shared_source_kernel

    calls = []

    def inspect(template, *args, **kwargs):
        calls.append((template.source_row_tile, template.source_head_group))
        return dict(spills=0 if template.source_head_group == 1 else bad_spills)

    monkeypatch.setattr(shared_source_kernel, "project", inspect)
    attention = torch.nn.Linear(1, 1, bias=False)
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    p = replace(packet(), shape=(1, 32, 2, 128))
    with pytest.raises(RuntimeError, match="paired projection spills"):
        paired_resource_controls((p,), adapter, (32, 4))
    assert calls == [(16, 1), (32, 4)]
