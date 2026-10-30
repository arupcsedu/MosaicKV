"""Synthetic-only analyzer fixtures; never paper measurement artifacts."""

import json

import evicastkv_donor_isolation as runner
import pytest
import torch

from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.evaluation.fox_campaign import sha


@pytest.mark.parametrize("corrupt_attention", [False, True])
def test_complete_analysis_serializes_booleans_and_rejects_nonzero_donor_mass(
    tmp_path, monkeypatch, corrupt_attention
):
    from dataclasses import asdict

    source, plan_sha = "b" * 40, "a" * 64
    jobs = []
    for index in range(2):
        folder = tmp_path / f"workers/{index}"
        old, joint = tmp_path / f"old{index}", tmp_path / f"joint{index}"
        for path in (folder, old, joint):
            path.mkdir(parents=True)
        full = torch.arange(5).float().repeat(32, 1)
        torch.save({"logits": {"full_kv_reference": full}}, old / "captures.pt")
        saved, parent, controls = {}, {}, {}
        for role in ("topk_fixture", "evicast_fixture"):
            positions = [[[0, 3]]]
            parent[role] = {
                "anchor": dict(logits=full, positions=positions),
                "restore_delimiter": dict(logits=full),
            }
            saved[role], controls[role] = {}, {}
            for mode in runner.MODES:
                saved[role][mode] = dict(logits=full, positions=positions)
                storage = {k: 100 for k in runner.common.STORAGE}
                trace = folder / f"{role}-{mode}-trace.json"
                trace.write_text(json.dumps(dict(cache_lifecycle=storage)))
                controls[role][mode] = dict(
                    storage=storage,
                    trace_path=str(trace),
                    trace_sha256=sha(trace),
                    intervention=dict(
                        frozen_masked_donor_payload_bytes=8,
                        physical_KV_bytes_saved=0,
                        availability_only_not_physical_eviction=True,
                        additional_mask_backing_bytes=8,
                        attention=[
                            dict(
                                prediction=p,
                                maximum_donor_attention_probability=0.5
                                if corrupt_attention and mode == "donor_only"
                                else 0.0,
                            )
                            for p in range(2, 33)
                        ],
                    ),
                )
        torch.save(parent, joint / "captures.pt")
        torch.save(saved, folder / "captures.pt")
        (folder / "controls.json").write_text(json.dumps(controls))
        (folder / "result.json").write_text(
            json.dumps(
                dict(
                    status="completed",
                    failures=[],
                    source_sha=source,
                    plan_sha256=plan_sha,
                    outputs={name: sha(folder / name) for name in ("captures.pt", "controls.json")},
                )
            )
        )
        jobs.append(
            dict(
                index=index,
                task=f"synthetic_fixture{index}",
                parent_worker=str(old),
                delimiter_worker=str(joint),
            )
        )
    plan = dict(
        jobs=jobs,
        source_sha=source,
        guard=asdict(DraftAllocationGuardConfig()),
        maximum_reference_kl=0.02,
        maximum_logit_difference=1.0,
    )
    monkeypatch.setattr(runner.common, "inputs", lambda _root, _sha: plan)
    if corrupt_attention:
        with pytest.raises(ValueError, match="nonzero attention"):
            runner.analyze(tmp_path, plan_sha)
    else:
        runner.analyze(tmp_path, plan_sha)
        gate = json.loads((tmp_path / "analysis/GATE.json").read_text())
        assert gate["completed_comparisons"] == 4
        assert gate["status"] == "complete_consumed_diagnostic"
        assert not gate["guard_weakened"] and not gate["calibration_authorized"]
        assert all(r["donor_only_guard_passed"] is True for r in gate["checks"])
