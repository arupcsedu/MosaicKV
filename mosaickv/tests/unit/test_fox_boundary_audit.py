"""Saved boundary diagnostics cannot turn failed FullKV controls into passes."""

import json

import evicastkv_boundary_audit as module
import pytest

qualify = module.qualify


def pairs(passed=True):
    row = dict(workload="coco", sample_id="input", method="full_kv", budget_bytes=None)
    full = dict(status="completed", fullkv_chunked_logit_control_passed=passed)
    compressed = dict(status="completed", first_token_changed=False, A_B_answer_changed=False)
    return [(row, full), (row | dict(method="mosaickv_exact", budget_bytes=20), compressed)]


def test_failed_reference_invalidates_all_method_diagnostics_not_primary_A():
    rows, gate = qualify(pairs(False))
    assert gate["status"] == "BLOCKED" and gate["completed_pairs"] == 2
    assert gate["passing_fullkv_controls"] == 0 and gate["qualified_pairs"] == 0
    assert not any(row["numerically_qualified"] for row in rows)
    assert gate["A_primary_gate_not_rewritten"] and not gate["quality_inference_authorized"]


def test_passing_reference_qualifies_only_completed_diagnostics_not_task_quality():
    rows, gate = qualify(pairs())
    assert all(row["numerically_qualified"] for row in rows)
    assert gate["status"] == "controls_passed" and gate["qualified_pairs"] == 2
    assert not gate["quality_inference_authorized"]
    data = pairs()
    data[1][1]["status"] = "blocked"
    assert qualify(data)[1]["status"] == "BLOCKED"


def test_missing_duplicate_or_nonboolean_reference_is_not_a_pass():
    with pytest.raises(ValueError, match="missing independent"):
        qualify(pairs()[1:])
    with pytest.raises(ValueError, match="duplicate"):
        qualify(pairs() + pairs()[:1])
    assert qualify(pairs("true"))[1]["status"] == "BLOCKED"
    assert qualify([])[1]["status"] == "BLOCKED"


@pytest.fixture
def saved(tmp_path, monkeypatch):
    root = tmp_path / "source"
    raw = root / "worker/records/input.json"
    boundary = root / "worker/boundary/input.json"
    gate = root / "analysis/scoring/GATE_scoring.json"
    for path in (raw, boundary, gate):
        path.parent.mkdir(parents=True, exist_ok=True)
    a, b = pairs(False)[0]
    raw.write_text(json.dumps(a))
    boundary.write_text(json.dumps(b))
    gate.write_text(json.dumps(dict(status="passed")))
    (root / "protocol_lock.json").write_text(json.dumps(dict(source_sha="a" * 40)))
    exported = tmp_path / "boundary-export.json"
    exported.write_text(
        json.dumps(
            dict(
                records=[
                    dict(
                        path=str(boundary),
                        sha256=module.sha(boundary),
                        **b,
                    )
                ]
            )
        )
    )
    monkeypatch.setattr(module, "clean_source", lambda *_: "b" * 40)
    worker = dict(records=[dict(path=str(raw), sha256=module.sha(raw))])
    monkeypatch.setattr(module, "collect", lambda *_: (None, [worker], [a], []))
    return root, exported, tmp_path / "derived", boundary


def test_saved_audit_preserves_failed_controls_and_is_write_once(saved):
    root, exported, output, boundary = saved
    original = boundary.read_bytes()
    module.audit(root, exported, module.sha(exported), output)
    gate = json.loads((output / "GATE_boundary_B.json").read_text())
    assert gate["status"] == "BLOCKED" and gate["completed_pairs"] == 1
    assert not gate["quality_inference_authorized"]
    assert boundary.read_bytes() == original
    assert (output / "first_token_sensitivity.csv").is_file()
    with pytest.raises(FileExistsError):
        module.audit(root, exported, module.sha(exported), output)


def test_changed_export_or_boundary_is_rejected_before_output(saved):
    root, exported, output, boundary = saved
    with pytest.raises(ValueError, match="export hash changed"):
        module.audit(root, exported, "0" * 64, output)
    boundary.write_text(json.dumps(dict(status="completed")))
    with pytest.raises(ValueError, match="missing or changed B"):
        module.audit(root, exported, module.sha(exported), output)
    assert not output.exists()


def test_missing_exported_case_cannot_create_surviving_intersection(saved):
    root, exported, output, _ = saved
    exported.write_text(json.dumps(dict(records=[])))
    with pytest.raises(ValueError, match="missing or changed B"):
        module.audit(root, exported, module.sha(exported), output)
    assert not output.exists()
