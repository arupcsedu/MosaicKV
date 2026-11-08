import json

import packed_history_study as study
import pytest


@pytest.mark.parametrize("feature_group", [32, 128])
def test_prepare_binds_both_inputs_and_only_submits_CPU_before_gates(
    tmp_path, monkeypatch, feature_group
):
    parent, archives = tmp_path / "parent", tmp_path / "archives"
    parent.mkdir()
    (parent / "plan.json").write_text(json.dumps(dict(bound_files={}, archives=str(archives))))
    for index in (0, 1):
        for root, names in (
            (parent, ("raw.jsonl", "manifest.json", "shared_source_prototypes-audit.logits.pt")),
            (archives, ("source_codes.pt", "capture.json", "manifest.json")),
        ):
            folder = root / f"workers/{index}"
            folder.mkdir(parents=True)
            for name in names:
                (folder / name).write_bytes(b"synthetic binding fixture, not measured data")
    calls = []
    monkeypatch.setattr(study, "PARENT", parent)
    monkeypatch.setattr(study, "clean_source", lambda *_: "a" * 40)
    monkeypatch.setattr(
        study, "submit", lambda root, action, **kwargs: calls.append((action, kwargs)) or "fixture"
    )
    output = tmp_path / "output"
    study.prepare(output, history_feature_group=feature_group)
    plan = json.loads((output / "plan.json").read_text())
    assert [name for name, _ in calls] == ["cpu", "after_cpu"]
    assert calls[-1][1] == dict(after="fixture")
    for index in (0, 1):
        assert (
            str(parent / f"workers/{index}/shared_source_prototypes-audit.logits.pt")
            in plan["bound_files"]
        )
        assert str(archives / f"workers/{index}/source_codes.pt") in plan["bound_files"]
    assert not plan["fresh_or_paper_campaign_authorized"]
    assert plan["history_feature_group"] == feature_group
    assert plan["guard_tokens"] == 32
    assert plan["maximum_KL"] == 0.02 and plan["maximum_logit_difference"] == 1.0


def test_missing_operator_evidence_blocks_model_submission(tmp_path):
    assert not study.operator_analysis(tmp_path)
    record = json.loads((tmp_path / "operator-analysis.json").read_text())
    assert not record["passed"] and record["timings"] == []
