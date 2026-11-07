import json
from types import SimpleNamespace

import pytest
import torch


def test_write_once_cpu_analysis_uses_real_eos_boundary_and_measured_storage(tmp_path, monkeypatch):
    import analyze_readout_horizon_eos as script

    root, output = tmp_path / "producer", tmp_path / "derived"
    root.mkdir()
    arms = ["full_kv", *(f"controlled_plan_{i}" for i in range(10))]
    plan = dict(
        source_sha="producer",
        evaluation_only_fixed_plan_horizon_audit=True,
        arms=arms,
        bound_files={},
        parent=dict(config=dict(model=dict(id="fixture", revision="pinned"))),
    )
    (root / "plan.json").write_text(json.dumps(plan))
    logits = torch.zeros(128, 4)
    logits[:5, 1] = 3
    logits[5:, 2] = 3
    for index in (0, 1):
        folder = root / f"workers/{index}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(
            json.dumps(dict(status="completed", errors=[], source_sha="producer"))
        )
        for i, arm in enumerate(arms):
            path = folder / f"{arm}.pt"
            torch.save(logits, path)
            row = dict(
                arm=arm,
                status="completed",
                logits_path=str(path),
                logits_sha256=script.sha(path),
                detail=dict(
                    answer=dict(answer_tokens=5, token_ids=[1] * 5, stopped_by_eos=True),
                    initial_encoded_bytes=1024,
                    final_encoded_bytes=2048,
                    reported_counter_peak_bytes=3072,
                ),
            )
            (folder / f"{i:02}-{arm}.json").write_text(json.dumps(row))
    monkeypatch.setattr(script, "clean_source", lambda path: "analysis")
    monkeypatch.setattr(
        script.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(eos_token_id=2)
    )
    monkeypatch.setattr("sys.argv", ["audit", "--root", str(root), "--output", str(output)])
    script.main()
    decision = json.loads((output / "decision.json").read_text())
    assert decision["results"]["0"]["semantic_prediction_limit"] == 6
    assert decision["results"]["0"]["storage_matching"]["all_boundaries_matched"]
    assert not decision["superiority_established"]
    assert not decision["useful_future_draft_horizons_on_both_consumed_inputs"]
    with pytest.raises(FileExistsError):
        script.main()
