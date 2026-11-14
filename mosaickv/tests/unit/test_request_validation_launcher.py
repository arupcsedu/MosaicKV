import json
from pathlib import Path

import evicastkv_request_validation as module
import pytest


def test_canary_keeps_dataset_configs_and_strips_all_saved_prefixes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    protocol = source / "mosaickv/docs/request_precision_validation_20261030.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("frozen protocol")
    parent = tmp_path / "parent"
    parent.mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text("[0.1,0.2]")
    media = tmp_path / "image.png"
    media.write_bytes(b"fixture image not decoded")
    jobs = []
    for index, task in enumerate(("description", "mmvet")):
        config = dict(
            dataset=dict(id=task, revision="pinned", split="validation"),
            bounded_decode=dict(incremental_history=dict(layer_profile_path=str(profile))),
        )
        payload = dict(
            task=task,
            sample_id=str(index),
            prompt="question",
            fullkv_prefix_tokens=[1, 2],
            media=[dict(kind="image", path=str(media), sha256=module.sha(media))],
        )
        target = parent / f"{index}.json"
        target.write_text(json.dumps(payload))
        jobs.append(dict(input=str(target), sample_id=str(index), arms=[dict(config=config)]))
    (parent / "plan.json").write_text(json.dumps(dict(jobs=jobs)))
    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "PARENT", parent)
    monkeypatch.setattr(module, "clean_source", lambda path: "a" * 40)
    root = tmp_path / "canary"
    digest = module.prepare_canary(root)
    assert digest == module.sha(root / "plan.json")
    plan = module.read(root / "plan.json")
    assert [j["config"]["dataset"]["id"] for j in plan["jobs"]] == ["description", "mmvet"]
    assert all("fullkv_prefix_tokens" not in module.read(j["inputs"][0]) for j in plan["jobs"])
    assert len(plan["arms"]) == 11 and plan["cohort_size"] == 128
    assert plan["request_policy"] == module.REQUEST_POLICY
    assert plan["request_tensor_auditing_enabled"] is False
    assert plan["paper_campaign_authorized"] is False
    with pytest.raises(ValueError, match="new bounded"):
        module.prepare_canary(root)


def test_messages_fail_closed_on_answer_or_saved_token_fields():
    base = dict(task="coco_caption", sample_id="ID", prompt="question", media=[])
    for forbidden in ("references", "answer", "fullkv_prefix_tokens"):
        with pytest.raises(ValueError, match="neither references nor draft tokens"):
            module.load_messages(dict(base, **{forbidden: ["secret"]}))


def test_media_identity_ignores_png_container_but_not_pixels(tmp_path):
    from PIL import Image

    image = Image.new("RGB", (5, 3), (10, 20, 30))
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    image.save(a, compress_level=0)
    image.save(b, compress_level=9)
    assert module.sha(a) != module.sha(b)
    assert module.rgb_identity(a) == module.rgb_identity(b)
    image.putpixel((0, 0), (11, 20, 30))
    image.save(b)
    assert module.rgb_identity(a) != module.rgb_identity(b)


def test_runtime_seed_and_manifest_execution_seed_match_without_mutating_template():
    base = dict(execution=dict(seed=7, attention_implementation="eager"), model=dict(id="pinned"))
    plan = dict(config=base, seed=31)
    resolved = module.resolved_mapping(plan, {})
    assert resolved["execution"]["seed"] == 31
    assert base["execution"]["seed"] == 7
    assert module.resolved_mapping(plan, dict(config=base)) == resolved


def test_paired_timing_orders_are_balanced_randomized_and_deterministic():
    arms = [("candidate", "query3"), ("baseline", "lossless")]
    a = module.balanced_pair_orders(arms, 20, 7)
    assert a == module.balanced_pair_orders(arms, 20, 7)
    assert sum(order[0] == arms[0] for order in a) == 10
    assert a != [arms] * 20
    with pytest.raises(ValueError, match="even"):
        module.balanced_pair_orders(arms, 19, 7)


def test_analyzer_rejects_profiled_rows_before_any_automatic_gpu_followup(tmp_path, monkeypatch):
    root = tmp_path / "canary"
    folder = root / "workers/0"
    folder.mkdir(parents=True)
    plan = dict(
        stage="canary",
        source_sha="a" * 40,
        jobs=[dict(index=0, inputs=["fixture"])],
        arms=[["prefixkv_reimpl", "query3"]],
    )
    row = dict(
        sample_id="fixture",
        method="prefixkv_reimpl",
        precision="query3",
        trial=0,
        warmup=False,
        status="completed",
        greedy_ids=[0] * 128,
        end_to_end_seconds=1.0,
        whole_request_allocator_peak_certified=True,
        source_KV_owner_retired_before_decode=True,
        audit_enabled=True,
    )
    raw = folder / "raw.jsonl"
    raw.write_text(json.dumps(row) + "\n")
    (folder / "result.json").write_text(
        json.dumps(
            dict(
                source_sha=plan["source_sha"],
                plan_sha256="bound",
                failures=[],
                outputs={"raw.jsonl": module.sha(raw)},
            )
        )
    )
    monkeypatch.setattr(module.common, "inputs", lambda root, expected: plan)
    monkeypatch.setattr(module, "materialize_cohort", lambda *args: pytest.fail("must not expand"))
    module.analyze(root, "bound")
    gate = module.read(root / "analysis/GATE.json")
    assert gate["complete"] is False
    assert any("auditing contaminated" in str(f) for f in gate["failures"])


def synthetic_timing_analysis(
    monkeypatch, candidate_seconds, baseline_seconds, *, failed_row=False, producer_failures=()
):
    """Run the real analyzer with only in-memory synthetic producer records."""
    import pandas as pd

    root = Path("/synthetic/request-timing-validation")
    plan = dict(
        stage="timing",
        source_sha="a" * 40,
        jobs=[dict(index=i, inputs=[f"synthetic-{i}"]) for i in range(2)],
        arms=[["mosaickv_exact", "query3"], ["prefixkv_reimpl", "lossless"]],
        timed_repetitions=20,
        selected_strongest_baseline="prefixkv_reimpl:lossless",
    )
    rows = {
        index: [
            dict(
                sample_id=f"synthetic-{index}",
                method=method,
                precision=precision,
                trial=trial,
                warmup=False,
                status="completed",
                greedy_ids=[0] * 128,
                end_to_end_seconds=(
                    candidate_seconds if method == "mosaickv_exact" else baseline_seconds
                ),
                whole_request_allocator_peak_certified=True,
                source_KV_owner_retired_before_decode=True,
                audit_enabled=False,
            )
            for trial in range(20)
            for method, precision in plan["arms"]
        ]
        for index in range(2)
    }
    if failed_row:
        failed = rows[0][0]
        failed.update(status="failed", error="synthetic failed request")
        for field in (
            "end_to_end_seconds",
            "greedy_ids",
            "whole_request_allocator_peak_certified",
            "source_KV_owner_retired_before_decode",
        ):
            del failed[field]
    producer = dict(
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256="bound",
        outputs={"raw.jsonl": "b" * 64},
    )

    def synthetic_text(path, *args, **kwargs):
        assert path.name == "raw.jsonl", "only synthetic raw rows may be read"
        return "\n".join(json.dumps(row) for row in rows[int(path.parent.name)]) + "\n"

    def forbidden(*args, **kwargs):
        pytest.fail("timing analysis must not submit jobs or materialize a cohort")

    def synthetic_producer(path):
        return dict(
            producer,
            failures=list(producer_failures) if path.parent.name == "0" else [],
        )

    written = {}
    monkeypatch.setattr(module.common, "inputs", lambda *a: plan)
    monkeypatch.setattr(module, "read", synthetic_producer)
    monkeypatch.setattr(module, "sha", lambda *a: "b" * 64)
    monkeypatch.setattr(Path, "exists", lambda path: path.name in ("result.json", "raw.jsonl"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: None)
    monkeypatch.setattr(Path, "read_text", synthetic_text)
    monkeypatch.setattr(Path, "iterdir", lambda *a: iter(()))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *a, **k: None)
    monkeypatch.setattr(
        module, "write_once", lambda path, value: written.update({path.name: value})
    )
    monkeypatch.setattr(module.launch, "submit", forbidden)
    monkeypatch.setattr(module, "materialize_cohort", forbidden)
    module.analyze(root, "bound")
    return written


@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize(
    "seconds",
    [True, False, "1.0", None, float("nan"), float("inf"), float("-inf"), 0, -1.0, 10**400],
)
def test_full_timing_analyzer_rejects_invalid_raw_timings_before_bootstrap(
    monkeypatch, side, seconds
):
    import numpy as np

    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda *a, **k: pytest.fail("invalid raw timings reached bootstrap sampling"),
    )
    with pytest.raises(ValueError, match="positive finite plain numbers"):
        synthetic_timing_analysis(
            monkeypatch,
            seconds if side == "candidate" else 1.0,
            seconds if side == "baseline" else 2.0,
        )


@pytest.mark.parametrize("candidate,baseline", [(1, 2), (1.0, 2.0), (1.25, 2.5)])
def test_valid_raw_timing_numbers_preserve_registered_statistics_and_scope(
    monkeypatch, candidate, baseline
):
    written = synthetic_timing_analysis(monkeypatch, candidate, baseline)
    assert written["GATE.json"]["complete"] is True
    timing = written["paired_latency.json"]
    assert timing["latency_win"] is True
    assert timing["strongest_baseline"] == "prefixkv_reimpl:lossless"
    assert timing["independent_timing_input_count"] == 2
    assert timing["no_general_superiority_claim_from_two_timing_inputs"] is True
    assert all(row["technical_ci95"] == [0.5, 0.5] for row in timing["comparisons"])
    assert all(row["repeat_count"] == 20 for row in timing["comparisons"])
    assert written["GATE.json"]["superiority_established"] is False
    assert written["manifest.json"]["no_paper_promotion"] is True


@pytest.mark.parametrize("producer_failed", [False, True])
def test_failed_request_without_timing_preserves_negative_analysis(monkeypatch, producer_failed):
    import numpy as np

    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda *a, **k: pytest.fail("failed requests must not reach bootstrap sampling"),
    )
    failure = dict(sample_id="synthetic-0", error="synthetic producer failure")
    written = synthetic_timing_analysis(
        monkeypatch,
        1.0,
        2.0,
        failed_row=True,
        producer_failures=(failure,) if producer_failed else (),
    )
    gate = written["GATE.json"]
    assert gate["complete"] is False
    assert gate["attempted_rows"] == gate["expected_rows"] == 80
    assert gate["successful_rows"] == 79
    assert gate["failures"]
    if producer_failed:
        assert failure in gate["failures"]
    assert "paired_latency.json" not in written
    assert gate["superiority_established"] is False
    assert written["manifest.json"]["no_paper_promotion"] is True


def test_failed_producer_does_not_hide_malformed_completed_timings(monkeypatch):
    import numpy as np

    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda *a, **k: pytest.fail("invalid completed timings must not reach bootstrap"),
    )
    with pytest.raises(ValueError, match="positive finite plain numbers"):
        synthetic_timing_analysis(
            monkeypatch,
            True,
            2.0,
            failed_row=True,
            producer_failures=(dict(error="synthetic failed producer"),),
        )
