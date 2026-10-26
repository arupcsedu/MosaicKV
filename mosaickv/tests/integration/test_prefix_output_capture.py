"""Random-weight CPU hook/isolation tests; no measured checkpoint quality."""

import pytest
import torch
from test_hf_tiny_models import _llava

from mosaickv.backends.hf_prefix_outputs import PrefixOutputRecorder
from mosaickv.backends.hf_runtime import _clone_decode_state


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def counts(adapter):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in adapter.model.modules()]


def test_hooks_capture_actual_gqa_heads_projection_and_residuals_without_changing_logits():
    torch.manual_seed(37)
    adapter = _llava()
    prefill = adapter.prefill(
        adapter.prepare_inputs([dict(role="user", content="tiny prompt")]), capture_queries=False
    )
    original = _clone_decode_state(adapter, prefill.state)
    copy = _clone_decode_state(adapter, prefill.state)
    baseline = adapter.decode_one_token(prefill.next_token_id, original, capture_queries=False)
    recorder = PrefixOutputRecorder(adapter)
    before = counts(adapter)
    with recorder.capture("candidate", 1):
        out = adapter.decode_one_token(prefill.next_token_id, copy, capture_queries=False)
    assert torch.equal(out.logits, baseline.logits)
    assert counts(adapter) == before and not recorder.active
    assert recorder.storage_bytes > 0
    rows = recorder.records["candidate"][1]
    assert len(rows) == 2
    for layer, row in zip(recorder.layers, rows, strict=True):
        assert set(row) == set(recorder.FIELDS)
        assert row["attention_heads"].shape == (2, 8)  # two query heads, one KV head
        with torch.no_grad():
            projected = layer.self_attn.o_proj(row["attention_heads"].flatten())
        torch.testing.assert_close(projected, row["projected_attention"])
        assert torch.equal(
            row["residual_input"] + row["projected_attention"], row["post_attention_residual"]
        )
    assert torch.equal(rows[0]["layer_output"], rows[1]["residual_input"])
    with pytest.raises(ValueError, match="duplicate"), recorder.capture("candidate", 1):
        pass


def test_hooks_are_removed_on_errors_and_partial_captures_are_not_published():
    adapter = _llava()
    recorder = PrefixOutputRecorder(adapter)
    before = counts(adapter)
    with pytest.raises(RuntimeError, match="fixture error"), recorder.capture("candidate", 1):
        raise RuntimeError("fixture error")
    assert counts(adapter) == before and recorder.records == {} and not recorder.active
    with pytest.raises(RuntimeError, match="incomplete"), recorder.capture("candidate", 1):
        pass
    assert counts(adapter) == before and not recorder.active
    with (
        pytest.raises(ValueError, match="nested"),
        recorder.capture("candidate", 1),
        recorder.capture("candidate", 2),
    ):
        pass
    assert counts(adapter) == before and not recorder.active


def test_frozen_output_replay_matches_uninstrumented_cpu_trajectories(tmp_path):
    from test_bounded_history_runtime import _config
    from test_hf_tiny_models import _runtime_request

    from mosaickv.backends.hf_allocation_guard import packed_positions_sha
    from mosaickv.evaluation.draft_head_diagnostic import DraftConditionedHeadDiagnostic
    from mosaickv.evaluation.head_substitution_followup import HeadSubstitutionFollowup
    from mosaickv.evaluation.prefix_output_replay import PrefixOutputReplay
    from mosaickv.types import MosaicKVMethod

    torch.manual_seed(37)
    adapter = _llava()
    cfg = _config(MosaicKVMethod.MOSAICKV_EXACT)

    class FixtureReference(HeadSubstitutionFollowup):
        def _observe_exact_anchor(self, prefill, full_state, anchor, attention, budget):
            self.followup["anchor_positions_sha256"] = packed_positions_sha(anchor)
            return super()._observe_exact_anchor(prefill, full_state, anchor, attention, budget)

    reference = FixtureReference(
        adapter,
        cfg,
        trace_directory=tmp_path / "reference",
        followup=dict(anchor_positions_sha256="fixture-only", survivors=[]),
    )
    assert reference.diagnose(_runtime_request("reference"))["controls_passed"]
    draft = DraftConditionedHeadDiagnostic(
        adapter,
        cfg,
        trace_directory=tmp_path / "draft",
        anchor_positions_sha256=reference.followup["anchor_positions_sha256"],
        reference_logits={role: reference.saved_logits[role] for role in ("fullkv", "anchor")},
    )
    frozen = draft.diagnose(_runtime_request("draft"))
    assert frozen["proposals"] and not frozen["selection"]["accepted"]
    replay = PrefixOutputReplay(
        adapter,
        cfg,
        trace_directory=tmp_path / "replay",
        frozen_record=frozen,
        reference_logits=draft.saved_logits,
        draft_evidence=draft.saved_evidence,
    )
    result = replay.diagnose(_runtime_request("replay"))
    assert result["instrumentation_passed"] and result["instrumented_logits_bitwise_replayed"]
    assert result["intermediate_null_controls_bitwise_equal"] and result["controls_passed"]
    assert result["frozen_proposals_replayed"] == len(frozen["proposals"])
    assert result["candidate_updates"] == frozen["candidate_updates"]
    assert result["active_cap_bytes"] == 512 and result["padding_bytes"] == 0
    assert result["measured_source_cache_unchanged"] and result["logical_positions_unchanged"]
    assert result["live_private_states_after_diagnostic"] == 0
    assert not result["proposal_construction_revised"] and result["accepted_proposals"] == 0
    assert not any(
        result[k]
        for k in (
            "benchmark_labels_accessed",
            "fresh_cohort_authorized",
            "calibration_authorized",
            "paper_campaign_authorized",
        )
    )
    assert set(replay.saved_outputs) == set(draft.saved_logits)
    # Exercise the write-once CPU analyzer on these synthetic tensors too,
    # including real artifact hashes and the manifest's actual schema keys.
    from analyze_prefix_output_replay import analyze
    from bounded_history_pilot import digest, write_once

    analysis_root = tmp_path / "synthetic-analysis-input"
    worker = analysis_root / "workers/0"
    worker.mkdir(parents=True)
    write_once(
        analysis_root / "plan.json",
        dict(
            protocol=dict(
                output_replay_from_v8="synthetic-test-fixture-only", active_cap_bytes=512
            ),
            source_sha="0" * 40,
            bound_files={},
            jobs=[dict(index=0, task="synthetic_fixture", config_sha="1" * 64)],
            fresh_cohort_authorized=False,
            calibration_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    write_once(worker / "diagnostic.json", dict(status="completed", result=result))
    torch.save(replay.saved_outputs, worker / "prefix-outputs.pt")
    torch.save(replay.saved_predictions, worker / "frozen-query-predictions.pt")
    write_once(worker / "artifact-hashes.json", {str(p): digest(p) for p in worker.iterdir()})
    write_once(
        worker / "manifest.json",
        dict(
            source=dict(git_sha="0" * 40, git_dirty=False, config_sha="1" * 64),
            artifacts=dict(
                raw_output_sha=digest(worker / "diagnostic.json"),
                log_sha=digest(worker / "artifact-hashes.json"),
            ),
            execution=dict(plan_sha256=digest(analysis_root / "plan.json")),
        ),
    )
    write_once(
        worker / "completed.json",
        dict(
            manifest=str(worker / "manifest.json"), manifest_sha256=digest(worker / "manifest.json")
        ),
    )
    report = analyze(analysis_root, tmp_path / "derived-analysis")
    assert report["tasks"][0]["proposals"] == len(frozen["proposals"])
    assert report["tasks"][0]["instrumentation_passed"] and not report["paper_campaign_authorized"]
    with pytest.raises(FileExistsError, match="write-once"):
        analyze(analysis_root, tmp_path / "derived-analysis")
    torch.save({}, worker / "prefix-outputs.pt")
    with pytest.raises(ValueError, match="artifact changed"):
        analyze(analysis_root, tmp_path / "derived-tampered")
    for key, predictions in replay.saved_predictions.items():
        proposal = frozen["proposals"][key]
        row = next(
            r
            for r in frozen["pool_trace"]
            if r["layer"] == proposal["layer"] and r["kv_head"] == proposal["head"]
        )
        before = (
            (predictions["predicted_anchor"] - predictions["fullkv_query_target"]).square().mean()
        )
        after = (
            (predictions["predicted_candidate"] - predictions["fullkv_query_target"])
            .square()
            .mean()
        )
        assert float(before) == pytest.approx(row["draft_output_error_before"], rel=1e-5)
        assert float(after) == pytest.approx(row["draft_output_error_after"], rel=1e-5)
    # A changed saved logit is rejected rather than silently changing the replay.
    draft.saved_logits["fullkv"][1][0, 0] += 1
    rejected = PrefixOutputReplay(
        adapter,
        cfg,
        trace_directory=tmp_path / "changed",
        frozen_record=frozen,
        reference_logits=draft.saved_logits,
        draft_evidence=draft.saved_evidence,
    )
    with pytest.raises(RuntimeError, match="instrumentation changed frozen logits"):
        rejected.diagnose(_runtime_request("changed"))


def test_sdpa_capture_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="eager Llama"):
        PrefixOutputRecorder(_llava(attention="sdpa"))
