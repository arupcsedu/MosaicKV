"""One accounting query per event; no scheduler or historical jobs are touched."""

import evicastkv_completion_v14 as completion
import pytest

from mosaickv.evaluation.fox_campaign import write_once


def accounting(monkeypatch, text):
    calls = []
    monkeypatch.setattr(completion.os, "getuid", lambda: 123)
    monkeypatch.setattr(
        completion.subprocess,
        "check_output",
        lambda command, **kwargs: calls.append(command) or text,
    )
    return calls


def test_one_sacct_query_omits_purged_terminal_states_and_keeps_active(monkeypatch):
    calls = accounting(
        monkeypatch,
        "10|10|COMPLETED|123\n11|11|FAILED|123\n12|12|PENDING|123\n"
        "13|13|RUNNING|123\n14|14|CANCELLED by 123|123\n15|15|OUT_OF_MEMORY|123\n",
    )
    retained, receipt = completion.resolve_dependencies(["10", "11", "12", "13", "14", "15"])
    assert retained == ["12", "13"]
    assert receipt["terminal_omitted"] == ["10", "11", "14", "15"]
    assert receipt["terminal_is_not_success"]
    assert len(calls) == receipt["queries"] == 1
    assert "--array" in calls[0] and "--duplicates" in calls[0]


def test_expanded_arrays_and_raw_numeric_child_ids_match_exact_requested_scope(monkeypatch):
    accounting(
        monkeypatch,
        "20_0|200|COMPLETED|123\n20_1|201|TIMEOUT|123\n21_0|210|COMPLETED|123\n21_1|211|RUNNING|123\n",
    )
    retained, receipt = completion.resolve_dependencies(["20", "21", "201"])
    assert retained == ["21"]
    assert receipt["terminal_omitted"] == ["20", "201"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "10|10|COMPLETED|999\n",
        "10|10|COMPLETED+|123\n",
        "10|10|COMPLETED|123\n10|10|COMPLETED|123\n",
        "10.batch|10.batch|COMPLETED|123\n",
        "11|11|COMPLETED|123\n",
        "10|10|UNKNOWN|123\n",
        "10|COMPLETED|123\n",
    ],
)
def test_missing_foreign_duplicate_unknown_or_ambiguous_states_fail_closed(monkeypatch, text):
    calls = accounting(monkeypatch, text)
    with pytest.raises(completion.DependencyResolutionError) as error:
        completion.resolve_dependencies(["10"])
    assert not error.value.receipt["passed"]
    assert error.value.receipt["queries"] == len(calls) == 1


def test_empty_dependencies_never_query_and_invalid_ids_never_submit(monkeypatch):
    monkeypatch.setattr(
        completion.subprocess, "check_output", lambda *a, **k: pytest.fail("must not query")
    )
    retained, receipt = completion.resolve_dependencies([])
    assert not retained and receipt["queries"] == 0
    with pytest.raises(ValueError):
        completion.resolve_dependencies(["12;sbatch"])


def ledger(path, source, campaign, mode, job, history=False):
    script = (
        source / f"mosaickv/slurm/evicastkv_{'history_v13' if history else 'review_v13'}.sbatch"
    )
    target = campaign / "history" if history else campaign
    write_once(
        path,
        dict(
            job=job,
            command=[
                "sbatch",
                "--parsable",
                "--partition=bii",
                str(script),
                str(source),
                str(target),
                mode,
            ],
        ),
    )


def test_frontier_uses_latest_acknowledged_analyzer_not_worker_or_unacknowledged_request(tmp_path):
    source, campaign = tmp_path / "source", tmp_path / "campaign"
    ledger(
        campaign / "history/jobs/attach-0-submission.json", source, campaign, "attach", "10", True
    )
    ledger(
        campaign / "history/jobs/after-segment-2-submission.json",
        source,
        campaign,
        "advance",
        "20",
        True,
    )
    write_once(
        campaign / "history/jobs/after-segment-3-request.json", dict(command=["not acknowledged"])
    )
    current, job = completion.frontier(campaign, source)
    assert job == "20" and current["rank"] == 6


def synthetic_parent(tmp_path):
    source, campaign = tmp_path / "source", tmp_path / "campaign"
    for index, (lane, name) in enumerate(completion.STATIC.items()):
        ledger(
            campaign / f"jobs/after-{lane}-submission.json",
            source,
            campaign,
            f"after-{lane}",
            str(100 + index),
        )
        write_once(
            campaign / lane / name,
            dict(source_sha=completion.PARENT_SHA, status="negative preserved"),
        )
    ledger(
        campaign / "history/jobs/after-segment-1-submission.json",
        source,
        campaign,
        "advance",
        "200",
        True,
    )
    return source, campaign


def test_rescue_event_reattaches_afterany_without_submitting_scans_or_GPU(tmp_path, monkeypatch):
    source, campaign = synthetic_parent(tmp_path)
    root = tmp_path / "rescue"
    monkeypatch.setattr(
        completion,
        "load",
        lambda p: dict(
            source_sha="current", parent_campaign=str(campaign), parent_source=str(source)
        ),
    )
    monkeypatch.setattr(
        completion,
        "resolve_dependencies",
        lambda ids: (
            ["200"],
            dict(
                retained=["200"],
                terminal_omitted=["100", "101", "102", "103"],
                queries=1,
                passed=True,
            ),
        ),
    )
    calls = []
    monkeypatch.setattr(
        completion.subprocess,
        "check_output",
        lambda command, **kwargs: calls.append(command) or "300",
    )
    result = completion.advance(root, 0)
    assert result["status"] == "attached_to_existing_CPU_callbacks"
    assert len(calls) == 1 and "--dependency=afterany:200" in calls[0]
    assert not any("--gres" in item or "--gpus" in item for item in calls[0])
    assert calls[0][-1] == "1"


def test_terminal_history_without_gate_blocks_instead_of_busy_reattachment(tmp_path, monkeypatch):
    source, campaign = synthetic_parent(tmp_path)
    root = tmp_path / "rescue"
    monkeypatch.setattr(
        completion,
        "load",
        lambda p: dict(
            source_sha="current", parent_campaign=str(campaign), parent_source=str(source)
        ),
    )
    monkeypatch.setattr(
        completion,
        "resolve_dependencies",
        lambda ids: ([], dict(retained=[], terminal_omitted=ids, queries=1, passed=True)),
    )
    monkeypatch.setattr(
        completion, "submit", lambda *a, **k: pytest.fail("must not reattach terminal history")
    )
    gate = completion.advance(root, 0)
    assert not gate["completed"] and gate["new_GPU_jobs"] == gate["new_scans"] == 0
    assert (root / "events/0/dependency-resolution.json").is_file()


def test_verified_existing_report_reused_and_history_submission_failure_preserved(
    tmp_path, monkeypatch
):
    source, campaign = synthetic_parent(tmp_path)
    root = tmp_path / "rescue"
    write_once(
        campaign / "history/GATE_history_v13.json",
        dict(source_sha=completion.PARENT_SHA, bindings={}, status="finite_not_fresh"),
    )
    write_once(
        campaign / "history/jobs/completion-submission-error.json",
        dict(error="Invalid Job ID", automatic_retry=False),
    )
    write_once(campaign / "FINAL_report.json", dict(already="verified by stub"))
    monkeypatch.setattr(
        completion,
        "load",
        lambda p: dict(
            source_sha="current", parent_campaign=str(campaign), parent_source=str(source)
        ),
    )
    monkeypatch.setattr(
        completion,
        "resolve_dependencies",
        lambda ids: ([], dict(retained=[], terminal_omitted=ids, queries=1, passed=True)),
    )
    monkeypatch.setattr(
        completion,
        "verify_report",
        lambda p: dict(
            path=str(p / "FINAL_report.json"), sha256=completion.sha(p / "FINAL_report.json")
        ),
    )
    monkeypatch.setattr(
        completion, "invoke_finish", lambda *a: pytest.fail("must not overwrite existing report")
    )
    gate = completion.advance(root, 0)
    assert gate["completed"] and gate["status"] == "reused_verified_parent_report"
    assert len(gate["original_history_submission_failures"]) == 1
    assert not gate["superiority_established"]


def test_external_finish_uses_distinct_V13_environment_and_no_V14_imports(tmp_path, monkeypatch):
    root = tmp_path / "rescue"
    root.mkdir()
    source, campaign = tmp_path / "v13source", tmp_path / "campaign"
    monkeypatch.setenv("PYTHONPATH", "forbidden-v14-source")
    calls = []
    monkeypatch.setattr(
        completion.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    monkeypatch.setattr(completion, "verify_report", lambda path: dict(verified=True))
    assert completion.invoke_finish(root, campaign, source)["verified"]
    command, options = calls[0]
    assert options["cwd"] == source and "PYTHONPATH" not in options["env"]
    assert str(source) in command and "mosaickv_review_v13" not in command
    assert 'PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"' in command[4]
    assert "evicastkv_review_v13.py finish" in command[4]
    assert options["timeout"] == 300 and options["check"]


def test_hop_limit_and_duplicate_scheduler_ack_failure_never_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "load", lambda p: dict(source_sha="current"))
    with pytest.raises(ValueError, match="bounded"):
        completion.advance(tmp_path, completion.MAX_HOPS)

    def fail(*a, **k):
        raise RuntimeError("lost acknowledgement")

    monkeypatch.setattr(completion.subprocess, "check_output", fail)
    with pytest.raises(RuntimeError):
        completion.submit(tmp_path, 1, ["10"])
    with pytest.raises(FileExistsError):
        completion.submit(tmp_path, 1, ["10"])
    assert not completion.read(tmp_path / "jobs/hop-1-submission-error.json")["automatic_retry"]


def test_known_prepare_failure_completes_report_only_with_three_real_gates(tmp_path, monkeypatch):
    source, campaign = tmp_path / "source", tmp_path / "campaign"
    gates = {}
    for index, (lane, name) in enumerate(completion.STATIC.items()):
        path = campaign / lane / name
        if lane == "providers":
            gates[lane] = dict(status="missing_gate", path=str(path))
            continue
        ledger(
            campaign / f"jobs/after-{lane}-submission.json",
            source,
            campaign,
            f"after-{lane}",
            str(100 + index),
        )
        value = dict(
            source_sha=completion.PARENT_SHA,
            status="scientific result remains independently scoped",
        )
        write_once(path, value)
        gates[lane] = dict(path=str(path), sha256=completion.sha(path), evidence=value)
    errorpath = campaign / "errors/prepare-providers.json"
    error = dict(
        status="blocked", error="ValueError: canonical source alias rejected", traceback="preserved"
    )
    write_once(errorpath, error)
    historypath = campaign / "history/GATE_history_v13.json"
    history = dict(source_sha=completion.PARENT_SHA, bindings={}, status="finite_not_fresh")
    write_once(historypath, history)
    gates["history"] = dict(
        path=str(historypath), sha256=completion.sha(historypath), evidence=history
    )
    write_once(
        campaign / "FINAL_report.json",
        dict(
            source_sha=completion.PARENT_SHA,
            gates=gates,
            errors=[dict(path=str(errorpath), sha256=completion.sha(errorpath), evidence=error)],
            superiority_established=False,
            fresh_confirmation_established=False,
            paper_campaign_authorized=False,
        ),
    )
    root = tmp_path / "rescue"
    monkeypatch.setattr(
        completion,
        "load",
        lambda p: dict(
            source_sha="current", parent_campaign=str(campaign), parent_source=str(source)
        ),
    )
    queried = []

    def resolve(ids):
        queried.extend(ids)
        return [], dict(retained=[], terminal_omitted=ids, queries=1, passed=True)

    monkeypatch.setattr(completion, "resolve_dependencies", resolve)
    monkeypatch.setattr(completion, "submit", lambda *a, **k: pytest.fail("no new jobs needed"))
    gate = completion.advance(root, 0)
    assert gate["completed"] and gate["completed_with_recorded_static_failures"]
    assert set(gate["recorded_static_failures"]) == {"providers"}
    assert len(queried) == 3 and not gate["superiority_established"]
    assert completion.verify_report(campaign)["sha256"] == completion.sha(
        campaign / "FINAL_report.json"
    )


def test_unacknowledged_scheduler_request_cannot_be_hidden_by_prepare_failure(
    tmp_path, monkeypatch
):
    source, campaign = synthetic_parent(tmp_path)
    write_once(campaign / "jobs/providers-request.json", dict(command=["sbatch"]))
    write_once(
        campaign / "errors/prepare-providers.json",
        dict(status="blocked", error="lost acknowledgement"),
    )
    root = tmp_path / "rescue"
    monkeypatch.setattr(
        completion,
        "load",
        lambda p: dict(
            source_sha="current", parent_campaign=str(campaign), parent_source=str(source)
        ),
    )
    monkeypatch.setattr(
        completion,
        "resolve_dependencies",
        lambda ids: pytest.fail("must not infer unacknowledged state"),
    )
    gate = completion.advance(root, 0)
    assert not gate["completed"] and "Unacknowledged" in gate["error"]


def ownership_wrapper(campaign):
    folder = campaign / "ownership"
    trace = folder / "worker/allocator_trace_pre_attribution.json.gz"
    trace.parent.mkdir(parents=True)
    trace.write_bytes(b"synthetic complete trace bytes; not a real GPU trace")
    protocol = dict(
        source_sha=completion.PARENT_SHA, policy={"observer_variant": completion.OWNERSHIP_WRAPPER}
    )
    write_once(folder / "protocol.json", protocol)
    inner = dict(
        source_sha=completion.PARENT_SHA,
        status="negative scientific result preserved",
        source_whole_request_KV_peak_certified=False,
    )
    write_once(folder / "GATE_source_ownership_closure.json", inner)
    receipt = dict(
        source_sha=completion.PARENT_SHA,
        observer_variant=completion.OWNERSHIP_WRAPPER,
        protocol_sha256=completion.sha(folder / "protocol.json"),
        trace_path=str(trace),
        trace_sha256=completion.sha(trace),
    )
    write_once(folder / "worker/trace_pre_attribution_receipt.json", receipt)
    outer = dict(
        observer_variant=completion.OWNERSHIP_WRAPPER,
        parent_gate_sha256=completion.sha(folder / "GATE_source_ownership_closure.json"),
        trace_first_receipt_sha256=completion.sha(
            folder / "worker/trace_pre_attribution_receipt.json"
        ),
        raw_trace_saved_before_attribution=True,
        exact_interval_predicate_unchanged=True,
        source_whole_request_KV_peak_certified=False,
        whole_device_memory_certified=False,
        source_gate=inner,
        fresh_confirmation_authorized=False,
        superiority_established=False,
    )
    write_once(folder / completion.STATIC["ownership"], outer)
    return folder, outer


def test_only_exact_pinned_ownership_wrapper_has_nested_provenance(tmp_path):
    folder, outer = ownership_wrapper(tmp_path)
    proof = completion.static_gate_provenance(tmp_path, "ownership")
    assert proof["source_sha"] == completion.PARENT_SHA
    assert proof["kind"] == "exact_versioned_ownership_wrapper"
    assert proof["gate_outcome_not_promoted"]
    assert len(proof["bindings"]) == 5
    assert completion.read(folder / completion.STATIC["ownership"]) == outer
    (folder / "worker/allocator_trace_pre_attribution.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="trace-first"):
        completion.static_gate_provenance(tmp_path, "ownership")


@pytest.mark.parametrize(
    "mutation", ["variant", "inner_source", "extra_top_source", "credible_top_conflict", "receipt"]
)
def test_wrapper_mismatch_cannot_be_waived_by_a_nested_sha(tmp_path, mutation):
    import json

    folder, outer = ownership_wrapper(tmp_path)
    if mutation == "variant":
        outer["observer_variant"] = "unreviewed_other_wrapper"
    elif mutation == "inner_source":
        outer["source_gate"]["source_sha"] = "foreign"
    elif mutation == "extra_top_source":
        outer["source_sha"] = "foreign"
    elif mutation == "credible_top_conflict":
        outer["source_sha"] = completion.PARENT_SHA
        outer["source_gate"]["source_sha"] = "foreign"
    else:
        outer["trace_first_receipt_sha256"] = "0" * 64
    (folder / completion.STATIC["ownership"]).write_text(json.dumps(outer))
    with pytest.raises(ValueError):
        completion.static_gate_provenance(tmp_path, "ownership")


def test_nested_producer_fallback_not_enabled_for_other_methods(tmp_path):
    write_once(
        tmp_path / "lookm" / completion.STATIC["lookm"],
        dict(source_gate={"source_sha": completion.PARENT_SHA}),
    )
    with pytest.raises(ValueError, match="differently produced"):
        completion.static_gate_provenance(tmp_path, "lookm")
