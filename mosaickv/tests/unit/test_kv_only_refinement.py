"""Pinned source proof tests; no actual trace rules or GPU work are created."""

import hashlib
import json

import evicastkv_kv_only_refinement as study
import pytest
from test_source_trace_adjudication import fixture, rule_for


def context(tmp_path):
    code = (
        "def cache_operator():\n    return 1\n\ndef pointwise_operator():\n    return 2\n\n"
        "def audit_request(adapter, session):\n"
        "    session.mark((*adapter.model.parameters(), *adapter.model.buffers()), 'model', "
        "'pinned immutable resident model parameters/buffers')\n\n"
        "def chosen(output, ledger):\n"
        "    ledger.track('selected_formats', output.tensors())\n"
    )
    trace, conservative, declared, _, file = fixture(tmp_path, code=code)
    declared["observations"][0]["owners"][0]["roles"] = ["original_prefill_KV"]
    ledger = dict(
        schema_version=1,
        boundary_policy_sha256=study.digest(study.POLICY),
        review_status="explicit_source_inspection_complete",
        reviewed_by="synthetic fixture review",
        stack_rules=[],
        model_identity_witnesses=[],
    )
    return trace, conservative, declared, ledger, file


def rule(trace, file, index, uses):
    value = rule_for(trace, file, stack_id=index, role="kv_workspace")
    value.pop("role")
    return dict(value, categories=uses)


def test_no_source_rules_does_not_trust_any_saved_allocator_role(tmp_path):
    trace, _, declared, ledger, _ = context(tmp_path)
    assert study.checked_assignments(trace, declared, ledger, source_bindings={}) == {}


def test_exact_source_operator_proofs_separate_payload_and_normal_activation(tmp_path):
    trace, _, declared, ledger, file = context(tmp_path)
    ledger["stack_rules"] = [
        rule(trace, file, 0, ["cache_payload"]),
        rule(trace, file, 1, ["normal_model_activation"]),
    ]
    uses = study.checked_assignments(trace, declared, ledger, source_bindings={})
    assert uses[(10, 0)] == {"cache_payload"}
    assert uses[(20, 0)] == {"normal_model_activation"}


def test_known_cache_owner_cannot_be_excluded_by_source_rule(tmp_path):
    trace, _, declared, ledger, file = context(tmp_path)
    ledger["stack_rules"] = [rule(trace, file, 0, ["normal_model_activation"])]
    with pytest.raises(ValueError, match="conflicts"):
        study.checked_assignments(trace, declared, ledger, source_bindings={})
    ledger["stack_rules"][0]["categories"].append("cache_payload")
    assert study.checked_assignments(trace, declared, ledger, source_bindings={})[(10, 0)] == {
        "cache_payload",
        "normal_model_activation",
    }


def test_changed_source_or_generic_fallback_cannot_supply_exclusion(tmp_path):
    trace, _, declared, ledger, file = context(tmp_path)
    ledger["stack_rules"] = [rule(trace, file, 1, ["normal_model_activation"])]
    file.write_text(file.read_text() + "# modified\n")
    with pytest.raises(ValueError, match="source"):
        study.checked_assignments(trace, declared, ledger, source_bindings={})
    ledger["stack_rules"][0]["fallback"] = "nonKV"
    with pytest.raises(ValueError, match="schema"):
        study.checked_assignments(trace, declared, ledger, source_bindings={})


def model_witness(trace, file):
    lines = file.read_text().splitlines(keepends=True)
    line = next(i + 1 for i, text in enumerate(lines) if "session.mark" in text)
    return dict(
        address=1,
        generation=0,
        evidence_sha256=study.digest(trace["allocator_records"][0]["evidence"]),
        reason="Exact synthetic resident parameter/buffer identity witness, not inherited role.",
        reviewed_by="synthetic review",
        source=dict(
            filename=str(file),
            function="audit_request",
            line=line,
            source_sha256=study.sha(file),
            source_line_sha256=hashlib.sha256(lines[line - 1].encode()).hexdigest(),
            qualified_function="audit_request",
        ),
    )


def test_model_baseline_requires_actual_identity_evidence_and_source_witness(tmp_path):
    trace, _, declared, ledger, file = context(tmp_path)
    ledger["model_identity_witnesses"] = [model_witness(trace, file)]
    assert study.checked_assignments(
        trace, declared, ledger, source_bindings={str(file): study.sha(file)}
    )[(1, 0)] == {"model_weights"}
    trace["allocator_records"][0]["evidence"] = []
    with pytest.raises(ValueError, match="identity"):
        study.checked_assignments(
            trace, declared, ledger, source_bindings={str(file): study.sha(file)}
        )


def selected_generation_rule(trace, declared, file):
    lines = file.read_text().splitlines(keepends=True)
    line = next(i + 1 for i, text in enumerate(lines) if "ledger.track" in text)
    observation = declared["observations"][0]
    owner = observation["owners"][0]
    owner["roles"] = ["selected_formats"]
    return dict(
        address=10,
        generation=0,
        stack_id=0,
        categories=["cache_payload"],
        reason="Selected packet owner is matched uniquely inside its actual allocation lifetime.",
        reviewed_by="synthetic source review",
        owner_witness=dict(
            observation_sha256=study.digest(observation),
            owner_sha256=study.digest(owner),
            required_tag="selected_formats",
            source=dict(
                filename=str(file),
                function="chosen",
                line=line,
                source_sha256=study.sha(file),
                source_line_sha256=hashlib.sha256(lines[line - 1].encode()).hexdigest(),
                qualified_function="chosen",
            ),
        ),
    )


def test_selected_and_unselected_same_stack_can_refine_only_actual_owner_generation(tmp_path):
    trace, _, declared, ledger, file = context(tmp_path)
    ledger["stack_rules"] = [rule(trace, file, 0, ["cache_constructor_workspace"])]
    ledger["generation_rules"] = [selected_generation_rule(trace, declared, file)]
    uses = study.checked_assignments(
        trace, declared, ledger, source_bindings={str(file): study.sha(file)}
    )
    assert uses[(10, 0)] == {"cache_constructor_workspace", "cache_payload"}


def test_proof_only_stack_does_not_classify_or_exclude_any_generation(tmp_path):
    from mosaickv.measurements.cache_only_refinement import reduce_generations

    trace, _, declared, ledger, file = context(tmp_path)
    ledger["stack_rules"] = [rule(trace, file, 0, []), rule(trace, file, 1, [])]
    uses = study.checked_assignments(trace, declared, ledger, source_bindings={})
    assert uses == {}
    result = reduce_generations(
        trace["allocator_records"], uses, complete_request=True, history_complete=True
    )
    assert result["unknown_generations"] == len(trace["allocator_records"])
    assert all(
        not m["scalar_interval_bounds_equal"] and not m["exact_peak_certified"]
        for m in result["metrics"].values()
    )


def test_proof_only_creator_can_refine_one_witness_and_leave_other_generations_unknown(tmp_path):
    trace, _, declared, ledger, file = context(tmp_path)
    ledger["stack_rules"] = [rule(trace, file, 0, []), rule(trace, file, 1, [])]
    ledger["generation_rules"] = [selected_generation_rule(trace, declared, file)]
    uses = study.checked_assignments(
        trace, declared, ledger, source_bindings={str(file): study.sha(file)}
    )
    assert uses == {(10, 0): {"cache_payload"}}


def test_proof_only_path_cannot_be_used_for_arbitrary_nonKV_exclusion(tmp_path):
    trace, _, declared, ledger, file = context(tmp_path)
    ledger["stack_rules"] = [rule(trace, file, 0, [])]
    entry = selected_generation_rule(trace, declared, file)
    entry["categories"] = ["normal_model_activation"]
    ledger["generation_rules"] = [entry]
    with pytest.raises(ValueError, match="owner witness"):
        study.checked_assignments(
            trace, declared, ledger, source_bindings={str(file): study.sha(file)}
        )


@pytest.mark.parametrize("defect", ["timestamp", "generation", "tag", "instrument_source"])
def test_generation_owner_refinement_rejects_ambiguous_or_forged_witness(tmp_path, defect):
    trace, _, declared, ledger, file = context(tmp_path)
    ledger["stack_rules"] = [rule(trace, file, 0, ["cache_constructor_workspace"])]
    entry = selected_generation_rule(trace, declared, file)
    if defect == "timestamp":
        declared["observations"][0]["begin_us"] = 0
        entry["owner_witness"]["observation_sha256"] = study.digest(declared["observations"][0])
    elif defect == "generation":
        entry["generation"] = 1
    elif defect == "tag":
        entry["owner_witness"]["required_tag"] = "original_prefill_KV"
    else:
        entry["owner_witness"]["source"]["source_sha256"] = "0" * 64
    ledger["generation_rules"] = [entry]
    with pytest.raises(ValueError):
        study.checked_assignments(
            trace, declared, ledger, source_bindings={str(file): study.sha(file)}
        )


def test_actual_refinement_needs_complete_snapshot_bound_bundle(tmp_path, monkeypatch):
    trace, conservative, declared, _, _ = context(tmp_path)
    monkeypatch.setattr(study, "SCRATCH", tmp_path)
    monkeypatch.setattr(study, "clean_source", lambda *a: "a" * 40)
    monkeypatch.setattr(study, "cpu_gate", lambda *a: dict(status="passed"))
    monkeypatch.setattr(
        study.audit,
        "_inputs",
        lambda *a: (
            dict(
                trace=trace,
                declared=declared,
                conservative=conservative,
                producer_sha="a" * 40,
                worker_complete=True,
            ),
            {},
            {},
        ),
    )
    with pytest.raises(ValueError, match="snapshot-bound"):
        study.run(
            tmp_path / "derived",
            parent=tmp_path,
            reviewed_rules=tmp_path / "none",
            cpu_gate_root=tmp_path,
        )


def test_end_to_end_synthetic_source_review_certifies_only_named_cache_boundaries(
    tmp_path, monkeypatch
):
    trace, conservative, declared, ledger, file = context(tmp_path)
    trace["final_allocator_snapshot"] = {"segments": [], "synthetic": True}
    ledger["stack_rules"] = [
        rule(trace, file, 0, ["cache_payload"]),
        rule(trace, file, 1, ["normal_model_activation"]),
    ]
    ledger["model_identity_witnesses"] = [model_witness(trace, file)]
    ledger.update(producer_sha="a" * 40, artifact_hashes={})
    rules = tmp_path / "reviewed.json"
    rules.write_text(json.dumps(ledger))
    (tmp_path / "protocol.json").write_text(json.dumps(dict(bindings={str(file): study.sha(file)})))
    monkeypatch.setattr(study, "SCRATCH", tmp_path)
    monkeypatch.setattr(study, "clean_source", lambda *a: "a" * 40)
    monkeypatch.setattr(study, "cpu_gate", lambda *a: dict(status="passed"))
    monkeypatch.setattr(
        study.audit,
        "_inputs",
        lambda *a: (
            dict(
                trace=trace,
                declared=declared,
                conservative=conservative,
                producer_sha="a" * 40,
                worker_complete=True,
            ),
            {},
            {},
        ),
    )
    gate = study.run(
        tmp_path / "derived", parent=tmp_path, reviewed_rules=rules, cpu_gate_root=tmp_path
    )
    assert all(metric["exact_peak_certified"] for metric in gate["metrics"].values())
    assert not gate["whole_device_memory_certified"] and not gate["superiority_established"]
    assert not gate["numerical_or_quality_guard_promoted"]
    assert not gate["historical_KV_roles_modified"]
