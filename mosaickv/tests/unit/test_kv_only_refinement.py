"""Pinned source proof tests; no actual trace rules or GPU work are created."""

import ast
import hashlib
import json

import evicastkv_kv_only_refinement as study
import pytest
from test_source_trace_adjudication import fixture, model_expression_fixture, rule_for


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


def empty_ledger(rules):
    return dict(
        schema_version=1,
        boundary_policy_sha256=study.digest(study.POLICY),
        review_status="explicit_source_inspection_complete",
        reviewed_by="synthetic exact-source review",
        stack_rules=rules,
        model_identity_witnesses=[],
    )


def predicate_fixture(tmp_path, suffix, scope, predicate, *, preamble="", argument=""):
    tmp_path.mkdir(parents=True, exist_ok=True)
    trace, _, _, _, _ = fixture(tmp_path)
    file = tmp_path / suffix.lstrip("/")
    file.parent.mkdir(parents=True, exist_ok=True)
    names = scope.split(".")
    prefix = f"class {names[0]}:\n" if len(names) == 2 else ""
    indent = "    " if prefix else ""
    body_indent = indent + "    "
    code = prefix + f"{indent}def {names[-1]}({argument}):\n"
    code += "".join(f"{body_indent}{line}\n" for line in preamble.splitlines())
    code += f"{body_indent}if (\n{body_indent}    {predicate}\n{body_indent}):\n"
    code += f"{body_indent}    raise ValueError('validation')\n"
    file.write_text(code)
    node = next(node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.If))
    trace["stack_table"][1] = [[str(file), names[-1], node.test.lineno]]
    trace["installed_and_local_source_hashes"][str(file)] = dict(
        sha256=study.sha(file), canonical_path=str(file.resolve())
    )
    proof = rule(trace, file, 1, ["cache_auxiliary_workspace"])
    proof["qualified_function"] = scope
    return trace, file, proof


PREDICATE_CASES = [
    (suffix, scope, predicate)
    for (suffix, scope), predicates in study.VALIDATION_PREDICATES.items()
    for predicate in predicates
]


@pytest.mark.parametrize("suffix,scope,predicate", PREDICATE_CASES)
@pytest.mark.parametrize("categories", [[], ["cache_auxiliary_workspace"]])
def test_closed_validation_predicates_are_only_proofs_or_included_auxiliary_workspace(
    tmp_path, suffix, scope, predicate, categories
):
    trace, _, proof = predicate_fixture(tmp_path, suffix, scope, predicate)
    proof["categories"] = categories
    assignments = study.checked_assignments(
        trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
    )
    assert assignments == ({(20, 0): set(categories)} if categories else {})
    assert study.POLICY["auxiliary_validation_and_demand_capture_in_workspace_metric"]
    assert "cache_auxiliary_workspace" in study.POLICY["workspace_classes"]


@pytest.mark.parametrize(
    "categories",
    [
        ["cache_payload"],
        ["normal_model_activation"],
        ["cache_metadata"],
        ["cache_auxiliary_workspace", "normal_query"],
    ],
)
def test_closed_validation_predicate_cannot_exclude_unknown_or_claim_payload(tmp_path, categories):
    trace, _, proof = predicate_fixture(tmp_path, *PREDICATE_CASES[1])
    proof["categories"] = categories
    with pytest.raises(ValueError, match="proof-only or auxiliary"):
        study.checked_assignments(
            trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "or arbitrary_call()",
        "or key.clone().any()",
        "or key.copy_(value).all()",
        "or self.k_proj(normalized).all()",
        "or (alias := key).all()",
    ],
)
def test_validation_predicate_rejects_extra_calls_storage_operations_and_walrus(tmp_path, mutation):
    suffix, scope, predicate = PREDICATE_CASES[1]
    trace, _, proof = predicate_fixture(tmp_path, suffix, scope, f"({predicate}) {mutation}")
    with pytest.raises(ValueError, match=r"expression|aliases"):
        study.checked_assignments(
            trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
        )


@pytest.mark.parametrize(
    "preamble",
    [
        "torch = arbitrary",
        "all = arbitrary",
        "import arbitrary as torch",
        "from arbitrary import torch",
        "del torch",
    ],
)
def test_identical_predicate_rejects_provider_aliases_and_shadowing(tmp_path, preamble):
    trace, _, proof = predicate_fixture(tmp_path, *PREDICATE_CASES[1], preamble=preamble)
    with pytest.raises(ValueError, match="aliases/shadowing"):
        study.checked_assignments(
            trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
        )


def test_identical_predicate_rejects_provider_argument_but_accepts_canonical_torch_import(tmp_path):
    trace, _, proof = predicate_fixture(tmp_path, *PREDICATE_CASES[0], argument="torch")
    with pytest.raises(ValueError, match="aliases/shadowing"):
        study.checked_assignments(
            trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
        )
    trace, _, proof = predicate_fixture(
        tmp_path / "canonical", *PREDICATE_CASES[0], preamble="import torch"
    )
    assert study.checked_assignments(
        trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
    )[(20, 0)] == {"cache_auxiliary_workspace"}


@pytest.mark.parametrize("defect", ["scope", "file", "body", "line_hash", "changed_source"])
def test_validation_requires_actual_file_scope_test_line_and_immutable_hashes(tmp_path, defect):
    suffix, scope, predicate = PREDICATE_CASES[1]
    if defect == "scope":
        scope = "other_operator"
    if defect == "file":
        suffix = "/evaluation/other_operator.py"
    trace, file, proof = predicate_fixture(tmp_path, suffix, scope, predicate)
    if defect == "body":
        proof["line"] += 2
        proof["source_line_sha256"] = hashlib.sha256(
            file.read_text().splitlines(keepends=True)[proof["line"] - 1].encode()
        ).hexdigest()
        trace["stack_table"][1][0][2] = proof["line"]
        proof["stack_sha256"] = study.digest(trace["stack_table"][1])
    elif defect == "line_hash":
        proof["source_line_sha256"] = "0" * 64
    elif defect == "changed_source":
        file.write_text(file.read_text() + "# changed\n")
    with pytest.raises(ValueError):
        study.checked_assignments(
            trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
        )


@pytest.mark.parametrize(
    "predicate", ["source_validation_enabled()", "compact_workspace", "bit_flags is not None"]
)
def test_generic_branch_tests_are_never_allocation_validation_proofs(tmp_path, predicate):
    suffix, scope, _ = PREDICATE_CASES[1]
    trace, _, proof = predicate_fixture(tmp_path, suffix, scope, predicate)
    with pytest.raises(ValueError, match="concrete allocation"):
        study.checked_assignments(
            trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
        )


MODEL_CASES = [
    ("llama", "LlamaDecoderLayer.forward", "hidden_states = residual + hidden_states"),
    ("llava", "LlavaModel.forward", "inputs_embeds = self.get_input_embeddings()(input_ids)"),
    (
        "llava",
        "LlavaModel.forward",
        "image_features = torch.cat(image_features, dim=0).to("
        "inputs_embeds.device, inputs_embeds.dtype)",
    ),
    (
        "llava",
        "LlavaModel.forward",
        "inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)",
    ),
    (
        "llava",
        "LlavaModel.get_placeholder_mask",
        "special_image_mask = input_ids == self.config.image_token_id",
    ),
    ("llava", "LlavaModel.get_placeholder_mask", "n_image_tokens = special_image_mask.sum()"),
    (
        "llava",
        "LlavaModel.get_placeholder_mask",
        "if inputs_embeds[special_image_mask].numel() != image_features.numel():",
    ),
]


@pytest.mark.parametrize("module,scope,expression", MODEL_CASES)
def test_existing_seven_closed_model_templates_only_exclude_normal_model_activations(
    tmp_path, module, scope, expression
):
    trace, _, _, broad = model_expression_fixture(
        tmp_path, module=module, scope=scope, expression=expression
    )
    proof = broad["rules"][0]
    proof.pop("role")
    proof["categories"] = ["normal_model_activation"]
    assert study.checked_assignments(
        trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
    )[(20, 0)] == {"normal_model_activation"}
    for categories in (
        ["normal_query"],
        ["cache_payload"],
        ["cache_auxiliary_workspace"],
        ["normal_model_activation", "normal_scores"],
    ):
        proof["categories"] = categories
        with pytest.raises(ValueError):
            study.checked_assignments(
                trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
            )


@pytest.mark.parametrize(
    "expression",
    [
        "hidden_states = self.k_proj(hidden_states)",
        "hidden_states = self.v_proj(hidden_states)",
        "hidden_states = residual - hidden_states",
        "hidden_states = normalized_source.clone()",
        "hidden_states = residual + (alias := hidden_states)",
    ],
)
def test_model_template_mutations_and_capture_assignments_do_not_get_exclusion(
    tmp_path, expression
):
    trace, _, _, broad = model_expression_fixture(
        tmp_path, module="llama", scope="LlamaDecoderLayer.forward", expression=expression
    )
    proof = broad["rules"][0]
    proof.pop("role")
    proof["categories"] = ["normal_model_activation"]
    with pytest.raises(ValueError, match=r"forward|aliases"):
        study.checked_assignments(
            trace, {"observations": []}, empty_ledger([proof]), source_bindings={}
        )


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


def typed_streaming_fixture(tmp_path, monkeypatch):
    import evicastkv_review_source_prefix as prefix
    import evicastkv_streaming_lossless_ownership as streaming
    import torch

    trace, _, _, _, file = context(tmp_path)
    parent = tmp_path / "typed"
    out = parent / "workers/0"
    out.mkdir(parents=True)
    revision = "b" * 40
    plan = dict(source_sha=revision, policy=dict(method=study.STREAMING_KIND), bindings={})
    plan["bindings"][str(file)] = study.sha(file)
    (parent / "protocol.json").write_text(json.dumps(plan))
    (parent / streaming.GATE).write_text(
        json.dumps(
            dict(
                source_sha=revision,
                complete_numerical_and_trace_controls=True,
                failures=[],
                summaries=[dict(index=0), dict(index=1)],
            )
        )
    )
    catalog = dict(owners=[], owner_catalog_sha256=study.digest([]))
    capture = dict(
        complete_request=True,
        allocator_history_complete=True,
        blocks_exact_certification=["unclassified_allocation_generations"],
        clock_calibration=trace["clock_calibration"],
    )
    bundle = dict(
        trace,
        schema=study.STREAMING_KIND,
        source_sha=revision,
        baseline_model_catalog=catalog,
        capture_metadata=capture,
        declared_owner_observations=[],
        final_allocator_snapshot={"fixture": True},
    )
    for name, value in {
        "baseline_model_catalog.json": catalog,
        "baseline_model_allocator_matches.json": dict(
            catalog_sha256=catalog["owner_catalog_sha256"], other_baseline_owners_exempted=False
        ),
        "GATE_cuda_operator.json": dict(passed=True),
        "trace_receipt.json": dict(source_sha=revision),
    }.items():
        (out / name).write_text(json.dumps(value))
    (out / "allocator_trace.json.gz").write_bytes(b"typed synthetic loader control")
    saved = dict(reference=torch.zeros(128, 8), candidate=torch.zeros(128, 8))
    torch.save(saved, out / "logits.pt")
    raw = dict(
        method=study.STREAMING_KIND,
        source_sha=revision,
        index=0,
        record={"fixture_request": True},
        counters={"fixture_counts": True},
        diagnostics=prefix.comparisons(saved["reference"], saved["candidate"]),
        first_prefill_logit_difference=0.0,
    )
    (out / "raw.json").write_text(json.dumps(raw))
    manifest = dict(
        source_sha=revision,
        protocol_sha256=study.sha(parent / "protocol.json"),
        method=study.STREAMING_KIND,
        no_labels=True,
        no_native_reference_forward_inside_window=True,
        job_id="1234",
        GPU=dict(
            devices=[dict(visible_to_process=True, name="A100")],
            background_processes=[],
            query_error=None,
            visible_process_concurrency=1,
        ),
        baseline_catalog_sha256=study.sha(out / "baseline_model_catalog.json"),
        baseline_model_metadata_digest=catalog["owner_catalog_sha256"],
    )
    (out / "manifest.json").write_text(json.dumps(manifest))

    def seal_outputs():
        value = dict(
            method=study.STREAMING_KIND,
            source_sha=revision,
            index=0,
            status="completed_numerical_diagnostic",
            failures=[],
            all_32_128_guards_pass=True,
            complete_trace_with_final_snapshot_archived=True,
            outputs={
                path.name: study.sha(path) for path in out.iterdir() if path.name != "result.json"
            },
        )
        (out / "result.json").write_text(json.dumps(value))

    seal_outputs()
    monkeypatch.setattr(streaming, "validate", lambda root: plan)
    monkeypatch.setattr(streaming, "load_bundle", lambda path, receipt: bundle)
    monkeypatch.setattr(
        streaming,
        "verify_request",
        lambda record, counts: record["fixture_request"] and counts["fixture_counts"],
    )
    monkeypatch.setattr(prefix, "analysis_cpu_controls", lambda: {"fixture": True})
    return parent, out, plan, bundle, seal_outputs


def test_typed_streaming_inputs_preserve_new_schema_and_do_not_auto_classify(tmp_path, monkeypatch):
    parent, _, plan, _, _ = typed_streaming_fixture(tmp_path, monkeypatch)
    values, paths, hashes = study.streaming_inputs(parent, 0)
    assert values["input_schema"] == values["producer_method"] == study.STREAMING_KIND
    assert values["producer_sha"] == plan["source_sha"]
    assert values["ownership_capture_complete"] is True
    assert "producer_gate" in paths and "logits.pt" in paths
    assert hashes == {name: study.sha(path) for name, path in paths.items()}
    assert (
        study.checked_assignments(
            values["trace"], values["declared"], empty_ledger([]), source_bindings={}
        )
        == {}
    )


@pytest.mark.parametrize(
    "defect",
    [
        "raw_hash",
        "producer_status",
        "producer_source",
        "gate",
        "isolation",
        "schema",
        "window",
        "history",
        "structural",
        "numerical",
        "prefix_length",
        "catalog",
    ],
)
def test_typed_streaming_inputs_reject_incomplete_tampered_or_negative_evidence(
    tmp_path, monkeypatch, defect
):
    import torch

    parent, out, _, bundle, seal = typed_streaming_fixture(tmp_path, monkeypatch)
    if defect in {"producer_status", "producer_source"}:
        result = json.loads((out / "result.json").read_text())
        result["status" if defect == "producer_status" else "source_sha"] = "failed"
        (out / "result.json").write_text(json.dumps(result))
    elif defect == "raw_hash":
        (out / "raw.json").write_text("{}")
    elif defect == "gate":
        import evicastkv_streaming_lossless_ownership as streaming

        gate = json.loads((parent / streaming.GATE).read_text())
        gate["complete_numerical_and_trace_controls"] = False
        (parent / streaming.GATE).write_text(json.dumps(gate))
    elif defect == "isolation":
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["GPU"]["background_processes"] = [dict(pid=999)]
        (out / "manifest.json").write_text(json.dumps(manifest))
        seal()
    elif defect == "schema":
        bundle["schema"] = "legacy_three_bit_ARM"
    elif defect in {"window", "history", "structural"}:
        field = "complete_request" if defect == "window" else "allocator_history_complete"
        if defect == "structural":
            bundle["capture_metadata"]["blocks_exact_certification"].append(
                "free_without_observed_allocation"
            )
        else:
            bundle["capture_metadata"][field] = False
    elif defect in {"numerical", "prefix_length"}:
        saved = torch.load(out / "logits.pt", weights_only=True)
        if defect == "numerical":
            saved["candidate"][50, 0] = 100
        else:
            saved["candidate"] = saved["candidate"][:32]
        torch.save(saved, out / "logits.pt")
        seal()
    elif defect == "catalog":
        bundle["baseline_model_catalog"] = dict(owners=["not actual baseline"])
    with pytest.raises((ValueError, KeyError)):
        study.streaming_inputs(parent, 0)


@pytest.mark.parametrize("worker_index", [None, 0])
def test_typed_streaming_run_remains_unknown_without_explicit_source_rules(
    tmp_path, monkeypatch, worker_index
):
    parent, _, plan, _, _ = typed_streaming_fixture(tmp_path, monkeypatch)
    _, _, hashes = study.streaming_inputs(parent, 0)
    ledger = empty_ledger([])
    ledger.update(producer_sha=plan["source_sha"], artifact_hashes=hashes)
    rules = tmp_path / "typed_rules.json"
    rules.write_text(json.dumps(ledger))
    monkeypatch.setattr(study, "SCRATCH", tmp_path)
    monkeypatch.setattr(study, "clean_source", lambda *args: plan["source_sha"])
    monkeypatch.setattr(study, "cpu_gate", lambda *args: {"synthetic": True})
    gate = study.run(
        tmp_path / "derived",
        parent=parent,
        reviewed_rules=rules,
        cpu_gate_root=tmp_path,
        worker_index=worker_index,
    )
    assert gate["unknown_generations"] > 0
    assert not any(metric["exact_peak_certified"] for metric in gate["metrics"].values())
    manifest = json.loads((tmp_path / "derived/manifest.json").read_text())
    assert manifest["input_schema"] == manifest["producer_method"] == study.STREAMING_KIND
    assert manifest["worker_index"] == 0
    assert not gate["numerical_or_quality_guard_promoted"]


def catalog_witness_fixture(tmp_path):
    trace, _, declared, ledger, _ = context(tmp_path)
    file = tmp_path / "scripts/evicastkv_streaming_lossless_ownership.py"
    file.parent.mkdir()
    file.write_text(
        "def mark_baseline_model(session, owner, catalog):\n"
        '    session.timeline.classify(owner["address"], "model", '
        '"pinned immutable resident model parameters/buffers actual named catalog " '
        '+ catalog["owner_catalog_sha256"])\n'
    )
    owner = dict(
        address=1, bytes=100, device="cuda:0", names=[dict(name="weight", family="parameter")]
    )
    catalog = dict(owners=[owner], owner_catalog_sha256=study.digest([owner]))
    trace.update(input_schema=study.STREAMING_KIND, baseline_model_catalog=catalog)
    row = trace["allocator_records"][0]
    row["evidence"] = [
        study.MODEL_WITNESS + " actual named catalog " + catalog["owner_catalog_sha256"]
    ]
    proof = dict(
        filename=str(file),
        function="mark_baseline_model",
        line=2,
        source_sha256=study.sha(file),
        source_line_sha256=hashlib.sha256(
            file.read_text().splitlines(keepends=True)[1].encode()
        ).hexdigest(),
        qualified_function="mark_baseline_model",
    )
    witness = dict(
        address=1,
        generation=0,
        catalog_sha256=catalog["owner_catalog_sha256"],
        owner_sha256=study.digest(owner),
        evidence_sha256=study.digest(row["evidence"]),
        source=proof,
        reason="Exact synthetic named parameter backing owner matches actual seeded allocation.",
        reviewed_by="synthetic source review",
    )
    ledger["catalog_model_identity_witnesses"] = [witness]
    return trace, declared, ledger, file


def test_explicit_named_model_catalog_witness_is_separate_from_legacy_model_roles(tmp_path):
    trace, declared, ledger, file = catalog_witness_fixture(tmp_path)
    assert study.checked_assignments(
        trace, declared, ledger, source_bindings={str(file): study.sha(file)}
    ) == {(1, 0): {"model_weights"}}
    ledger["catalog_model_identity_witnesses"] = []
    assert study.checked_assignments(trace, declared, ledger, source_bindings={}) == {}


@pytest.mark.parametrize(
    "defect", ["schema", "bytes", "owner", "baseline", "evidence", "source", "operator"]
)
def test_named_model_catalog_cannot_exempt_unrelated_or_changed_baseline_owner(tmp_path, defect):
    trace, declared, ledger, file = catalog_witness_fixture(tmp_path)
    if defect == "schema":
        trace["input_schema"] = "legacy"
    elif defect == "bytes":
        trace["allocator_records"][0]["bytes"] += 1
    elif defect == "owner":
        ledger["catalog_model_identity_witnesses"][0]["owner_sha256"] = "0" * 64
    elif defect == "baseline":
        trace["allocator_records"][0]["baseline"] = False
    elif defect == "evidence":
        trace["allocator_records"][0]["evidence"] = ["model"]
    elif defect == "source":
        file.write_text(file.read_text() + "# changed\n")
    else:
        file.write_text(file.read_text().replace('owner["address"]', 'arbitrary["address"]'))
        witness = ledger["catalog_model_identity_witnesses"][0]
        witness["source"]["source_sha256"] = study.sha(file)
        witness["source"]["source_line_sha256"] = hashlib.sha256(
            file.read_text().splitlines(keepends=True)[1].encode()
        ).hexdigest()
    with pytest.raises(ValueError):
        study.checked_assignments(
            trace, declared, ledger, source_bindings={str(file): study.sha(file)}
        )
