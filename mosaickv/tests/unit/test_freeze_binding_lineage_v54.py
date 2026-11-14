"""Pure synthetic transactions with literal, pinned public source projections."""

from copy import deepcopy
from pathlib import Path

import pytest
from test_negative_seal_bindings_v53 import arguments as seal_arguments

from mosaickv.evaluation import freeze_binding_lineage_v54 as lineage
from mosaickv.evaluation import freeze_writer_bindings_v54 as runtime
from mosaickv.evaluation import negative_seal_bindings_v53 as seals
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.parquet_binding_recovery_v49 import classify_binding


def facts(version=76):
    """Literal reviewed metadata only: no scheduler, Git, YAML or body reads."""
    suffix = {
        76: "choice-c-v76-disjoint-ocr-calibration-v1",
        88: "choice-c-v88-initial-query-coverage-calibration-v1",
    }[version]
    root = "/sfs/weka/scratch/djy8hg/runs/mosaickv/" + suffix
    protocol = (
        "mosaickv/configs/experiments/"
        + {
            76: "choice_c_v76_disjoint_ocr_calibration.yaml",
            88: "choice_c_v88_initial_query_coverage_calibration.yaml",
        }[version]
    )
    wrapper = f"mosaickv/slurm/choice_c_v{version}_freeze.sbatch"
    source, tree, protocol_hash, wrapper_hash, base_hash = {
        76: (
            "e8d8b542a594c96b3fb819956a85e692f8a54f2f",
            "7848fe4bcda6c46063d6194866c29a665929bff5",
            "b8b7a09783fb39ed58ed45c5207f2ecf4661c73dc59b8aa86969d6f8fe72b983",
            "0fd3de0d0055f335f1a5104d29987b98ad8508cd7d44b72be4043d991fc30552",
            "c5fe8128434025f9f964dd9fb79c7453c344a5c023600f46ca5b93d703979430",
        ),
        88: (
            "9e482f71f86f52f92f2946d24a73e3fa7de63cc9",
            "e69af26d586b9582f0b921fdbd003fbc5d9e75d4",
            "f9bee74c82365ecb999ea8d34ca10b50f8402618157f92f5f153a1ae766d6efb",
            "3e7cf23d90757746d932f8b9a4f5fcaf29890abefd59027abf817a3f87eee605",
            "be7693b6303b0c20bb3dcd3e19703f5df7c3e99d949a26d340371242aa27783c",
        ),
    }[version]
    sources = {
        protocol: protocol_hash,
        wrapper: wrapper_hash,
        runtime.BASE_WRITER: base_hash,
        "mosaickv/scripts/assert_clean_worktree.sh": (
            "5522a24655bb8d55e4866759902462c0363bd8258429cc7e03c514d0279dd70e"
        ),
        "mosaickv/scripts/cache_env.sh": (
            "a471d60c076e67a10ee1ba358810a5e281d8cca89928c7b9aeda6b33dbc78c17"
        ),
        "mosaickv/scripts/choice_c_v74_protocol.py": {
            76: "59898ce61822220e52b3639441d7daf692b153de26ea7bc5eb6d6856319fd3c2",
            88: "6463265eeada9feebccc1c4b592a4bb21113eef6e18894169889bc9aee4028e3",
        }[version],
        "mosaickv/scripts/run_choice_c_gate.py": (
            "7c130e0bb60b28f6fac5cf1c28b8e73ad021280aec069d6ae6606b2f21117bea"
        ),
        "mosaickv/scripts/run_choice_c_v72_heldout_quality_memory_worker.py": (
            "09bd0b0a273b10d4e0727ec8656044b43c4da27917c0310fa279f7da31268a16"
        ),
    }
    outputs = [
        dict(
            path=root + "/analysis/label_blind_candidate_ranking.parquet",
            artifact_sha256={
                76: "199ec3b66f69cfa037b8bba3361b392fb9b225cb072b4beb8989db48bfa42a83",
                88: "d03406158e45c474ba2f17f387cdce1ee4441ee465d8598f7d4be7db0496d619",
            }[version],
            writer_path=runtime.BASE_WRITER,
            writer_sha256=base_hash,
        )
    ]
    if version == 88:
        robust_hash = "51f2f04ae6376718bf1d8193605669a0f34ea35dd5ea71abd35ce2e144fb1f77"
        sources[runtime.ROBUST_WRITER] = robust_hash
        outputs.append(
            dict(
                path=root + "/analysis/robust_fidelity_candidate_ranking.parquet",
                artifact_sha256="a9828c4f7bb9a8e307c2702b4d39e719ec6ff7a457dd03edd1ab15991255d9ab",
                writer_path=runtime.ROBUST_WRITER,
                writer_sha256=robust_hash,
            )
        )
    result = dict(
        schema="historical_freeze_runtime_facts_v54",
        version=version,
        job_id={76: "18572260", 88: "18650057"}[version],
        source_sha=source,
        repository_tree_oid=tree,
        run_root=root,
        frozen_output=root + "/analysis/frozen_candidate.yaml",
        protocol_path=protocol,
        protocol_sha256=protocol_hash,
        seal_path=f"mosaickv/configs/ledgers/choice_c_v{version}_negative_calibration_seal.yaml",
        seal_sha256={
            76: "69e6fa0d75e597764ff19998a54cab35017ee30b368d968d7ff8e4dea34be70e",
            88: "c08b3d97576f57aa989ab6ff4e824cf7a2579ece6f481f23b7ae6274fac5e300",
        }[version],
        expected_state="COMPLETED" if version == 88 else "FAILED",
        expected_exit_code="0:0" if version == 88 else "2:0",
        accounting_sha256={
            76: "17081d847ed09502412ea7993cc3c83002ccec5774e796cf6ae1f02c524098fa",
            88: "d296cec075fe568052dc55d59507f4cafc2611ea8bed77c991848389ed16a42c",
        }[version],
        startup_sha256={
            76: "af6e7abba98d1eb93aed5a34bd7c5728027871b109ac3f6583b426af308e074c",
            88: "d80c44803ecc24c08bc94db25388a16ff0a8c441ef2a5be5548806b55ed7ae72",
        }[version],
        batch_record_sha256={
            76: "d034ce234a3a21cb82901227d100ebfba63dab785221756ecc767a01b7df0482",
            88: "d0ca0363eee523378d3c28e2bdf0c67726230a60aac665f084f3166a2a5784d9",
        }[version],
        wrapper_path=wrapper,
        wrapper_sha256=wrapper_hash,
        entrypoint_path=runtime.ROBUST_WRITER if version == 88 else runtime.BASE_WRITER,
        source_blob_sha256s=sources,
        outputs=outputs,
        critical_source_closure_sha256=digest(sources),
        negative_result_preserved=True,
        quality_gate_pass=False,
        scientific_dependency_validation=False,
        canonical_eligible=False,
    )
    result["runtime_evidence_sha256"] = digest(result)
    return result


def arguments(version=76, output_index=0, *, old_claim=False):
    values = seal_arguments()
    runtime_facts = facts(version)
    output = runtime_facts["outputs"][output_index]
    row, transaction = values["original_row"], values["transaction"]
    row["path"] = transaction["path"] = output["path"]
    transaction["before_sha256"] = transaction["after_sha256"] = output["artifact_sha256"]
    registration = {
        name: row[name]
        for name in seals.contract.OCCURRENCE_FIELDS | {"path", "signature", "operation"}
    }
    row["original_registration_sha256"] = digest(registration)
    claim = seals.artifact_claim(
        dict(
            run_root=runtime_facts["run_root"],
            classification=seals.CLASSIFICATION,
            immutable_result=True,
            frozen_candidate_created=False,
            artifacts={"ranking": dict(path=output["path"], sha256=output["artifact_sha256"])},
        ),
        runtime.PROJECT + "/" + runtime_facts["seal_path"],
        runtime_facts["seal_sha256"],
        output["path"],
    )
    original_claims = [dict(claim, metadata_sha256="e" * 64)] if old_claim else []
    previous_writers = values["reviewed_writers"]
    previous_v52 = classify_binding(
        row, transaction, original_claims, reviewed_writers=previous_writers
    )
    previous_v53 = seals.augment_seal_binding(
        previous_v52, row, transaction, original_claims, [claim], reviewed_writers=previous_writers
    )
    return dict(
        previous_v53=previous_v53,
        previous_v52=previous_v52,
        original_row=row,
        transaction=transaction,
        original_claims=original_claims,
        sealed_claims=[claim],
        runtime_claim=dict(
            claim,
            role="runtime_writer",
            recorded_source_sha=runtime_facts["source_sha"],
            source_blob_sha256=output["writer_sha256"],
            writer_path=output["writer_path"],
            source_evidence_verified=True,
        ),
        runtime_facts=runtime_facts,
        previous_reviewed_writers=previous_writers,
        reviewed_writers=dict(previous_writers)
        | {
            (runtime_facts["source_sha"], item["writer_path"]): item["writer_sha256"]
            for item in runtime_facts["outputs"]
        },
    )


@pytest.mark.parametrize("version,index", [(76, 0), (88, 0), (88, 1)])
@pytest.mark.parametrize("old_claim", [False, True])
def test_rederives_entire_parent_and_preserves_opaque_transaction(version, index, old_claim):
    values = arguments(version, index, old_claim=old_claim)
    saved = deepcopy(values)
    result = lineage.promote_freeze_binding(**values)
    assert values == saved
    assert result["schema"] == "historical_freeze_writer_binding_v54"
    assert result["status"] == "runtime_producer_and_artifact_binding_verified"
    assert result["parent_binding_sha256"] == digest(values["previous_v53"])
    assert result["parent_v53_binding_sha256"] == digest(values["previous_v53"])
    assert result["parent_v52_binding_sha256"] == digest(values["previous_v52"])
    assert result["runtime_facts_sha256"] == digest(values["runtime_facts"])
    assert (
        result["source_runtime_evidence_sha256"]
        == values["runtime_facts"]["runtime_evidence_sha256"]
    )
    assert result["original_freeze_runtime_claim_sha256"] == digest(values["runtime_claim"])
    assert (
        result["original_negative_seal_claim_sha256s"]
        == values["previous_v53"]["original_negative_seal_claim_sha256s"]
    )
    assert (
        result["original_writer_closure_sha256"]
        == values["runtime_facts"]["critical_source_closure_sha256"]
    )
    for name in ("path", "signature", "artifact_sha256", "original_registration_sha256"):
        assert result[name] == values["previous_v53"][name]
    assert result["original_producer_commit"] == values["runtime_facts"]["source_sha"]
    assert result["original_producer_writer_path"] == values["runtime_claim"]["writer_path"]
    assert (
        result["original_producer_source_blob_sha256"]
        == values["runtime_claim"]["source_blob_sha256"]
    )
    assert result["newly_resolved_occurrences"] == result["GPU_jobs"] == 0
    assert result["reused_hash_transaction"] is True
    assert result["negative_result_preserved"] is True
    for name in (
        *seals.contract.FALSE_FIELDS,
        "canonical_eligible",
        "clean_runtime_execution_claimed",
        "quality_gate_pass",
        "scientific_dependency_validation",
        "occurrence_resolved",
    ):
        assert result[name] is False


def test_distinct_parent_and_current_registries_reach_existing_classifiers(monkeypatch):
    values = arguments(old_claim=True)
    calls = []
    original = seals.augment_seal_binding

    def observe_parent(*args, **kwargs):
        calls.append(("parent", deepcopy(args), deepcopy(kwargs)))
        return original(*args, **kwargs)

    def observe_classifier(*args, **kwargs):
        calls.append(("new", deepcopy(args), deepcopy(kwargs)))
        return classify_binding(*args, **kwargs)

    monkeypatch.setattr(seals, "augment_seal_binding", observe_parent)
    monkeypatch.setattr(lineage, "classify_binding", observe_classifier)
    lineage.promote_freeze_binding(**values)
    assert len(calls) == 2
    assert calls[0][2] == {"reviewed_writers": values["previous_reviewed_writers"]}
    assert calls[0][1][:3] == (
        values["previous_v52"],
        values["original_row"],
        values["transaction"],
    )
    assert calls[1][1] == (
        values["original_row"],
        values["transaction"],
        values["original_claims"] + values["sealed_claims"] + [values["runtime_claim"]],
    )
    assert calls[1][2] == {"reviewed_writers": values["reviewed_writers"]}


@pytest.mark.parametrize(
    "field",
    [
        "parent_binding_sha256",
        "artifact_sha256",
        "signature",
        "original_registration_sha256",
        "original_negative_seal_claim_sha256s",
        "original_metadata_claim_sha256s",
        "status",
        "original_producer_commit",
        "canonical_eligible",
        "occurrence_resolved",
        *seals.contract.FALSE_FIELDS,
    ],
)
def test_any_saved_v53_parent_edit_fails_closed(field):
    values = arguments()
    values["previous_v53"][field] = "edited"
    with pytest.raises(ValueError):
        lineage.promote_freeze_binding(**values)


@pytest.mark.parametrize(
    "schema",
    [
        "historical_parquet_binding_recovery_v49",
        "historical_original_manifest_binding_v52",
        "historical_freeze_writer_binding_v54",
        None,
    ],
)
def test_arbitrary_old_or_already_promoted_schemas_cannot_be_upgraded(schema):
    values = arguments()
    values["previous_v53"]["schema"] = schema
    with pytest.raises(ValueError):
        lineage.promote_freeze_binding(**values)


@pytest.mark.parametrize(
    "name",
    [
        "previous_v52",
        "original_row",
        "transaction",
        "original_claims",
        "sealed_claims",
    ],
)
def test_every_saved_parent_input_is_rederived(name):
    values = arguments(old_claim=True)
    if name in {"original_claims", "sealed_claims"}:
        values[name][0]["metadata_sha256"] = "f" * 64
    elif name == "transaction":
        values[name]["signature"][1] += 1
    elif name == "original_row":
        values[name]["original_registration_sha256"] = "f" * 64
    else:
        values[name]["original_metadata_claim_sha256s"] = []
    with pytest.raises(ValueError):
        lineage.promote_freeze_binding(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("metadata_path", runtime.PROJECT + "/other.yaml"),
        ("metadata_sha256", "e" * 64),
        ("artifact_sha256", "e" * 64),
        ("recorded_source_sha", "e" * 40),
        ("source_blob_sha256", "e" * 64),
        ("writer_path", runtime.ROBUST_WRITER),
        ("role", "input_source"),
        ("source_evidence_verified", False),
        ("source_evidence_verified", 1),
        ("extra", True),
    ],
)
def test_runtime_claim_must_match_exact_old_seal_and_actual_new_writer(field, value):
    values = arguments()
    values["runtime_claim"][field] = value
    with pytest.raises(ValueError):
        lineage.promote_freeze_binding(**values)


@pytest.mark.parametrize("transfer", ["protocol", "seal_commit", "entrypoint"])
def test_protocol_seal_commit_or_robust_entrypoint_cannot_replace_actual_writer(transfer):
    values = arguments(88)
    claim, projection = values["runtime_claim"], values["runtime_facts"]
    if transfer == "protocol":
        claim["source_blob_sha256"] = projection["protocol_sha256"]
        claim["writer_path"] = projection["protocol_path"]
    elif transfer == "seal_commit":
        claim["recorded_source_sha"] = "75736bbc115d2ad5d82ae7c7a3a26df41c89d8fe"
    else:
        claim["writer_path"] = projection["entrypoint_path"]
        claim["source_blob_sha256"] = projection["source_blob_sha256s"][
            projection["entrypoint_path"]
        ]
    with pytest.raises(ValueError):
        lineage.promote_freeze_binding(**values)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "version",
        "job_id",
        "source_sha",
        "repository_tree_oid",
        "outputs",
        "source_blob_sha256s",
        "critical_source_closure_sha256",
        "runtime_evidence_sha256",
        "accounting_sha256",
        "startup_sha256",
        "batch_record_sha256",
        "negative_result_preserved",
        "quality_gate_pass",
        "scientific_dependency_validation",
        "canonical_eligible",
    ],
)
def test_missing_runtime_fact_or_an_asserted_success_flag_is_not_proof(field):
    values = arguments()
    del values["runtime_facts"][field]
    with pytest.raises(ValueError):
        lineage.promote_freeze_binding(**values)


@pytest.mark.parametrize("change", ["hash", "path", "empty", "wrong_writer", "duplicate"])
def test_runtime_output_edits_fail_even_with_a_recomputed_caller_digest(change):
    values = arguments()
    projection = values["runtime_facts"]
    if change == "hash":
        projection["outputs"][0]["artifact_sha256"] = "e" * 64
    elif change == "path":
        projection["outputs"][0]["path"] += ".other.parquet"
    elif change == "empty":
        projection["outputs"] = []
    elif change == "wrong_writer":
        projection["outputs"][0]["writer_path"] = runtime.ROBUST_WRITER
    else:
        projection["outputs"].append(deepcopy(projection["outputs"][0]))
    projection["runtime_evidence_sha256"] = digest(
        {key: value for key, value in projection.items() if key != "runtime_evidence_sha256"}
    )
    with pytest.raises(ValueError):
        lineage.promote_freeze_binding(**values)


@pytest.mark.parametrize("change", ["missing", "wrong_blob", "wrong_commit"])
def test_actual_runtime_writer_requires_the_reviewed_exact_registry_entry(change):
    values = arguments()
    claim = values["runtime_claim"]
    key = claim["recorded_source_sha"], claim["writer_path"]
    if change == "missing":
        del values["reviewed_writers"][key]
    elif change == "wrong_blob":
        values["reviewed_writers"][key] = "e" * 64
    else:
        values["reviewed_writers"][("e" * 40, key[1])] = values["reviewed_writers"].pop(key)
    with pytest.raises(ValueError):
        lineage.promote_freeze_binding(**values)


def test_no_filesystem_git_network_or_scheduler_access(monkeypatch):
    values = arguments(88, 1, old_claim=True)

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure V54 lineage validator performed I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("os.open", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    for method in ("open", "stat", "lstat", "resolve", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, method, forbidden)
    assert lineage.promote_freeze_binding(**values)["reused_hash_transaction"] is True
