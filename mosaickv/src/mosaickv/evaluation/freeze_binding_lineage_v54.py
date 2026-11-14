"""Pure V53-to-runtime lineage using the unchanged original V49 transaction.

The I/O coordinator validates the original scheduler/source records. This layer
requires their pinned, digest-bound facts and re-derives the entire saved V53
parent before adding the exact output's actual writer. A seal or protocol SHA
never supplies that writer identity. No historical artifact is opened here.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation import negative_seal_bindings_v53 as seals
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.freeze_writer_bindings_v54 import validate_runtime_facts
from mosaickv.evaluation.parquet_binding_recovery_v49 import CLAIM_FIELDS, classify_binding
from mosaickv.evaluation.source_archive_bindings_v51 import _bounded, _schema


def promote_freeze_binding(
    previous_v53: dict[str, Any],
    previous_v52: dict[str, Any],
    original_row: dict[str, Any],
    transaction: dict[str, Any],
    original_claims: list[dict[str, Any]],
    sealed_claims: list[dict[str, Any]],
    runtime_claim: dict[str, Any],
    *,
    runtime_facts: dict[str, Any],
    previous_reviewed_writers: Mapping[tuple[str, str], str],
    reviewed_writers: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """Promote only a re-derived original negative seal's exact runtime writer.

    ``previous_reviewed_writers`` is the unchanged V53 classifier registry;
    ``reviewed_writers`` additionally contains the reviewed actual freeze
    writers. Their distinct roles prevent a new registry from rewriting the
    parent proof. Runtime facts must pass the pinned pure facts validator, not
    merely carry a caller-provided success flag.
    """
    _bounded(previous_v53)
    if (
        not isinstance(previous_v53, dict)
        or previous_v53.get("schema") != "historical_negative_seal_binding_v53"
        or previous_v53.get("status") != "artifact_binding_only"
    ):
        raise ValueError("exact saved artifact-only V53 negative-seal parent required")
    parent = seals.augment_seal_binding(
        previous_v52,
        original_row,
        transaction,
        original_claims,
        sealed_claims,
        reviewed_writers=previous_reviewed_writers,
    )
    if digest(parent) != digest(previous_v53):
        raise ValueError("saved V53 parent differs from its complete original lineage")

    facts = validate_runtime_facts(runtime_facts)
    _schema(runtime_claim, CLAIM_FIELDS, "original freeze runtime claim")
    target = parent["path"]
    outputs = [entry for entry in facts["outputs"] if entry["path"] == target]
    if len(outputs) != 1:
        raise ValueError("exact original target missing or ambiguous in runtime output map")
    output = outputs[0]
    if output["artifact_sha256"] != parent["artifact_sha256"]:
        raise ValueError("runtime output hash differs from unchanged original artifact")
    expected = dict(
        metadata_path=runtime_claim["metadata_path"],
        metadata_sha256=facts["seal_sha256"],
        artifact_sha256=output["artifact_sha256"],
        role="runtime_writer",
        recorded_source_sha=facts["source_sha"],
        source_blob_sha256=output["writer_sha256"],
        writer_path=output["writer_path"],
        source_evidence_verified=True,
    )
    if runtime_claim != expected or runtime_claim["source_evidence_verified"] is not True:
        raise ValueError("runtime claim differs from actual output writer/source facts")
    metadata = seals._canonical(runtime_claim["metadata_path"], "original sealed metadata")
    # The complete re-derived parent already binds the historical metadata
    # root. The facts additionally identify its exact repository-relative seal.
    seal_parts = PurePosixPath(facts["seal_path"]).parts
    if PurePosixPath(metadata).parts[-len(seal_parts) :] != seal_parts or not any(
        claim["metadata_path"] == runtime_claim["metadata_path"]
        and claim["metadata_sha256"] == runtime_claim["metadata_sha256"]
        and claim["artifact_sha256"] == runtime_claim["artifact_sha256"]
        for claim in sealed_claims
    ):
        raise ValueError("runtime claim must reuse one exact original negative-seal claim")
    if facts["source_blob_sha256s"].get(output["writer_path"]) != output["writer_sha256"]:
        raise ValueError("actual output writer is missing from the critical source closure")
    new = classify_binding(
        original_row,
        transaction,
        [*original_claims, *sealed_claims, runtime_claim],
        reviewed_writers=reviewed_writers,
    )
    if (
        new["status"] != "runtime_producer_and_artifact_binding_verified"
        or new["original_producer_commit"] != facts["source_sha"]
        or new["original_producer_writer_path"] != output["writer_path"]
        or new["original_producer_source_blob_sha256"] != output["writer_sha256"]
        or new["artifact_sha256"] != parent["artifact_sha256"]
        or new["signature"] != parent["signature"]
    ):
        raise ValueError("actual freeze writer failed the unchanged V49 binding classifier")
    result = copy.deepcopy(new)
    result.update(
        schema="historical_freeze_writer_binding_v54",
        parent_binding_sha256=digest(previous_v53),
        parent_v53_binding_sha256=digest(previous_v53),
        parent_v52_binding_sha256=digest(previous_v52),
        original_negative_seal_claim_sha256s=list(parent["original_negative_seal_claim_sha256s"]),
        original_freeze_runtime_claim_sha256=digest(runtime_claim),
        runtime_facts_sha256=digest(facts),
        source_runtime_evidence_sha256=facts["runtime_evidence_sha256"],
        original_writer_closure_sha256=facts["critical_source_closure_sha256"],
        binding_extension_schema="original_freeze_runtime_writer_v54",
        reused_hash_transaction=True,
        canonical_eligible=False,
        clean_runtime_execution_claimed=False,
        negative_result_preserved=True,
        quality_gate_pass=False,
        scientific_dependency_validation=False,
        occurrence_resolved=False,
        newly_resolved_occurrences=0,
        GPU_jobs=0,
        **dict.fromkeys(contract.FALSE_FIELDS, False),
    )
    _bounded(result)
    return result
