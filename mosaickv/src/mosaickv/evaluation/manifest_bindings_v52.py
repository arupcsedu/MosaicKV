"""Pure extension of three registered original-manifest byte bindings.

The I/O caller verifies metadata, both historical source blobs, and saved-proof
hashes. Every artifact transaction is reclassified with the existing V49
contract; this layer introduces no alternative transaction or semantic waiver.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.parquet_binding_recovery_v49 import classify_binding
from mosaickv.evaluation.source_archive_bindings_v51 import _bounded

WRITER = "mosaickv/scripts/evicastkv_consumed_instrumentation.py"
GUARD = "mosaickv/src/mosaickv/evaluation/fox_campaign.py"
GUARD_SHA = "4fa92bd9908d48ab6b040e09475d87a982521a1e934532bc5bca4e615ce0d421"
BASE = "/sfs/weka/scratch/djy8hg/runs/evicastkv"
_ROUTING = BASE + "/consumed-routing-tier-diagnostic-20261029-v1/analysis"
_COMPACT = BASE + "/compact-exact-views-consumed-20261029-v1/analysis"
REGISTERED = {
    _ROUTING + "/tier_operations.parquet": dict(
        metadata_path=_ROUTING + "/manifest.json",
        metadata_sha256="a302f875904e64982a77d0f2fe17de3d6f59d90e1532b3ee230d2f473ed0d8a9",
        recorded_source_sha="da58a3370a92fc6378ecc77c26bf6ae22887547c",
        source_blob_sha256="39a1b4caee3e66751cc29e550de2fb2090e73f7f03379f72559aefc7c7a81c71",
        artifact_sha256="88936077b0862ebb8133420460ac273d135130d663383190d88ae0fcab4b98aa",
    ),
    _COMPACT + "/tier_operations.parquet": dict(
        metadata_path=_COMPACT + "/manifest.json",
        metadata_sha256="62851f2c14ed0878b3aadcd6dfd216190428a39aa89e238a029c3c8bcf10ee20",
        recorded_source_sha="7472d7ccf9bb670b6921398cada04f6cfd2a5f85",
        source_blob_sha256="19a778e6453dfc6ff70cd1ab4260d4c313cf8b822a2a3c272d3736f0c0bf1e1b",
        artifact_sha256="1a071b9814203286dc097777b00b07021862acb66f4203add710920b3ea7736c",
    ),
    _COMPACT + "/representation_timing_pairs.parquet": dict(
        metadata_path=_COMPACT + "/manifest.json",
        metadata_sha256="62851f2c14ed0878b3aadcd6dfd216190428a39aa89e238a029c3c8bcf10ee20",
        recorded_source_sha="7472d7ccf9bb670b6921398cada04f6cfd2a5f85",
        source_blob_sha256="19a778e6453dfc6ff70cd1ab4260d4c313cf8b822a2a3c272d3736f0c0bf1e1b",
        artifact_sha256="33252fd4182863b6cf1c590094c30fcc31de13baeef9f4604b3318a98dbfdf89",
    ),
}


def augment_manifest_binding(
    previous: dict[str, Any],
    original_row: dict[str, Any],
    transaction: dict[str, Any],
    original_claims: list[dict[str, Any]],
    claim: dict[str, Any],
    *,
    source_closure: Mapping[str, str],
    previous_reviewed_writers: Mapping[tuple[str, str], str],
    reviewed_writers: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """Reuse one saved transaction and admit only its exact registered writer."""
    if not isinstance(previous, dict) or not isinstance(original_row, dict):
        raise ValueError("explicit original inventory and completed binding witnesses required")
    _bounded(previous)
    if (
        previous.get("schema") != "historical_parquet_binding_recovery_v49"
        or previous.get("status") not in {"unrecovered", "artifact_binding_only"}
        or previous.get("original_producer_commit") is not None
        or any(previous.get(name) is not False for name in contract.FALSE_FIELDS)
    ):
        raise ValueError("only an original unresolved unqualified-writer binding may advance")
    registered = REGISTERED.get(original_row.get("path"))
    if registered is None:
        raise ValueError("target is outside the three registered manifest bindings")
    if not isinstance(claim, dict):
        raise ValueError("exact original manifest claim required")
    expected_claim = dict(
        registered,
        role="runtime_writer",
        writer_path=WRITER,
        source_evidence_verified=True,
    )
    if claim != expected_claim or type(claim.get("source_evidence_verified")) is not bool:
        raise ValueError("original manifest/source/role/artifact binding differs")
    expected_closure = {WRITER: registered["source_blob_sha256"], GUARD: GUARD_SHA}
    if not isinstance(source_closure, Mapping) or dict(source_closure) != expected_closure:
        raise ValueError("complete original writer and clean-source guard closure required")
    old = classify_binding(
        original_row,
        transaction,
        original_claims,
        reviewed_writers=previous_reviewed_writers,
    )
    if digest(old) != digest(previous):
        raise ValueError("saved original witness differs from its bound classifier inputs")
    new = classify_binding(
        original_row,
        transaction,
        [*original_claims, claim],
        reviewed_writers=reviewed_writers,
    )
    if (
        new["status"] != "runtime_producer_and_artifact_binding_verified"
        or new["original_producer_commit"] != registered["recorded_source_sha"]
        or new["original_producer_writer_path"] != WRITER
        or new["original_producer_source_blob_sha256"] != registered["source_blob_sha256"]
    ):
        raise ValueError("registered writer failed the unchanged binding classifier")
    # Identity/semantic recovery remains explicitly separate from source binding.
    result = copy.deepcopy(new)
    result.update(
        schema="historical_original_manifest_binding_v52",
        parent_binding_sha256=digest(previous),
        original_manifest_claim_sha256=digest(claim),
        original_writer_closure_sha256=digest(expected_closure),
        binding_extension_schema="registered_original_manifest_field_v52",
        reused_hash_transaction=True,
        newly_resolved_occurrences=0,
        occurrence_resolved=False,
        GPU_jobs=0,
        **dict.fromkeys(contract.FALSE_FIELDS, False),
    )
    _bounded(result)
    return result
