#!/usr/bin/env python3
"""CPU replay of a separately pinned GPU producer; never a new GPU measurement.

Preserve producer SHA 21fbfe2 and its original gate. New validation uses the
stronger receipt/ownership cross-link checks without pretending those saved
requests executed the analyzer's newer source revision.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath

from evicastkv_bounded_reader_v26 import parity, read

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.whole_request_kv_ledger import replay_whole_request

SOURCE = Path(__file__).resolve().parents[2]
PRODUCER = "21fbfe2aaa98a1e7ab73f315c65616053bc4e599"
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v31/run-20261111-1")
PARENT_PLAN_SHA = "74f70c760530e5e514415b779fd8ccdaf7ccffaf4e8367db53ca369c091e5330"
REQUIRED_OUTPUTS = frozenset(
    {
        "raw.jsonl",
        "logits.pt",
        "manifest.json",
        "prefill-lifecycle.json",
        "traces/result.json",
        "traces/whole_allocator_ownership.json",
    }
)
SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_worker_outputs(out, result):
    """Every critical read must be covered by the original producer manifest."""
    if (
        not isinstance(result, dict)
        or result.get("source_sha") != PRODUCER
        or result.get("plan_sha256") != PARENT_PLAN_SHA
        or result.get("failures") != []
        or type(result.get("actual_rows")) is not int
        or result["actual_rows"] != 1
    ):
        raise ValueError("complete source/plan-matched original worker required")
    outputs = result.get("outputs")
    if not isinstance(outputs, dict) or not outputs.keys() >= REQUIRED_OUTPUTS:
        raise ValueError("all six critical original worker outputs require explicit hash bindings")
    out = Path(out).resolve(strict=True)
    resolved = {}
    for name, expected in outputs.items():
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"..", "."} for part in PurePosixPath(name).parts)
            or PurePosixPath(name).as_posix() != name
            or not isinstance(expected, str)
            or not SHA256.fullmatch(expected)
        ):
            raise ValueError(
                "canonical relative output names and explicit SHA-256 bindings required"
            )
        path = (out / name).resolve(strict=True)
        if not path.is_relative_to(out) or not path.is_file():
            raise ValueError("original worker output escapes its immutable artifact directory")
        if sha(path) != expected:
            raise ValueError("registered original worker artifact changed")
        resolved[name] = path
    if len({resolved[name] for name in REQUIRED_OUTPUTS}) != len(REQUIRED_OUTPUTS):
        raise ValueError("distinct critical artifacts cannot alias one backing file")
    return resolved


def validate_ownership_sidecar(row, outputs, result):
    """The row's physical ownership proof must be the hash-bound critical sidecar."""
    whole = row.get("whole_KV")
    if not isinstance(whole, dict) or not isinstance(whole.get("allocator"), dict):
        raise ValueError("complete physical ownership proof is required")
    evidence = whole.get("evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("native_CUDA_free_completed_verified") is not True
        or not isinstance(evidence.get("native_retirement_receipts"), list)
        or not evidence["native_retirement_receipts"]
    ):
        raise ValueError("critical actual native-owner retirement proof is missing")
    allocator = whole["allocator"]
    if (
        allocator.get("constructor_snapshot_reuse") is not True
        or not isinstance(allocator.get("snapshot_cache"), dict)
        or allocator["snapshot_cache"].get("every_observation_has_proof") is not True
    ):
        raise ValueError("registered constructor snapshot-reuse proofs are missing")
    name = "traces/whole_allocator_ownership.json"
    if (
        name not in outputs
        or name not in result.get("outputs", {})
        or allocator.get("ownership_delta_artifact") != name
        or allocator.get("ownership_delta_sha256") != result["outputs"][name]
    ):
        raise ValueError(
            "physical ownership sidecar differs from its original required hash binding"
        )
    return outputs[name]


def validate_parent(plan, gate):
    if (
        plan.get("source_sha") != PRODUCER
        or gate.get("source_sha") != PRODUCER
        or gate.get("plan_sha256") != PARENT_PLAN_SHA
        or gate.get("failures") != []
        or any(
            gate.get(key) is not True
            for key in ("complete", "source_fidelity_128_pass", "whole_request_KV_byte_match_pass")
        )
    ):
        raise ValueError(
            "complete passed original producer gate required; do not duplicate inference"
        )


def crosslink_native_owners(record, deltas, *, layers=32):
    """An unrelated valid CUDA free cannot stand in for the native cache."""
    if type(layers) is not int or layers < 1:
        raise ValueError("positive strict native layer count required")
    evidence = record.get("evidence")
    receipts = evidence.get("native_retirement_receipts") if isinstance(evidence, dict) else None
    if (
        not isinstance(evidence, dict)
        or evidence.get("native_CUDA_free_completed_verified") is not True
        or not isinstance(receipts, list)
        or len(receipts) != layers
    ):
        raise ValueError("complete actual native layer receipts required")
    for layer, receipt in enumerate(receipts):
        if (
            not isinstance(receipt, dict)
            or receipt.get("actual_free_completed_verified") is not True
            or receipt.get("label") != f"layer_{layer}"
            or not isinstance(receipt.get("allocations"), list)
            or not receipt["allocations"]
        ):
            raise ValueError("ordered actual native free-completed allocation receipts required")
        for allocation in receipt["allocations"]:
            if (
                not isinstance(allocation, dict)
                or not isinstance(allocation.get("owner_ranges"), list)
                or not allocation["owner_ranges"]
            ):
                raise ValueError("complete native owner-range proof required")
            if any(
                not isinstance(backing, list)
                or len(backing) != 2
                or any(type(value) is not int or value < 1 for value in backing)
                for backing in allocation["owner_ranges"]
            ):
                raise ValueError(
                    "native owner ranges require positive strict pointer/byte integers"
                )
    native = [
        {
            tuple(backing)
            for allocation in receipt["allocations"]
            for backing in allocation["owner_ranges"]
        }
        for receipt in receipts
    ]
    live, published, retired = set(), set(), set()
    for delta in deltas:
        live.difference_update(tuple(value) for value in delta["removed_owners"])
        live.update(tuple(value) for value in delta["added_owners"])
        for layer, owners in enumerate(native):
            if delta["phase"] == f"prefill_layer_{layer}_KV_published":
                if layer in published or not owners or not owners <= live:
                    raise ValueError("native receipt does not bind the actual published KV owners")
                published.add(layer)
            if delta["phase"] == f"prefill_layer_{layer}_native_retired":
                if layer not in published or layer in retired or owners & live:
                    raise ValueError("native receipt cannot retire still-declared KV owners")
                retired.add(layer)
    if published != set(range(layers)) or retired != published:
        raise ValueError("complete native publication/retirement cross-links required")
    return dict(
        verified_layers=layers,
        published_native_owner_ranges_bound=True,
        retired_native_owner_ranges_absent=True,
    )


def adjudicate(output):
    import evicastkv_review_source_prefix as fixed
    import torch

    revision = clean_source(SOURCE)
    output = Path(output).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError("new immutable derived-analysis directory required")
    plan_path = PARENT / "plan.json"
    if sha(plan_path) != PARENT_PLAN_SHA:
        raise ValueError("registered parent plan changed")
    parent_gate_path = PARENT / "analysis/GATE_bounded_reader_v28.json"
    plan, parent_gate = read(plan_path), read(parent_gate_path)
    validate_parent(plan, parent_gate)
    bindings = {str(path): sha(path) for path in (plan_path, parent_gate_path)}
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"registered producer input changed: {name}")
        bindings[name] = expected
    fixed.analysis_cpu_controls()
    failures, records = [], []
    for index in range(4):
        try:
            out = PARENT / f"workers/{index}"
            result = read(out / "result.json")
            outputs = validate_worker_outputs(out, result)
            bindings[str(out / "result.json")] = sha(out / "result.json")
            for name, path in outputs.items():
                bindings[str(path)] = result["outputs"][name]
            rows = [json.loads(line) for line in outputs["raw.jsonl"].read_text().splitlines()]
            if len(rows) != 1:
                raise ValueError("one complete registered arm per worker required")
            row = rows[0]
            arm = ("source", "raw")[index % 2]
            if row["arm"] != arm or row["worker"] != index or row["fullkv_fallback"] is not False:
                raise ValueError("worker identity/arm/fallback changed")
            tensors = torch.load(outputs["logits.pt"], map_location="cpu", weights_only=True)
            numerical = parity(tensors["expected"], tensors["actual"])
            if numerical["pass_128"] is not True or digest(numerical) != digest(row["parity"]):
                raise ValueError("actual saved 128-prediction fidelity failed")
            fixed.verify_recomputed_diagnostics(
                fixed.comparisons(tensors["FullKV"], tensors["actual"]), row["FullKV_diagnostic"]
            )
            delta_path = validate_ownership_sidecar(row, outputs, result)
            deltas = read(delta_path)["ownership_deltas"]
            replay = replay_whole_request(
                row["whole_KV"], deltas, layer_count=32, source=arm == "source"
            )
            role_proof = crosslink_native_owners(row["whole_KV"], deltas)
            records.append(
                dict(
                    worker=index,
                    arm=arm,
                    sample_id=row["sample_id"],
                    task=row["task"],
                    parity=numerical,
                    boundaries=replay,
                    ownership_artifact=str(delta_path),
                    ownership_sha256=sha(delta_path),
                    observation_count=len(deltas),
                    native_role_crosslink=role_proof,
                )
            )
        except BaseException as exc:
            failures.append(dict(worker=index, error=f"{type(exc).__name__}: {exc}"))
    pairs = []
    if not failures:
        for index in (0, 2):
            left, right = records[index : index + 2]
            keys = [
                f"{phase}_{kind}_bytes"
                for phase in ("initial", "final", "peak")
                for kind in ("tensor_backing", "allocator_block")
            ]
            if (left["sample_id"], left["task"]) != (right["sample_id"], right["task"]):
                raise ValueError("pair identity mismatch")
            pairs.append(
                dict(
                    sample_id=left["sample_id"],
                    task=left["task"],
                    exact_bytes_matched=all(
                        left["boundaries"][key] == right["boundaries"][key] for key in keys
                    ),
                    source={key: left["boundaries"][key] for key in keys},
                    raw={key: right["boundaries"][key] for key in keys},
                )
            )
    complete = not failures and len(records) == 4
    gate = dict(
        source_sha=PRODUCER,
        actual_execution_source_sha=PRODUCER,
        validation_source_sha=revision,
        gate_origin="derived_validation_not_new_GPU_execution",
        independent_replay_from_actual_GPU_artifacts=complete,
        actual_execution_root=str(PARENT.resolve()),
        actual_execution_root_sha256=digest(str(PARENT.resolve())),
        actual_plan_path=str(plan_path.resolve()),
        actual_plan_sha256=PARENT_PLAN_SHA,
        actual_gate_path=str(parent_gate_path.resolve()),
        actual_gate_sha256=sha(parent_gate_path),
        input_gate=str(parent_gate_path),
        input_gate_sha256=sha(parent_gate_path),
        input_plan_sha256=PARENT_PLAN_SHA,
        complete=complete,
        failures=failures,
        records=records,
        observations=pairs,
        source_fidelity_128_pass=complete,
        whole_request_KV_peak_certified=complete,
        whole_request_KV_byte_match_pass=complete
        and len(pairs) == 2
        and all(p["exact_bytes_matched"] for p in pairs),
        whole_request_GPU_peak_certified=False,
        independent_quality_confirmation=False,
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(
        output / "provenance.json",
        dict(
            execution_source_sha=PRODUCER,
            analyzer_source_sha=revision,
            bindings=bindings,
            preserves_original_artifacts=True,
            GPU_inference_repeated=False,
        ),
    )
    write_once(output / "GATE_bounded_reader_v27.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = adjudicate(args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "complete",
                    "source_fidelity_128_pass",
                    "whole_request_KV_byte_match_pass",
                    "failures",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
