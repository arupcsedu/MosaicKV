#!/usr/bin/env python3
"""CPU replay of a separately pinned GPU producer; never a new GPU measurement.

Preserve producer SHA f6defdf and its original gate. New validation uses the
stronger receipt/ownership cross-link checks without pretending those saved
requests executed the analyzer's newer source revision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evicastkv_bounded_reader_v26 import parity, read

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.whole_request_kv_ledger import replay_whole_request

SOURCE = Path(__file__).resolve().parents[2]
PRODUCER = "f6defdfe189b71bffa9ef400943ac1e9b8a6c63f"
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v28/run-20261111-1")
PARENT_PLAN_SHA = "dd7da6d872669d4a0e900403612bc023ad99bb9271c2935b36837a87bd369083"


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
    receipts = record["evidence"]["native_retirement_receipts"]
    if len(receipts) != layers:
        raise ValueError("complete actual native layer receipts required")
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
            if (
                result.get("source_sha") != PRODUCER
                or result.get("plan_sha256") != PARENT_PLAN_SHA
                or result.get("failures") != []
                or result.get("actual_rows") != 1
            ):
                raise ValueError("complete registered original worker required")
            bindings[str(out / "result.json")] = sha(out / "result.json")
            for name, expected in result["outputs"].items():
                path = (out / name).resolve(strict=True)
                if not path.is_relative_to(out.resolve()) or sha(path) != expected:
                    raise ValueError("registered original worker artifact changed")
                bindings[str(path)] = expected
            rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
            if len(rows) != 1:
                raise ValueError("one complete registered arm per worker required")
            row = rows[0]
            arm = ("source", "raw")[index % 2]
            if row["arm"] != arm or row["worker"] != index or row["fullkv_fallback"] is not False:
                raise ValueError("worker identity/arm/fallback changed")
            tensors = torch.load(out / "logits.pt", map_location="cpu", weights_only=True)
            numerical = parity(tensors["expected"], tensors["actual"])
            if numerical["pass_128"] is not True or digest(numerical) != digest(row["parity"]):
                raise ValueError("actual saved 128-prediction fidelity failed")
            fixed.verify_recomputed_diagnostics(
                fixed.comparisons(tensors["FullKV"], tensors["actual"]), row["FullKV_diagnostic"]
            )
            allocator = row["whole_KV"]["allocator"]
            delta_path = out / allocator["ownership_delta_artifact"]
            if sha(delta_path) != allocator["ownership_delta_sha256"]:
                raise ValueError("actual ownership delta hash changed")
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
