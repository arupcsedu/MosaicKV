#!/usr/bin/env python3
"""Fail-closed parity and invariant gate for native serving transactions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_directory(directory: Path) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"trace root is not an object: {path}")
        sample_id = str(payload.get("sample_id", ""))
        if not sample_id or sample_id in traces:
            raise ValueError(f"missing or duplicate sample_id in {path}")
        traces[sample_id] = payload
    if not traces:
        raise ValueError(f"no trace JSON files found under {directory}")
    return traces


def _tokens(trace: dict[str, Any]) -> tuple[int, ...]:
    trials = trace.get("trials")
    if not isinstance(trials, list) or len(trials) < 2:
        raise ValueError("each parity trace requires at least two deterministic trials")
    first: tuple[int, ...] | None = None
    request_ids: set[str] = set()
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError(f"trial {index} is not an object")
        current = tuple(int(token) for token in trial.get("token_ids", ()))
        if not current:
            raise ValueError(f"trial {index} produced no token IDs")
        if first is None:
            first = current
        elif current != first:
            raise ValueError(f"trial {index} is not deterministic")
        request_id = str(trial.get("request_id", ""))
        if not request_id or request_id in request_ids:
            raise ValueError("trials must have unique non-empty request IDs")
        request_ids.add(request_id)
        for field in ("request_latency_seconds", "throughput_tokens_per_second"):
            value = float(trial[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"trial {index} has invalid {field}")
    assert first is not None
    return first


def _selection(trace: dict[str, Any], *, identity: bool) -> dict[str, Any]:
    expected_type = f"{trace['backend']}_native_mosaickv"
    if trace.get("measurement_type") != expected_type:
        raise ValueError(
            f"native trace has incorrect measurement_type: {trace.get('measurement_type')}"
        )
    if trace.get("method") != "mosaickv_exact" or trace.get("native_mosaickv") is not True:
        raise ValueError("native trace is not explicitly labeled mosaickv_exact")
    if trace.get("native_scope") != "request_wide_whole_token_blocks_transport_gate":
        raise ValueError("native trace omitted its restricted transport-validation scope")
    trials = trace["trials"]
    selections: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        audit = trial.get("native_audit")
        if not isinstance(audit, dict):
            raise ValueError(f"native trial {index} omitted the scheduler audit")
        if audit.get("status") != "scheduler_transaction_committed":
            raise ValueError(f"native trial {index} did not commit its scheduler transaction")
        current = audit.get("selection")
        if not isinstance(current, dict):
            raise ValueError(f"native trial {index} omitted selection metadata")
        if int(audit.get("logical_next_position", -1)) != int(current["logical_positions"]):
            raise ValueError(f"native trial {index} changed the next logical decode position")
        compression = float(audit.get("compression_seconds", -1))
        if not math.isfinite(compression) or compression < 0:
            raise ValueError(f"native trial {index} has invalid compression timing")
        selections.append(current)
    first = selections[0]
    for current in selections[1:]:
        for key in (
            "logical_positions",
            "block_size",
            "source_blocks",
            "selected_blocks",
            "mandatory_blocks",
            "selected_positions",
        ):
            if current[key] != first[key]:
                raise ValueError(f"native trials selected different {key}")
    source = int(first["source_blocks"])
    selected = tuple(int(value) for value in first["selected_blocks"])
    mandatory = tuple(int(value) for value in first["mandatory_blocks"])
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("selected blocks are not strictly increasing and unique")
    if not set(mandatory).issubset(selected):
        raise ValueError("mandatory blocks were dropped")
    if mandatory != tuple(sorted({0, source - 1})):
        raise ValueError("native transaction did not mark both boundary blocks mandatory")
    if identity:
        if selected != tuple(range(source)):
            raise ValueError("retention 1.0 did not preserve every source block")
        if int(first["selected_positions"]) != int(first["logical_positions"]):
            raise ValueError("retention 1.0 changed the active prompt length")
    elif len(selected) >= source:
        raise ValueError("compressed gate did not free any physical source block")
    return first


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("vllm", "sglang"))
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--identity-dir", type=Path, required=True)
    parser.add_argument("--compressed-dir", type=Path, required=True)
    args = parser.parse_args()

    full = _load_directory(args.full_dir)
    identity = _load_directory(args.identity_dir)
    compressed = _load_directory(args.compressed_dir)
    if not (set(full) == set(identity) == set(compressed)):
        raise ValueError("FullKV, identity, and compressed sample IDs differ")

    summaries: list[dict[str, object]] = []
    for sample_id in sorted(full):
        reference = full[sample_id]
        if reference.get("backend") != args.backend:
            raise ValueError(f"FullKV trace has wrong backend for {sample_id}")
        if reference.get("measurement_type") != f"{args.backend}_fullkv":
            raise ValueError(f"reference trace is not FullKV for {sample_id}")
        if reference.get("method") != "full_kv" or reference.get("native_mosaickv") is not False:
            raise ValueError(f"reference trace is mislabeled for {sample_id}")
        for candidate in (identity[sample_id], compressed[sample_id]):
            for field in ("prompt_sha256", "media_sha256", "model", "generation"):
                if candidate.get(field) != reference.get(field):
                    raise ValueError(f"{field} differs across parity runs for {sample_id}")
        reference_tokens = _tokens(reference)
        identity_tokens = _tokens(identity[sample_id])
        compressed_tokens = _tokens(compressed[sample_id])
        if identity_tokens != reference_tokens:
            raise ValueError(f"retention 1.0 token parity failed for {sample_id}")
        identity_selection = _selection(identity[sample_id], identity=True)
        compressed_selection = _selection(compressed[sample_id], identity=False)

        isolation = compressed[sample_id].get("request_isolation")
        if sample_id == sorted(compressed)[0]:
            if args.backend == "sglang" and isinstance(isolation, dict):
                isolation = isolation.get("post_intervening_request_probe")
            if not isinstance(isolation, dict) or isolation.get("performed") is not True:
                raise ValueError("compressed A-B-A request-isolation probe was not recorded")
            if isolation.get("token_ids_match_anchor") is not True:
                raise ValueError("compressed A-B-A request-isolation token parity failed")

        summaries.append(
            {
                "sample_id": sample_id,
                "identity_token_parity": True,
                "identity_source_blocks": int(identity_selection["source_blocks"]),
                "compressed_selected_blocks": len(compressed_selection["selected_blocks"]),
                "compressed_source_blocks": int(compressed_selection["source_blocks"]),
                "compressed_generated_tokens": len(compressed_tokens),
            }
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": args.backend,
                "scope": "version_pinned_native_whole_block_transport",
                "paper_method_rows_authorized": False,
                "reason": "common MosaicKV selector-plan bridge is outside this transport gate",
                "samples": summaries,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
