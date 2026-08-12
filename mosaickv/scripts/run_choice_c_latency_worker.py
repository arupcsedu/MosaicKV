#!/usr/bin/env python3
"""Run one side of the frozen Choice-C host-transfer latency A/B test."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml
from run_choice_c_gate import _load_ledger, _mosaic_config, _run, _sha

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.hf_cli import _model_kwargs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("per_group_reference", "deferred_layer"),
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--replica", required=True)
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    profile_path = args.profile.resolve()
    ledger_path = args.ledger.resolve()
    output_root = args.output_root.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    profile = json.loads(profile_path.read_text())
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen fidelity profile has no parameter mapping")
    ledger, ledger_sha = _load_ledger(ledger_path)
    if not args.replica.replace("-", "").isalnum():
        raise ValueError("replica must contain only letters, digits, and hyphens")
    config = _mosaic_config(
        parameters,
        ledger,
        ledger_sha,
        retention_ratio=float(spec["retention_ratio"]),
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
    )
    config = replace(
        config,
        graph=replace(config.graph, attention_host_transfer_mode=args.mode),
        prototypes=replace(
            config.prototypes,
            source_gather_mode=(
                "per_head_reference"
                if args.mode == "per_group_reference"
                else "batched_layer"
            ),
        ),
    )
    adapter = load_hf_adapter(
        config.model.id,
        revision=config.model.revision,
        model_kwargs=_model_kwargs(config, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    warmup_ledger = dict(ledger)
    warmup_ledger["sample_ids"] = [cast("list[str]", ledger["sample_ids"])[0]]
    warmup_ledger["sample_count"] = 1
    warmup_path = output_root / "warmup_ledgers" / f"{args.mode}-{args.replica}.json"
    warmup_payload = json.dumps(warmup_ledger, indent=2, sort_keys=True) + "\n"
    warmup_path.parent.mkdir(parents=True, exist_ok=True)
    if warmup_path.exists() and warmup_path.read_text() != warmup_payload:
        raise RuntimeError(f"refusing to change existing warmup ledger: {warmup_path}")
    warmup_path.write_text(warmup_payload)
    warmup_sha = _sha(warmup_path)
    warmup_config = replace(
        config,
        diagnostics=replace(
            config.diagnostics,
            sample_ids=tuple(cast("list[str]", warmup_ledger["sample_ids"])),
            sample_ledger_sha=warmup_sha,
        ),
    )
    _run(
        warmup_config,
        adapter,
        run_id=f"choice-c-latency-warmup-{args.mode}-{args.replica}",
        output_root=output_root,
        ledger_path=warmup_path,
        ledger_sha=warmup_sha,
    )
    run_id = f"choice-c-latency-{args.mode}-{args.replica}-{_sha(spec_path)[:10]}"
    _run(
        config,
        adapter,
        run_id=run_id,
        output_root=output_root,
        ledger_path=ledger_path,
        ledger_sha=ledger_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
