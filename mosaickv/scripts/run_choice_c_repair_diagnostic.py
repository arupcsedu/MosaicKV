#!/usr/bin/env python3
"""Run an evaluation-only repair mechanism diagnostic on an already-open sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml
from run_choice_c_gate import _load_ledger, _mosaic_config, _run

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.hf_cli import _model_kwargs
from mosaickv.types import RepairCandidateRanking, RepairPolicy


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--max-blocks", required=True, type=int, choices=(1, 2, 4))
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    source_spec = args.source_spec.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(source_spec.read_text()))
    source_output = Path(str(spec["output_root"])).resolve()
    profile = cast(
        "dict[str, Any]",
        json.loads((source_output / "frozen_fidelity_profile.json").read_text()),
    )
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen source profile has no parameter mapping")
    ledger_path = args.ledger.resolve()
    ledger, ledger_sha = _load_ledger(ledger_path)
    if ledger.get("selection_rule") != (
        "mechanism_diagnostic_from_already_open_v11_internal_failure"
    ):
        raise RuntimeError("repair diagnostics may only use the declared opened sample")

    config = _mosaic_config(
        parameters,
        ledger,
        ledger_sha,
        retention_ratio=float(spec["retention_ratio"]),
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
        repair_policy=RepairPolicy.ORACLE,
        repair_evaluation_only=True,
    )
    config = replace(
        config,
        repair=replace(
            config.repair,
            max_blocks_per_step=args.max_blocks,
            candidate_ranking=RepairCandidateRanking.DECISION_INFLUENCE,
            preserve_partial_prototype=True,
        ),
    )
    adapter = load_hf_adapter(
        config.model.id,
        revision=config.model.revision,
        model_kwargs=_model_kwargs(config, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"choice-c-v11-tail-oracle-r{args.max_blocks}-{_sha(source_spec)[:10]}"
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
