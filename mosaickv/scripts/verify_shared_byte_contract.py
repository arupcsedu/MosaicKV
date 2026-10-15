#!/usr/bin/env python3
"""Read-only verification of the comparison freeze; never submits jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaickv.evaluation.shared_byte_contract import (
    candidate_content_digest,
    file_digest,
    load_contract,
    validate_ledger,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-parent-ledgers", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    path = source / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml"
    contract = load_contract(path)
    if candidate_content_digest(source) != contract["candidate"]["content_sha256"]:
        raise ValueError("candidate content fingerprint differs from freeze")
    reviewed = {}
    missing = []
    for lane, rules in contract["lanes"].items():
        for task, route in rules["tasks"].items():
            if route.get("parent_ledger") and args.check_parent_ledgers:
                ids = validate_ledger(
                    Path(route["parent_ledger"]),
                    route["parent_ledger_sha256"],
                    route,
                    task,
                )
                reviewed[task] = len(ids)
            elif not route.get("parent_ledger"):
                missing.append(f"{lane}/{task}: {route['ledger_status']}")
    print(
        json.dumps(
            {
                "contract_valid": True,
                "contract_sha256": file_digest(path),
                "frozen_content_matches": True,
                "verified_parent_ledger_counts": reviewed,
                "remaining_ledger_bindings": missing,
                "execution_authorized": False,
                "submitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
