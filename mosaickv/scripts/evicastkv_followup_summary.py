#!/usr/bin/env python3
"""Finite afterany summary; never schedules GPU work or waives scientific gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaickv.evaluation.fox_campaign import sha, write_once


def summarize(root):
    registration = json.loads((root / "registration.json").read_text())
    reports = {}
    for name, spec in registration["reports"].items():
        path = Path(spec["path"])
        reports[name] = dict(path=str(path), exists=path.exists())
        if path.exists():
            reports[name].update(sha256=sha(path), report=json.loads(path.read_text()))
    gate = dict(
        source_sha=registration["source_sha"],
        reports=reports,
        execution_artifacts_complete=all(r["exists"] for r in reports.values()),
        confirmation_authorized=False,
        explanation=(
            "Engineering repairs and diagnostic completion cannot waive the original "
            "numerical or global media-freshness gates. No automatic successor."
        ),
    )
    write_once(root / "summary.json", gate)
    lines = ["# Bounded repair follow-up", "", "Confirmation remains disabled.", ""]
    for name, result in reports.items():
        payload = result.get("report", {})
        state = payload.get("status", "completed_diagnostic" if result["exists"] else "MISSING")
        lines.append(f"- {name}: {state}; `{result['path']}`")
    with (root / "summary.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    summarize(parser.parse_args().root)
