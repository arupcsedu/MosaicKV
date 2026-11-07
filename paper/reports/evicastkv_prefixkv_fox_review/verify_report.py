#!/usr/bin/env python3
"""Verify report-only hashes and a compiled PDF; never run model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((HERE / "provenance.json").read_text())
    counts = {}
    for group in ["source_artifact_sha256", "derived_sha256", "latex_sources_sha256"]:
        entries = provenance[group]
        for name, expected in entries.items():
            path = Path(name) if group == "source_artifact_sha256" else HERE / name
            if digest(path) != expected:
                raise ValueError(f"stale or changed report input/output: {path}")
        counts[group] = len(entries)
    if digest(HERE / "export_report.py") != provenance["export_script_sha256"]:
        raise ValueError("exporter changed after provenance export")
    log = args.log.read_text()
    forbidden = ["Overfull", "undefined", "LaTeX Error:", "Emergency stop"]
    if any(message in log for message in forbidden):
        raise ValueError("PDF log contains an overflow, unresolved reference, or error")
    text = subprocess.check_output(
        ["pdftotext", "-layout", str(args.pdf), "-"], text=True
    )
    for token in ["2,631", "109.271", "93.787", "103.990", "22.92", "111.48"]:
        if token not in text:
            raise ValueError(
                f"expected measured value absent from compiled PDF: {token}"
            )
    print(
        json.dumps(
            {
                "status": "passed_report_only",
                "verified_file_counts": counts,
                "pdf_sha256": digest(args.pdf),
                "compiler_log_sha256": digest(args.log),
                "independent_quality_established": False,
                "new_gpu_inference": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
