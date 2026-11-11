#!/usr/bin/env python3
"""Write-once CPU evidence recovery; never submit or run GPU inference."""

import argparse
from pathlib import Path

from mosaickv.evaluation.coco_numerical_status import audit
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    revision = clean_source(source)
    root = args.root.resolve()
    scratch = Path("/scratch/djy8hg").resolve()
    if root == scratch or not root.is_relative_to(scratch) or root.exists():
        raise ValueError("new immutable scratch recovery root required")
    record = audit()
    write_once(root / "GATE_coco_numerical_recovery.json", dict(source_sha=revision, **record))
    write_once(
        root / "manifest.json",
        dict(
            source_sha=revision,
            producer=str(Path(__file__).resolve()),
            producer_sha256=sha(Path(__file__).resolve()),
            gate_sha256=sha(root / "GATE_coco_numerical_recovery.json"),
            measurement_type="CPU_only_recovery_of_sealed_consumed_input_numerical_evidence",
        ),
    )


if __name__ == "__main__":
    main()
