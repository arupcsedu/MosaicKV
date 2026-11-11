#!/usr/bin/env python3
"""Recover completed V12 COCO hash observations on CPU; never release a cohort."""

import argparse
import json
from pathlib import Path

from mosaickv.evaluation.coco_hash_recovery import recover
from mosaickv.evaluation.fox_campaign import clean_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = clean_source(Path(__file__).resolve().parents[2])
    print(json.dumps(recover(args.output, reader_source_sha=revision), sort_keys=True))


if __name__ == "__main__":
    main()
