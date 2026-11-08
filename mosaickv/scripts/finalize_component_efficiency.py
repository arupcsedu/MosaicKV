#!/usr/bin/env python3
"""Hash-bound, write-once consolidation of completed component studies."""

import argparse
import json
from pathlib import Path

from mosaickv.evaluation.component_efficiency_decision import build, markdown
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    base = Path("/scratch/djy8hg/runs/evicastkv")
    roots = [
        base / name
        for name in (
            "component-admission-paired-20261107-v3",
            "source-dictionary-screen-20261107-v1",
            "source-dictionary-facility-screen-20261107-v1",
            "gray-source-screen-20261107-v1",
        )
    ]
    paths = [root / "analysis/decision.json" for root in roots]
    decision = build(*(json.loads(path.read_text()) for path in paths))
    stage_path = base / "six-stage-impact-20261107-v6-retained/audit.json"
    stages = json.loads(stage_path.read_text())
    if stages["superiority_established"] or len(stages["stages"]) != 6:
        raise ValueError("unmodified non-superiority six-stage context required")
    decision["earlier_stage_context"] = stages["stages"]
    paths.append(stage_path)
    for root in roots:
        paths += [root / "plan.json", root / "submission.json"]
        paths += list((root / "workers").glob("*/manifest.json"))
        paths += list((root / "workers").glob("*/result.json"))
        paths += list((root / "workers").glob("*/raw.jsonl"))
    decision.update(
        analyzer_git_sha=clean_source(Path(__file__).resolve().parents[2]),
        input_sha256={str(path): sha(path) for path in paths},
    )
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(a.output / "decision.json", decision)
    with (a.output / "report.md").open("x") as stream:
        stream.write(markdown(decision))
    print(json.dumps(dict(output=str(a.output), independent_quality_superiority=False)))


if __name__ == "__main__":
    main()
