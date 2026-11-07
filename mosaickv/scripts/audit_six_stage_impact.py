#!/usr/bin/env python3
"""Write-once CPU report from existing profiles/ablations; never launches GPUs."""

import argparse
import json
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.six_stage_impact import build, markdown


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--latency-root", type=Path, required=True)
    p.add_argument("--panel-root", type=Path, required=True)
    a = p.parse_args()
    source = clean_source(Path(__file__).resolve().parents[2])
    paths = [a.latency_root / f"workers/{i}/profile.json" for i in (0, 1)]
    paths += [a.panel_root / "analysis/panel_stopping_decision.json"]
    paths += [a.latency_root / "analysis/decision.json"]
    profiles = [json.loads(path.read_text()) for path in paths[:2]]
    audit = build(profiles, json.loads(paths[2].read_text()), json.loads(paths[3].read_text()))
    manifests = [a.latency_root / f"workers/{i}/manifest.json" for i in (0, 1)]
    paths += [a.latency_root / "plan.json", a.panel_root / "plan.json", *manifests]
    audit["paired_profile_provenance"] = [json.loads(p.read_text()) for p in manifests]
    audit.update(analyzer_git_sha=source, input_sha256={str(path): sha(path) for path in paths})
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(a.output / "audit.json", audit)
    with (a.output / "report.md").open("x") as stream:
        stream.write(markdown(audit))
    print(json.dumps(dict(output=str(a.output), superiority_established=False)))


if __name__ == "__main__":
    main()
