#!/usr/bin/env python3
"""CPU review closure and one bounded source-length diagnostic; dry by default."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import evicastkv_review_v8_diagnostics as shared

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/sfs/weka/scratch/djy8hg/workdir")
V9 = Path("/scratch/djy8hg/audits/fox-review-v9-shape-normalization-20261110")
INVENTORY = Path(
    "/scratch/djy8hg/audits/fox-review-v8-consumed-diagnostics-20261110/"
    "freshness-continuation/aggregate/inventory.json"
)
MANUSCRIPT = WORKSPACE / "fox-review-v10/PROPOSED_MANUSCRIPT_PATCH.md"
BINDINGS = {
    str(
        V9 / "FINAL_report.json"
    ): "17e162cbe05208415f9184551740b1015c48803d2ab52049a57e48018e0e95bb",
    str(INVENTORY): "15d9f114918224312c6ec53537305732daaa6208390cca9f4076f5459eb77005",
}
BYTE_GATE_SHA = "7f018924fd22b222822ce4feeff55c1be677b06547691f680a72d3d1f65d2095"
LANES = {
    "bytes": "evicastkv_review_byte_feasibility",
    "freshness": "evicastkv_freshness_producer_canary",
    "scaling": "evicastkv_source_length_scaling",
}


def module(lane):
    return importlib.import_module(LANES[lane])


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"bound review artifact changed: {path}")


def validate_root(root):
    root = Path(root).resolve()
    if not root.is_relative_to(WORKSPACE / "fox-review-v10"):
        raise ValueError("write-once review output must be under the V10 scratch workspace")
    return root


def prepare(root):
    """CPU execution only, with no scheduler access."""
    revision = clean_source(SOURCE)
    verify_bindings(BINDINGS)
    cpu = shared.cpu_gate(root, revision)
    v9 = shared.read(V9 / "FINAL_report.json")
    norm = str(V9 / "norm/GATE_shape_transport.json")
    if (
        v9["errors"]
        or v9["artifacts"][norm]["value"]["status"] != "passed_new_shape_transport"
        or sha(Path(norm)) != v9["artifacts"][norm]["sha256"]
    ):
        raise ValueError("V9 numerical gate or provenance incomplete")
    bindings = {**BINDINGS, norm: sha(Path(norm)), str(MANUSCRIPT): sha(MANUSCRIPT)}
    bindings.update(
        {
            str(SOURCE / f"mosaickv/scripts/{name}.py"): sha(SOURCE / f"mosaickv/scripts/{name}.py")
            for name in LANES.values()
        }
    )
    write_once(
        root / "registration.json",
        dict(
            source_root=str(SOURCE),
            source_sha=revision,
            bindings=bindings,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            worker_count=1,
            worker_walltime_seconds=1200,
            synthetic_scaling_only=True,
            scope="saved-evidence CPU audits and one consumed-image source operator scaling worker",
            old_numerical_and_quality_negatives_preserved=True,
            no_new_quality_rows=True,
            fresh_confirmation_authorized=False,
            freeze_authorized=False,
            paper_admission_authorized=False,
        ),
    )
    write_once(root / "GATE_cpu_suite.json", cpu)
    try:
        geometry = module("bytes")
        result = geometry.derive(geometry.PARENT, BYTE_GATE_SHA)
        result.update(
            analysis_source_sha=revision, analysis_script_sha256=sha(Path(geometry.__file__))
        )
        write_once(root / "byte-feasibility.json", result)
    except Exception as error:
        shared.failure(root, "byte-feasibility", error)
    try:
        module("freshness").run(
            root / "producer-canary",
            cpu_gate_root=root,
            inventory_sha256=BINDINGS[str(INVENTORY)],
        )
    except Exception as error:
        shared.failure(root, "freshness-canary", error)
    try:
        scaling = module("scaling").prepare(root / "source-length")
        write_once(root / "source-length-prepared.json", scaling)
    except Exception as error:
        shared.failure(root, "source-length-prepare", error)
    return dict(status="cpu_review_work_complete", output=str(root))


def load(root):
    record = shared.read(root / "registration.json")
    clean_source(SOURCE, record["source_sha"])
    verify_bindings(record["bindings"])
    if (
        record["synthetic_scaling_only"] is not True
        or record["fresh_confirmation_authorized"] is not False
        or record["paper_admission_authorized"] is not False
    ):
        raise ValueError("registered diagnostic scope changed")
    return record


def submit(root):
    registration = load(root)
    if not (root / "source-length-prepared.json").is_file():
        raise ValueError("no source-length worker may run before preparation")
    shared.cpu_gate(root, registration["source_sha"])
    snapshot = shared.gpu_snapshot()
    dependencies = sorted(
        {row["job"].split("_")[0] for row in snapshot["active"] + snapshot["pending"]}
    )
    write_once(
        root / "admission.json",
        dict(
            snapshot=snapshot,
            wait_for_own_gpu_jobs=dependencies,
            workers=1,
            maximum_a100s=6,
            worker_walltime_seconds=1200,
        ),
    )
    kwargs = dict(launcher="evicastkv_review_v10", job_prefix="fox10")
    try:
        gpu = shared.submit(
            root, "source-length", "source-length-worker", dependencies, gpu=True, **kwargs
        )
        analysis = shared.submit(root, "analysis", "analyze", [gpu], **kwargs)
    except shared.SubmissionError as error:
        return shared.submission_block(root, "v10", error)
    return dict(worker=gpu, analysis=analysis)


def finish(root):
    lock = load(root)
    artifacts = {}
    for relative in (
        "GATE_cpu_suite.json",
        "byte-feasibility.json",
        "producer-canary/GATE_freshness_producer_canary.json",
        "source-length/GATE_source_length_scaling.json",
        "source-length/scaling_summary.json",
    ):
        path = root / relative
        artifacts[relative] = (
            dict(path=str(path), sha256=sha(path)) if path.exists() else dict(status="missing")
        )
    errors = {p.stem: shared.read(p) for p in sorted((root / "errors").glob("*.json"))}
    final = dict(
        source_sha=lock["source_sha"],
        status="review_diagnostics_not_confirmation",
        artifacts=artifacts,
        errors=errors,
        manuscript_proposal=str(MANUSCRIPT),
        manuscript_approval_claimed=False,
        superiority_established=False,
        fresh_confirmation_authorized=False,
        original_negative_results_preserved=True,
    )
    write_once(root / "FINAL_report.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "source-length-worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = validate_root(args.root)
    if not args.execute:
        value = dict(status="dry_run", mode=args.mode, root=str(root), workers_submitted=0)
    elif args.mode == "prepare":
        value = prepare(root)
    elif args.mode == "submit":
        value = submit(root)
    elif args.mode == "source-length-worker":
        load(root)
        if shared.gpu_snapshot()["active_gpus"] > 6:
            raise ValueError("six-A100 aggregate ceiling exceeded")
        value = module("scaling").worker(root / "source-length", index=0)
    else:
        load(root)
        try:
            module("scaling").analyze(root / "source-length")
        except Exception as error:
            shared.failure(root, "source-length-analysis", error)
        value = finish(root)
    print(json.dumps(value, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
