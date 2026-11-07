#!/usr/bin/env python3
"""ID-only availability audit and calibration preparation; never confirms a claim.

Inference uses the unchanged existing clean source SHA, not this dirty tree.
The audit excludes even materialized/reserved IDs, not only completed examples.
No benchmark outcomes are used to select either cohort. Confirmation is ID-only
reserved and remains unmaterialized until calibration/power/freeze succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path

import yaml

COCO_REVISION = "c3213b54e173ae86517d481de17f31439d6f8c72"
COCO_ID = re.compile(r"(?:coco_caption:)?(COCO_val2014_\d{12}\.jpg)$")
MILE_ID = re.compile(
    r"(?:milebench:)?((?:ActionPrediction|ActionSequence|CLEVR-Change|Spot-the-Diff)-\d{4})$"
)
PRUNED = {
    "outputs",
    "attempts",
    "analysis",
    "source",
    "media",
    "configs",
    "logs",
    ".git",
    "third_party",
    "warmup-traces",
    "traces",
    "build",
    "hf",
    "vllm",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identities(value: object) -> tuple[set[str], set[str]]:
    """Read identity fields only; never score/filter using captions or answers."""
    coco, mile = set(), set()

    def add(item):
        if isinstance(item, str):
            c, m = COCO_ID.fullmatch(item), MILE_ID.fullmatch(item)
            if c:
                coco.add("coco_caption:" + c[1])
            if m:
                mile.add(m[1])
        elif isinstance(item, list):
            for child in item:
                add(child)

    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {
                    "sample_id",
                    "sample_ids",
                    "excluded_ids",
                    "selected_ids",
                } or key.endswith("_sample_ids"):
                    add(child)
                elif isinstance(child, (list, dict)):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                if isinstance(child, (list, dict)):
                    visit(child)

    visit(value)
    return coco, mile


def prior_identity_inventory(root: Path, *, seconds: float = 90) -> dict:
    """Bounded metadata scan, excluding output trees/scheduler JSONL entirely."""
    start = time.monotonic()
    root = root.resolve(strict=True)
    coco, mile, files, seen = set(), set(), {}, set()
    seeds = set()
    for pattern in (
        "*/plan.json",
        "*/*/plan.json",
        "*/materialization.json",
        "*/historical_ids.json",
        "*/ledger.json",
        "*/*/ledger.json",
        "*/*exclusion*.json",
        "*/*/*exclusion*.json",
    ):
        seeds.update(root.glob(pattern))
    pending = sorted(seeds)

    def references(item, key=""):
        if isinstance(item, dict):
            for name, child in item.items():
                references(child, name)
        elif isinstance(item, list):
            for child in item:
                references(child, key)
        elif (
            isinstance(item, str)
            and item.startswith("/")
            and item.endswith(".json")
            and (
                key in {"ledger", "ledgers", "ledger_path", "parent_plan", "source_ledger", "ids"}
                or "ledger" in key
                or "/ledgers/" in item
                or item.endswith("historical_ids.json")
            )
        ):
            path = Path(item).resolve(strict=True)
            if root in path.parents and not (set(path.relative_to(root).parts) & PRUNED):
                pending.append(path)

    while pending:
        if time.monotonic() - start > seconds:
            raise TimeoutError("ID audit exceeded its bounded metadata scan budget")
        path = pending.pop().resolve(strict=True)
        if root not in path.parents:
            raise ValueError(f"metadata link escapes audit root: {path}")
        if set(path.relative_to(root).parts) & PRUNED or path in seen:
            continue
        seen.add(path)
        if path.stat().st_size > 64 * 2**20:
            raise ValueError(f"unexpectedly large metadata file: {path}")
        payload = json.loads(path.read_text())
        c, m = identities(payload)
        if c or m:
            coco.update(c)
            mile.update(m)
            files[str(path)] = digest(path)
        references(payload)
    return dict(
        coco_excluded_ids=sorted(coco),
        milebench_excluded_ids=sorted(mile),
        contributing_files=files,
        scanned_metadata_files=len(seen),
        scan_wall_seconds=time.monotonic() - start,
        scope=(
            "project top-level/two-level plans and materialization/ID manifests, "
            "plus their transitive ledger references; output trees and scheduler logs pruned"
        ),
        outcomes_used=False,
        reserved_and_unexecuted_materialized_ids_also_excluded=True,
    )


def coco_catalog(cache: Path) -> list[dict]:
    """Project only question_id from already cached validation Arrow streams."""
    import pyarrow as pa

    if cache.name != COCO_REVISION:
        raise ValueError("wrong cached dataset revision")
    result, seen = [], set()
    for shard in range(14):
        path = cache / f"coco-caption-val-{shard:05d}-of-00014.arrow"
        with pa.memory_map(str(path), "r") as file:
            stream = pa.ipc.open_stream(file)
            index = stream.schema.get_field_index("question_id")
            if index < 0:
                raise ValueError("missing canonical COCO identity column")
            offset = 0
            for batch in stream:
                # No image/answer column is converted to Python or deserialized.
                for row, value in enumerate(batch.column(index).to_pylist()):
                    match = COCO_ID.fullmatch(value)
                    if match is None:
                        raise ValueError("ambiguous COCO ID")
                    sid = "coco_caption:" + match[1]
                    if sid in seen:
                        raise ValueError("duplicate COCO identity")
                    seen.add(sid)
                    result.append(dict(sample_id=sid, arrow_file=str(path), row=offset + row))
                offset += len(batch)
    return result


def split_id_catalog(
    catalog: list[dict], excluded: set[str], *, seed: int, calibration: int, confirmation: int
) -> dict:
    if calibration < 1 or confirmation < 1:
        raise ValueError("positive cohort sizes required")
    if len({r["sample_id"] for r in catalog}) != len(catalog):
        raise ValueError("duplicate candidate IDs")
    available = sorted(
        (r for r in catalog if r["sample_id"] not in excluded),
        key=lambda r: hashlib.sha256(f"modest-memory-v1:{seed}:{r['sample_id']}".encode()).digest(),
    )
    if len(available) < calibration + confirmation:
        raise ValueError("insufficient disjoint COCO identities")
    return dict(
        available_count=len(available),
        calibration=available[:calibration],
        confirmation_id_only_reservation=available[calibration : calibration + confirmation],
    )


def audit(args):
    if args.output.exists():
        raise FileExistsError("audit is write-once")
    protocol = yaml.safe_load(args.protocol.read_text())
    inventory = prior_identity_inventory(args.runs)
    rule = protocol["lanes"]["coco"]
    catalog = coco_catalog(args.arrow_cache.resolve(strict=True))
    selected = split_id_catalog(
        catalog,
        set(inventory["coco_excluded_ids"]),
        seed=protocol["controls"]["seed"],
        calibration=rule["calibration_samples"],
        confirmation=rule["confirmation_initial_samples"],
    )
    mile = {}
    for task in protocol["lanes"]["milebench"]["subtasks"]:
        path = args.milebench_root / task / f"{task}.json"
        payload = json.loads(path.read_text())
        ids = {f"{task}-{r['sample_id']:04d}" for r in payload["data"]}
        mile[task] = dict(
            total=len(ids),
            available_unreserved_ids=sorted(ids - set(inventory["milebench_excluded_ids"])),
            annotation=str(path),
            annotation_sha256=digest(path),
        )
    report = dict(
        schema_version=1,
        status="passed",
        protocol=str(args.protocol.resolve()),
        protocol_sha256=digest(args.protocol),
        preparer_sha256=digest(Path(__file__)),
        **inventory,
        coco_dataset_id=rule["dataset"],
        coco_dataset_revision=rule["dataset_revision"],
        coco_total_ids=len(catalog),
        **selected,
        milebench=mile,
        milebench_execution_authorized=False,
        confirmation_materialized=False,
        gpu_inference=False,
    )
    write_once(args.output, report)
    print(
        json.dumps(
            dict(
                audit=str(args.output),
                audit_sha256=digest(args.output),
                coco_total=len(catalog),
                coco_available=selected["available_count"],
                calibration=len(selected["calibration"]),
                confirmation_reserved=len(selected["confirmation_id_only_reservation"]),
                milebench_available={
                    k: len(v["available_unreserved_ids"]) for k, v in mile.items()
                },
            )
        ),
        flush=True,
    )


def clean_source(source: Path, expected: str):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != expected:
        raise ValueError("inference source SHA mismatch")
    subprocess.run(["bash", "mosaickv/scripts/assert_clean_worktree.sh"], cwd=source, check=True)
    import mosaickv

    if source not in Path(mosaickv.__file__).resolve().parents:
        raise ValueError("Python imported a different MosaicKV checkout")


def prepare(args):
    from datasets import Dataset

    from mosaickv.config import canonical_config, config_sha256, load_config
    from mosaickv.evaluation.shared_byte_contract import (
        candidate_content_digest,
        load_contract,
        validate_configuration,
    )

    protocol = yaml.safe_load(args.protocol.read_text())
    rule = protocol["lanes"]["coco"]
    source = Path(rule["source_root"]).resolve()
    clean_source(source, rule["source_sha"])
    audit_report = json.loads(args.audit.read_text())
    if (
        audit_report["protocol_sha256"] != digest(args.protocol)
        or audit_report["status"] != "passed"
    ):
        raise ValueError("audit protocol binding mismatch")
    if args.root.exists():
        raise FileExistsError("calibration output is write-once")
    if digest(Path(rule["parent_plan"])) != rule["parent_plan_sha256"]:
        raise ValueError("immutable parent plan changed")
    parent = json.loads(Path(rule["parent_plan"]).read_text())
    if (
        parent["source_sha"] != rule["source_sha"]
        or candidate_content_digest(source) != parent["candidate_content_sha256"]
    ):
        raise ValueError("frozen candidate content changed")
    contract = yaml.safe_load(Path(parent["comparison_contract"]).read_text())
    # Keep the old clean worker's stricter FOUR-worker ceiling. This is below
    # the user's six-GPU maximum, and needs no scheduler/source modification.
    contract["lanes"]["vl_cache"]["caps_mib"] = rule["caps_mib"]
    contract_path = args.root / "contract.yaml"
    contract_path.parent.mkdir(parents=True)
    write_once(args.root / "protocol.json", protocol)
    with contract_path.open("x") as stream:
        yaml.safe_dump(contract, stream, sort_keys=False)
    checked_contract = load_contract(contract_path)
    spec = importlib.util.spec_from_file_location(
        "modest_freeze", source / "mosaickv/scripts/materialize_coco_shared_bytes.py"
    )
    assert spec and spec.loader
    freezer = importlib.util.module_from_spec(spec)
    # That existing module imports sibling scripts through the normal script path.
    spec.loader.exec_module(freezer)
    datasets = {}
    samples = []
    for record in audit_report["calibration"]:
        path = record["arrow_file"]
        if path not in datasets:
            datasets[path] = Dataset.from_file(path)
        sample = freezer.freeze_sample(
            datasets[path][record["row"]], record["sample_id"], args.root
        )
        samples.append(sample)
    selected = {s["sample_id"] for s in samples}
    if (
        len(samples) != 128
        or len(selected) != 128
        or selected & set(audit_report["coco_excluded_ids"])
        or selected & {s["sample_id"] for s in audit_report["confirmation_id_only_reservation"]}
    ):
        raise ValueError("calibration disjointness/count gate failed")
    bound = {
        str((args.root / "protocol.json").resolve()): digest(args.root / "protocol.json"),
        str(args.audit.resolve()): digest(args.audit),
        str(Path(rule["parent_plan"]).resolve()): digest(Path(rule["parent_plan"])),
        str(contract_path.resolve()): digest(contract_path),
    }
    plan = dict(
        parent,
        jobs=[],
        classification="prospective_modest_memory_calibration_not_paper_confirmation",
        caps_mib=rule["caps_mib"],
        source_root=str(source),
        comparison_contract=str(contract_path.resolve()),
        comparison_contract_sha256=digest(contract_path),
        bound_files=bound,
        validation_execution_authorized=True,
        larger_execution_authorized=False,
        final_execution_authorized=False,
        heldout_claim=False,
        paper_claim_authorized=False,
        calibration_sample_count=128,
        stage_sample_counts={"validation": 128},
        max_global_a100s=4,
        user_authorized_aggregate_ceiling=6,
        prior_use_disclosures=[
            "New identities excluding all audited prior materialized/reserved cohorts"
        ],
        confirmation_materialization_authorized=False,
        calibration_protocol=str((args.root / "protocol.json").resolve()),
        audit=str(args.audit.resolve()),
        sample_ids=[s["sample_id"] for s in samples],
    )
    configs = {}
    for cap in rule["caps_mib"]:
        for method in rule["methods"]:
            if method == "full_kv" and cap != rule["caps_mib"][0]:
                continue
            original = next(
                j
                for j in parent["jobs"]
                if j["stage"] == "larger"
                and j["method"] == method
                and (method == "full_kv" or j["cap_mib"] == 40)
            )
            payload = json.loads(Path(original["config"]).read_text())
            if method != "full_kv":
                payload["cache"]["budget_value"] = cap * 2**20
            path = args.root / "configs" / f"{method}-{cap}.json"
            write_once(path, payload)
            cfg = load_config(path)
            validate_configuration(checked_contract, "vl_cache", "coco_caption", cfg)
            # JSON serializes dataclass tuple fields as arrays. Compare after
            # that lossless representation normalization, not tuple vs list.
            if json.loads(json.dumps(canonical_config(cfg))) != payload:
                raise ValueError("configuration round-trip changed the frozen controls")
            configs[method, None if method == "full_kv" else cap] = (path, config_sha256(cfg))
    for shard, offset in enumerate(range(0, 128, 16)):
        chosen = samples[offset : offset + 16]
        ledger = args.root / "ledgers" / f"{shard:03d}.json"
        write_once(
            ledger,
            dict(
                schema_version=1,
                task="coco_caption",
                dataset_id=rule["dataset"],
                dataset_revision=rule["dataset_revision"],
                split="val",
                samples=chosen,
            ),
        )
        for (method, cap), (config, sha) in configs.items():
            plan["jobs"].append(
                dict(
                    stage="validation",
                    shard=shard,
                    method=method,
                    cap_mib=cap,
                    sample_ids=[s["sample_id"] for s in chosen],
                    ledger=str(ledger.resolve()),
                    ledger_sha=digest(ledger),
                    config=str(config.resolve()),
                    config_sha=sha,
                    output=str((args.root / "outputs" / f"{shard:03d}-{method}-{cap}").resolve()),
                )
            )
    write_once(args.root / "plan.json", plan)
    print(
        json.dumps(
            dict(
                plan=str(args.root / "plan.json"),
                sha256=digest(args.root / "plan.json"),
                jobs=len(plan["jobs"]),
                samples=128,
                submitted=False,
                max_concurrent_a100s=4,
                confirmation_blocked=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--protocol", type=Path, required=True)
    a.add_argument("--runs", type=Path, required=True)
    a.add_argument("--arrow-cache", type=Path, required=True)
    a.add_argument("--milebench-root", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare-calibration")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    (audit if arguments.action == "audit" else prepare)(arguments)
