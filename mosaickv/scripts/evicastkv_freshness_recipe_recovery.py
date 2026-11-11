#!/usr/bin/env python3
"""Recover canonical configs by exact digests, never load historical YAML.

The absent historical producer and the independently recovered public snapshot
remain distinct. This exports configuration provenance, not model results or a
fresh dataset. No network access or source checkout occurs at canonical run.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

from evicastkv_freshness_producer_canary import (
    DEFAULT_CAMPAIGN,
    DEFAULT_INVENTORY,
    HISTORY,
    WORKSPACE,
    YAML_GAP,
    hash_bytes,
    read_bytes,
    read_json,
)
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
SNAPSHOT = Path("/scratch/djy8hg/workdir/fox-v11-writer-snapshot-4e5d3a").resolve()
SNAPSHOT_SHA = "4e5d3a61d832e771e2e53ce60d1ce17adef8e38a"
ORIGINAL_PRODUCER = "1ed3ae456068a30e068d1dccf398f38179148def"
BUNDLE = Path("/scratch/djy8hg/workdir/fox-v11-recovered-writer-4e5d3a.tar")
BUNDLE_SHA = "b66c9daa3d9295c46cc1327937684b926fa98858dcb687ed2793fd5d13f74a80"
MATRIX_RELATIVE = "mosaickv/configs/experiments/main_quality.yaml"
MATRIX_SHA = "8d54284bb7187288ce1735a86f3d537d6994787fc0852e031ecb6936e31c33ac"
INVENTORY_SHA = "15d9f114918224312c6ec53537305732daaa6208390cca9f4076f5459eb77005"
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"

# Reviewed pure-enumeration child, with no script-directory shadowing and no
# writes/network/subprocesses. Only the snapshot's tracked matrix is YAML-loaded.
CHILD_CODE = r"""
import sys, os, json, hashlib, dataclasses
from pathlib import Path
def guard(event, args):
    if event in {'subprocess.Popen','socket.connect','socket.getaddrinfo'}:
        raise RuntimeError('configuration reconstruction cannot launch processes or network')
    if event == 'open' and len(args) > 2 and isinstance(args[2], int):
        if args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            raise RuntimeError('configuration reconstruction child cannot write files')
sys.addaudithook(guard)
from mosaickv import config as schema
from mosaickv.experiment_matrix import load_experiment_matrix, expand_experiment_matrix
p=json.loads(Path(sys.argv[1]).read_text())
snapshot=Path(p['snapshot']).resolve()
if not Path(schema.__file__).resolve().is_relative_to(snapshot):
    raise RuntimeError('wrong reconstruction source import')
matrix_path=snapshot/p['matrix_relative']
if hashlib.sha256(matrix_path.read_bytes()).hexdigest()!=p['matrix_sha256']:
    raise RuntimeError('tracked reconstruction matrix changed')
ledgers={}
for task, binding in p['matrix']['sample_ledgers'].items():
    raw=Path(binding['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=binding['sha256']:
        raise RuntimeError('bound ID-only ledger changed')
    ledgers[task]=json.loads(raw)
matrix=load_experiment_matrix(matrix_path)
if matrix.sha256!=p['matrix']['matrix_sha']:
    raise RuntimeError('resolved matrix fingerprint differs')
payloads={}
for run in expand_experiment_matrix(matrix):
    ids=ledgers[run.task_key]['sample_ids']
    if p['matrix']['sample_limit'] is not None:
        ids=ids[:p['matrix']['sample_limit']]
    cfg=dataclasses.replace(run.config,diagnostics=schema.FidelityDiagnosticsConfig(
        enabled=True,sample_ids=tuple(ids[:min(p['matrix']['diagnostic_size'],len(ids))]),
        sample_ledger_sha=p['matrix']['sample_ledgers'][run.task_key]['sha256']))
    canonical=schema.canonical_config_json(cfg)
    key=hashlib.sha256(canonical.encode()).hexdigest()
    if key in payloads: raise RuntimeError('duplicate reconstructed canonical payload')
    payloads[key]=json.loads(canonical)
if set(payloads)!=set(p['expected_config_shas']):
    raise RuntimeError('reconstruction does not exactly cover indexed config digests')
print(json.dumps(dict(snapshot=p['snapshot_sha'],payloads=payloads),sort_keys=True))
"""


def verify_source_bundle(bundle=BUNDLE, snapshot=SNAPSHOT):
    raw, actual = read_bytes(bundle, bound=32 * 2**20, expected=BUNDLE_SHA)
    bindings = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            if member.isdir():
                continue
            parts = Path(member.name).parts
            if (
                not member.isfile()
                or not parts
                or parts[0] != "writer-4e5d3a"
                or ".." in parts
                or member.size > 8 * 2**20
                or len(bindings) >= 1024
            ):
                raise ValueError("source bundle contains an unsupported member")
            path = snapshot.joinpath(*parts[1:]).resolve(strict=True)
            if not path.is_relative_to(snapshot):
                raise ValueError("source bundle member escapes snapshot")
            stream = archive.extractfile(member)
            expected = hash_bytes(stream.read())
            if sha(path) != expected:
                raise ValueError("reconstruction snapshot differs from immutable source bundle")
            bindings[str(path)] = expected
    if str(snapshot / MATRIX_RELATIVE) not in bindings:
        raise ValueError("bundle omits the tracked reconstruction matrix")
    return dict(bundle_path=str(bundle), bundle_sha256=actual, source_files=bindings)


def inputs(parent, inventory_path, inventory_sha256):
    parent = Path(parent).resolve(strict=True)
    matrix_path = parent / "matrix_manifest.json"
    matrix_raw, matrix_sha = read_bytes(matrix_path)
    matrix = read_json(matrix_raw)
    if matrix["source_git_sha"] != ORIGINAL_PRODUCER:
        raise ValueError("historical producer metadata differs")
    raw, index_sha = read_bytes(
        parent / "jobs.jsonl", bound=32 * 2**20, expected=matrix["index_sha256"]
    )
    lines = raw.splitlines()
    if len(lines) != matrix["job_count"] or not 1 <= len(lines) <= 10000:
        raise ValueError("campaign index count differs or exceeds finite bound")
    rows = [read_json(line) for line in lines]
    inventory_raw, inventory_sha = read_bytes(
        inventory_path, bound=64 * 2**20, expected=inventory_sha256
    )
    inventory = read_json(inventory_raw)
    gaps = {row["path"]: row for row in inventory["errors"] if row.get("error") == YAML_GAP}
    if any(row["config_path"] not in gaps for row in rows):
        raise ValueError("indexed config is not in the bound original YAML-error inventory")
    return (
        matrix,
        rows,
        gaps,
        dict(
            original_error_count=len(inventory["errors"]),
            original_errors_sha256=digest(inventory["errors"]),
            bindings={
                str(matrix_path): matrix_sha,
                str(parent / "jobs.jsonl"): index_sha,
                str(inventory_path): inventory_sha,
            },
        ),
    )


def enumerate_payloads(root, matrix, rows, snapshot):
    child_input = dict(
        snapshot=str(snapshot),
        snapshot_sha=SNAPSHOT_SHA,
        matrix=matrix,
        matrix_relative=MATRIX_RELATIVE,
        matrix_sha256=MATRIX_SHA,
        expected_config_shas=sorted({row["config_sha"] for row in rows}),
    )
    path = root / "reconstruction-input.json"
    write_once(path, child_input)
    env = dict(
        os.environ,
        PYTHONPATH=f"{snapshot}/mosaickv/src:{snapshot}/mosaickv/scripts",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    result = subprocess.run(
        [PYTHON, "-B", "-c", CHILD_CODE, str(path)],
        cwd=snapshot,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    write_once(root / "reconstruction-stderr.json", result.stderr)
    if result.returncode:
        raise ValueError("pure reconstruction child failed; see recorded stderr")
    reconstructed = read_json(result.stdout)
    if reconstructed["snapshot"] != SNAPSHOT_SHA:
        raise ValueError("child used a different reconstruction snapshot")
    return reconstructed["payloads"]


def verify_rows(parent, matrix, rows, payloads, gaps):
    import yaml

    if yaml.__version__ != "6.0.3" or set(payloads) != {row["config_sha"] for row in rows}:
        raise ValueError("serializer or reconstructed config scope differs")
    rendered = {}
    for key, payload in payloads.items():
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
        if hash_bytes(canonical) != key:
            raise ValueError("reconstructed canonical JSON digest differs")
        rendered[key] = yaml.safe_dump(payload, sort_keys=True).encode()
    ledgers, shards, proofs = {}, {}, []
    for task, binding in matrix["sample_ledgers"].items():
        raw, actual = read_bytes(binding["path"], expected=binding["sha256"])
        ledger = read_json(raw)
        ids = ledger["sample_ids"]
        if len(ids) != ledger["sample_count"] or len(ids) != len(set(ids)):
            raise ValueError("ID-only parent ledger count/uniqueness differs")
        if not all(isinstance(value, str) for value in ids):
            raise ValueError("parent ledger contains noncanonical identities")
        ledgers[task] = (ledger, actual)
    seen = set()
    for index, row in enumerate(rows):
        key, task, shard_index = row["config_sha"], row["task_key"], row["shard_index"]
        path = Path(row["config_path"]).resolve(strict=True)
        expected_path = parent / "configs" / f"{index:06d}-{key}-{shard_index:05d}.yaml"
        if row["array_index"] != index or path != expected_path or path in seen:
            raise ValueError("indexed config ordering/path/uniqueness differs")
        seen.add(path)
        raw, actual_yaml_sha = read_bytes(path)
        if raw != rendered[key]:
            raise ValueError(f"historical YAML byte mismatch at index {index}; never parsed")
        ledger, ledger_sha = ledgers[task]
        if (row["full_sample_ledger_path"], row["full_sample_ledger_sha"]) != (
            matrix["sample_ledgers"][task]["path"],
            ledger_sha,
        ):
            raise ValueError("index does not bind the exact parent ledger")
        shard_path = Path(row["sample_id_file"]).resolve(strict=True)
        if shard_path != parent / "sample_shards" / task / f"{shard_index:05d}.json":
            raise ValueError("ID-only shard redirects from the indexed producer")
        if shard_path not in shards:
            raw, shard_sha = read_bytes(shard_path, expected=row["sample_id_file_sha"])
            shards[shard_path] = (read_json(raw), shard_sha)
        shard, shard_sha = shards[shard_path]
        ids = ledger["sample_ids"]
        ids = ids[: matrix["sample_limit"]] if matrix["sample_limit"] is not None else ids
        size = matrix["shard_size"]
        if type(size) is not int or size < 1 or not 0 <= shard_index < shard["shard_count"]:
            raise ValueError("invalid registered ID-only shard geometry")
        if (
            row["sample_id_file_sha"] != shard_sha
            or shard["parent_ledger_sha"] != ledger_sha
            or shard["shard_index"] != shard_index
            or row["shard_count"] != shard["shard_count"]
            or shard["shard_count"] != (len(ids) + size - 1) // size
            or shard["sample_ids"] != ids[shard_index * size : (shard_index + 1) * size]
        ):
            raise ValueError("ID-only shard is not the exact immutable ledger slice")
        for field in ("dataset_id", "dataset_revision", "split", "task"):
            if shard[field] != ledger[field]:
                raise ValueError("shard/parent dataset provenance differs")
        payload = payloads[key]
        dataset = dict(
            id=ledger["dataset_id"], revision=ledger["dataset_revision"], split=ledger["split"]
        )
        if payload["dataset"] != dataset or row["task"] != ledger["task"]:
            raise ValueError("reconstructed config dataset differs from indexed ledger")
        diagnostic_ids = ids[: min(matrix["diagnostic_size"], len(ids))]
        diagnostics = payload["diagnostics"]
        if (
            diagnostics["sample_ids"] != diagnostic_ids
            or diagnostics["sample_ledger_sha"] != ledger_sha
            or row["diagnostic_sample_count"] != len(diagnostic_ids)
        ):
            raise ValueError("reconstructed diagnostic identities differ from parent ledger")
        proofs.append(
            dict(
                index=index,
                config_path=str(path),
                canonical_json_sha256=key,
                yaml_sha256=actual_yaml_sha,
                parent_ledger_sha256=ledger_sha,
                sample_shard_sha256=shard_sha,
                original_error_sha256=digest(gaps[row["config_path"]]),
                byte_identical=True,
                historical_yaml_parsed=False,
                error_waived=False,
            )
        )
    return proofs, len(ledgers), len(shards)


def run(
    root,
    *,
    cpu_gate_root,
    parent=DEFAULT_CAMPAIGN,
    inventory_path=DEFAULT_INVENTORY,
    inventory_sha256=INVENTORY_SHA,
    snapshot=SNAPSHOT,
    bundle=BUNDLE,
):
    root, parent, snapshot = Path(root).resolve(), Path(parent).resolve(), Path(snapshot).resolve()
    if root.exists() or not root.is_relative_to(WORKSPACE) or not parent.is_relative_to(HISTORY):
        raise ValueError("new immutable workspace output and registered historical parent required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    clean_source(snapshot, SNAPSHOT_SHA)
    source_bundle = verify_source_bundle(bundle, snapshot)
    matrix, rows, gaps, provenance = inputs(parent, inventory_path, inventory_sha256)
    plan = dict(
        source_sha=revision,
        cpu_gate=cpu,
        reconstruction_snapshot=SNAPSHOT_SHA,
        historical_producer=ORIGINAL_PRODUCER,
        original_producer_object_recovered=False,
        reconstruction_repository="https://github.com/arupcsedu/MosaicKV.git",
        no_network_at_canonical_execution=True,
        source_bundle=source_bundle,
        **provenance,
    )
    write_once(root / "plan.json", plan)
    try:
        payloads = enumerate_payloads(root, matrix, rows, snapshot)
        proofs, ledgers, shards = verify_rows(parent, matrix, rows, payloads, gaps)
        clean_source(snapshot, SNAPSHOT_SHA)
        verify_source_bundle(bundle, snapshot)
        write_once(root / "canonical_configs.json", payloads)
        write_once(root / "indexed_coverage_overlay.json", proofs)
        outcome = dict(
            status="completed_exact_config_recovery_coverage_overlay",
            canonical_configs=len(payloads),
            indexed_configs=len(proofs),
            verified_parent_ledgers=ledgers,
            verified_ID_shards=shards,
            canonical_configs_sha256=sha(root / "canonical_configs.json"),
            coverage_overlay_sha256=sha(root / "indexed_coverage_overlay.json"),
        )
    except Exception as exc:
        outcome = dict(
            status="blocked_exact_config_recovery",
            error=f"{type(exc).__name__}: {exc}",
            scope_all_or_nothing=True,
            unresolved_indexed_config_count=len(rows),
        )
    gate = dict(
        **outcome,
        source_sha=revision,
        plan_sha256=sha(root / "plan.json"),
        reconstruction_snapshot=SNAPSHOT_SHA,
        historical_producer=ORIGINAL_PRODUCER,
        historical_yaml_parsed=False,
        original_errors_preserved=True,
        error_waivers=0,
        original_error_count=provenance["original_error_count"],
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        models_loaded=False,
        source_configuration_export_not_dataset_materialization=True,
        no_new_historical_tree_scan=True,
    )
    write_once(root / "GATE_freshness_recipe_recovery.json", gate)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, cpu_gate_root=args.cpu_gate_root)))
