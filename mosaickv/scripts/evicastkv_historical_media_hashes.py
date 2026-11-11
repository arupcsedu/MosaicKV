#!/usr/bin/env python3
"""Finite provider-bound historical image hashes, never a global certificate.

Only reviewed JSON ledgers and one typed media manifest are inputs. No root
walk, dataset/model loading, image copies, labels, or automatic retries. IDs
without media mappings and missing/conflicting assets remain explicit gaps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from evicastkv_coco_media_availability import MEDIA, MEDIA_SHA, image_identity, metadata
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate
from evicastkv_review_freshness_stream import signature

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
RECOVERY = Path("/scratch/djy8hg/workdir/fox-review-v11/run-20261110-1/freshness")
COCO = (
    (
        "/scratch/djy8hg/runs/mosaickv/coco-shared-pool-edb6edf-retry1-20261014/ledger.json",
        "bff5f9972e2b4721d6d07c86d061dc4083a5cabb54c828a1d638457b589f02b9",
    ),
    (
        "/scratch/djy8hg/runs/evicastkv/paper-quality-coco-20261106-guard-recovery1/ledger.json",
        "7ea86125cca99bf06fe690499326095a9fb8808ce3bb05d1c3da1a7ec58c683b",
    ),
)
ROOTS = tuple(
    Path(f"/scratch/djy8hg/{p}").resolve()
    for p in ("runs/mosaickv", "runs/evicastkv", "data/mosaickv", "cache/mosaickv")
)
HASH = re.compile(r"[0-9a-f]{64}")
SHARDS, SECONDS, MAX_IMAGE = 8, 240, 32 * 2**20


def read(path, expected=None):
    return read_json(read_bytes(path, bound=128 * 2**20, expected=expected)[0])


def recovered_ledgers(root):
    gate_path, plan_path = (
        Path(root) / "GATE_freshness_recipe_recovery.json",
        Path(root) / "plan.json",
    )
    gate = read(gate_path)
    plan = read(plan_path, gate["plan_sha256"])
    if gate["status"] != "completed_exact_config_recovery_coverage_overlay":
        raise ValueError("successful bound producer recovery required")
    matches = [
        (Path(name), expected)
        for name, expected in plan["bindings"].items()
        if Path(name).name == "matrix_manifest.json"
    ]
    if len(matches) != 1:
        raise ValueError("one exact recovered producer matrix required")
    matrix = read(*matches[0])
    if set(matrix["sample_ledgers"]) != {"chartqa", "docvqa", "mmstar", "textvqa", "videomme"}:
        raise ValueError("five registered parent ledger routes differ")
    bindings = {
        str(gate_path): sha(gate_path),
        str(plan_path): sha(plan_path),
        str(matches[0][0]): matches[0][1],
    }
    return [(row["path"], row["sha256"]) for row in matrix["sample_ledgers"].values()], bindings


def extract(obj, source, source_sha):
    """Only recognized sample-media pairs; generic artifact paths never pair."""
    assets, unresolved = [], []
    samples = obj.get("samples")
    if not isinstance(samples, list):
        unresolved.append(
            dict(
                provider=str(source),
                reason="ID_only_or_unrecognized_media_schema",
                identity_references=len(obj.get("sample_ids", [])),
                identity_digest=digest(obj.get("sample_ids", [])),
            )
        )
        return assets, unresolved
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            unresolved.append(
                dict(
                    provider=str(source),
                    sample_index=index,
                    reason="unrecognized_sample_media_schema",
                )
            )
            continue
        sid = sample.get("sample_id")
        media = sample.get("media")
        if not isinstance(sid, str) or not isinstance(media, list) or not media:
            unresolved.append(
                dict(
                    provider=str(source),
                    sample_index=index,
                    reason="sample_identity_or_media_mapping_missing",
                )
            )
            continue
        for number, medium in enumerate(media):
            if not isinstance(medium, dict):
                unresolved.append(
                    dict(
                        provider=str(source),
                        sample_id=sid,
                        media_index=number,
                        reason="unrecognized_media_entry_schema",
                    )
                )
                continue
            path, expected = medium.get("path"), medium.get("sha256")
            ref = dict(
                provider=str(source),
                provider_sha256=source_sha,
                sample_id=sid,
                media_index=number,
                original_media_path=path,
            )
            if (
                not isinstance(path, str)
                or not isinstance(expected, str)
                or not HASH.fullmatch(expected)
            ):
                unresolved.append(
                    dict(reference=ref, reason="recognized_media_path_or_SHA_missing")
                )
                continue
            asset = Path(path)
            if not asset.is_absolute():
                asset = Path(source).parent / asset
            assets.append(dict(path=str(asset), recorded_sha256=expected, reference=ref))
    return assets, unresolved


def prepare(
    root,
    *,
    cpu_gate_root,
    recovery_root=RECOVERY,
    coco_ledgers=COCO,
    typed_media=MEDIA,
    typed_media_sha256=MEDIA_SHA,
    allowed_roots=ROOTS,
):
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch historical-media root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(cpu_gate_root, revision)
    extra, bindings = recovered_ledgers(recovery_root)
    raw, unresolved, providers = [], [], []
    for name, expected in [*coco_ledgers, *extra]:
        source = Path(name).resolve(strict=True)
        obj, actual = metadata(source, expected)
        bindings[str(source)] = actual
        found, gaps = extract(obj, source, actual)
        raw.extend(found)
        unresolved.extend(gaps)
        providers.append(
            dict(
                path=str(source),
                sha256=actual,
                recognized_media_refs=len(found),
                unresolved_entries=len(gaps),
                dataset_id=obj.get("dataset_id"),
                dataset_revision=obj.get("dataset_revision"),
                split=obj.get("split"),
            )
        )
    typed, actual = metadata(typed_media, typed_media_sha256)
    bindings[str(typed_media)] = actual
    prior = {}
    for name, record in typed["identities"].items():
        encoded, pixel = record.get("encoded_sha256"), record.get("pixel_sha256")
        if (
            not isinstance(encoded, str)
            or not HASH.fullmatch(encoded)
            or not isinstance(pixel, str)
            or not HASH.fullmatch(pixel)
        ):
            raise ValueError("existing typed encoded/pixel hash pair malformed")
        if encoded in prior and prior[encoded]["pixel_sha256"] != pixel:
            raise ValueError("conflicting typed pixel identity for one encoded body")
        prior.setdefault(
            encoded,
            dict(
                asset_key=f"encoded:{encoded}",
                encoded_sha256=encoded,
                pixel_sha256=pixel,
                existing_typed_refs=[],
            ),
        )["existing_typed_refs"].append(name)
    allowed = tuple(Path(p).resolve(strict=True) for p in allowed_roots)
    by_path = {}
    conflicts = set()
    for row in raw:
        try:
            path = Path(row["path"]).resolve(strict=True)
            if not any(path.is_relative_to(r) for r in allowed):
                raise ValueError("outside_allowed_roots")
            row = dict(row, path=str(path))
            old = by_path.setdefault(
                str(path),
                dict(
                    path=str(path),
                    recorded_sha256=row["recorded_sha256"],
                    recorded_SHA256_claims=[row["recorded_sha256"]],
                    signature=signature(path),
                    references=[],
                ),
            )
            if old["recorded_sha256"] != row["recorded_sha256"]:
                conflicts.add(str(path))
                old["recorded_SHA256_claims"].append(row["recorded_sha256"])
            old["references"].append(row["reference"])
        except (OSError, ValueError) as error:
            unresolved.append(
                dict(
                    reference=row["reference"],
                    path=row["path"],
                    reason="unavailable_or_outside_allowed_media_roots",
                    error_type=type(error).__name__,
                )
            )
    groups = defaultdict(list)
    for path, record in sorted(by_path.items()):
        if path in conflicts:
            unresolved.append(
                dict(
                    path=path,
                    references=record["references"],
                    recorded_SHA256_claims=sorted(set(record["recorded_SHA256_claims"])),
                    reason="conflicting_recorded_SHA_for_canonical_path",
                )
            )
            continue
        record["references"] = sorted(
            {digest(r): r for r in record["references"]}.values(), key=digest
        )
        groups[record["recorded_sha256"]].append(record)
    queue = [
        dict(encoded_sha256=key, aliases=rows, index=int(key, 16) % SHARDS)
        for key, rows in sorted(groups.items())
    ]
    plan = dict(
        source_sha=revision,
        measurement_type="historical_media_identity_audit",
        libraries={"Pillow": version("Pillow")},
        common_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
        models_loaded=False,
        gpu_count=0,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        bindings=bindings,
        providers=providers,
        unresolved=unresolved,
        queue=queue,
        existing_typed_assets=prior,
        allowed_roots=list(map(str, allowed)),
        seconds_per_worker=SECONDS,
        maximum_image_bytes=MAX_IMAGE,
        shard_count=SHARDS,
        segment_count=1,
        finite_provider_set_only=True,
        global_historical_media_inventory_complete=False,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_finite_historical_media_providers",
        recognized_media_refs=len(raw),
        encoded_groups=len(queue),
        unresolved_provider_entries=len(unresolved),
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable historical-media provider changed")
    return plan


def project_group(group, plan, deadline):
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("finite historical image transaction deadline")

    def source_guard(alias):
        path = Path(alias["path"])
        canonical = path.resolve(strict=True)
        if canonical != path or not any(
            canonical.is_relative_to(Path(r)) for r in plan["allowed_roots"]
        ):
            raise ValueError("registered canonical media path now aliases outside its frozen scope")
        if signature(path) != alias["signature"]:
            raise ValueError("registered media signature changed")
        return path

    raw_first = None
    proofs = []
    for alias in group["aliases"]:
        check()
        path = source_guard(alias)
        raw, before = read_bytes(path, bound=MAX_IMAGE, expected=group["encoded_sha256"])
        if raw_first is None:
            raw_first = raw
        proofs.append(
            dict(path=str(path), encoded_sha256_before=before, signature=alias["signature"])
        )
    check()
    prior = plan["existing_typed_assets"].get(group["encoded_sha256"])
    pixel = (
        dict(encoded_sha256=prior["encoded_sha256"], pixel_sha256=prior["pixel_sha256"])
        if prior
        else image_identity(raw_first)
    )
    if pixel["encoded_sha256"] != group["encoded_sha256"] or not HASH.fullmatch(
        pixel["pixel_sha256"]
    ):
        raise ValueError("constructed pixel identity does not bind the verified encoded body")
    check()
    for alias, proof in zip(group["aliases"], proofs, strict=True):
        check()
        path = source_guard(alias)
        _raw, after = read_bytes(path, bound=MAX_IMAGE, expected=group["encoded_sha256"])
        if signature(path) != alias["signature"] or after != proof["encoded_sha256_before"]:
            raise ValueError("media source changed across pixel hashing")
        source_guard(alias)
        proof["encoded_sha256_after"] = after
    return dict(
        asset_key=f"encoded:{group['encoded_sha256']}",
        **pixel,
        alias_proofs=proofs,
        references=[r for alias in group["aliases"] for r in alias["references"]],
        pixel_method="reused_bound_typed_encoded_body"
        if prior
        else "EXIF_transposed_RGB_shape_prefixed_SHA256",
        no_image_copies_or_labels_exported=True,
    )


def worker(root, index):
    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("registered one-CPU image shard required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline = time.monotonic() + plan["seconds_per_worker"]
    groups = [g for g in plan["queue"] if g["index"] == index]
    committed, pending, blocked = [], [], []
    for number, group in enumerate(groups):
        try:
            committed.append(project_group(group, plan, deadline))
        except TimeoutError:
            pending.extend(groups[number:])
            break
        except (OSError, ValueError, TypeError, KeyError) as error:
            blocked.append(
                dict(group=group, reason=type(error).__name__, errno=getattr(error, "errno", None))
            )
    path = root / "workers" / f"shard-{index}.json"
    write_once(
        path,
        dict(
            source_sha=plan["source_sha"],
            index=index,
            committed=committed,
            pending=pending,
            blocked=blocked,
            fresh_confirmation_authorized=False,
        ),
    )
    return dict(
        index=index,
        hashed=len(committed),
        pending=len(pending),
        blocked=len(blocked),
        sha256=sha(path),
    )


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    assets = {key: dict(value) for key, value in plan["existing_typed_assets"].items()}
    missing, pending, blocked, bound = [], [], [], {}
    expected = {g["encoded_sha256"]: g for g in plan["queue"]}
    committed_count = 0
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        result = read(path)
        if result["source_sha"] != plan["source_sha"] or result["index"] != index:
            raise ValueError("image worker source/index differs")
        bound[str(path)] = sha(path)
        keys = (
            [r["encoded_sha256"] for r in result["committed"]]
            + [r["encoded_sha256"] for r in result["pending"]]
            + [r["group"]["encoded_sha256"] for r in result["blocked"]]
        )
        if len(keys) != len(set(keys)) or set(keys) != {
            key for key, g in expected.items() if g["index"] == index
        }:
            raise ValueError("registered image groups dropped or multiply classified")
        for record in result["committed"]:
            key = record["encoded_sha256"]
            aliases = expected[key]["aliases"]
            if len(record["alias_proofs"]) != len(aliases) or any(
                p["path"] != a["path"]
                or p["signature"] != a["signature"]
                or p["encoded_sha256_before"] != key
                or p["encoded_sha256_after"] != key
                for p, a in zip(record["alias_proofs"], aliases, strict=True)
            ):
                raise ValueError("encoded image transaction proof differs")
            if key in assets and assets[key]["pixel_sha256"] != record["pixel_sha256"]:
                raise ValueError("typed pixel identity conflicts across verified providers")
            previous = assets.get(key, {})
            assets[key] = dict(record, existing_typed_refs=previous.get("existing_typed_refs", []))
            committed_count += 1
        if any(group != expected[group["encoded_sha256"]] for group in result["pending"]) or any(
            row["group"] != expected[row["group"]["encoded_sha256"]] for row in result["blocked"]
        ):
            raise ValueError("unresolved image group differs from its registered source aliases")
        pending.extend(result["pending"])
        blocked.extend(result["blocked"])
    absent = [group for group in plan["queue"] if group["index"] in missing]
    unresolved = [
        *plan["unresolved"],
        *blocked,
        *(dict(reason="finite_worker_budget_pending", group=g) for g in pending),
        *(dict(reason="missing_worker", group=g) for g in absent),
    ]
    output = dict(
        source_sha=plan["source_sha"],
        assets=list(assets.values()),
        providers=plan["providers"],
        unresolved=unresolved,
        worker_bindings=bound,
        finite_provider_set_only=True,
        global_historical_media_inventory_complete=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "typed_asset_inventory.json", output)
    report = dict(
        status="finite_historical_media_audit_complete"
        if not (missing or pending or unresolved)
        else "finite_historical_media_audit_incomplete",
        source_sha=plan["source_sha"],
        existing_typed_encoded_assets=len(plan["existing_typed_assets"]),
        registered_encoded_groups=len(expected),
        verified_encoded_groups=committed_count,
        typed_encoded_assets=len(assets),
        pending_groups=len(pending),
        blocked_groups=len(blocked),
        unresolved_provider_entries=len(plan["unresolved"]),
        missing_shards=missing,
        missing_worker_groups=len(absent),
        worker_bindings=bound,
        typed_inventory_sha256=sha(root / "typed_asset_inventory.json"),
        finite_provider_set_only=True,
        global_historical_media_inventory_complete=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "GATE_historical_media_hashes.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--recovery-root", type=Path, default=RECOVERY)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    result = (
        prepare(args.root, cpu_gate_root=args.cpu_gate_root, recovery_root=args.recovery_root)
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    print(json.dumps(result))
