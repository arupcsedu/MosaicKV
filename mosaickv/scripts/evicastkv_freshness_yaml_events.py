#!/usr/bin/env python3
"""Bounded YAML parser-event metadata projection, never YAML object loading.

Only Reader/Scanner/Parser are instantiated. No Composer, Constructor,
Resolver, load/safe_load/BaseLoader, tags, aliases, anchors or merge keys.
Scalar events are transient lexical strings (including discarded fields), so
this deliberately does not claim JSON's before-scalar-decoding redaction.
Exports contain only existing collector identities and immutable hash proofs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import yaml
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate
from evicastkv_review_freshness_stream import FORBIDDEN, collect, safe_key, signature

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
YAML_VERSION = "6.0.3"
MAX_BYTES, MAX_EVENTS, MAX_DEPTH, SECONDS, SHARDS = 2 * 2**20, 100000, 40, 240, 8
YAML_ERROR = "YAML requires separately reviewed lexical identity projection"
IDENTITIES = (
    "sample_ids",
    "coco_ids",
    "media_hashes",
    "source_group_ids",
    "scopes",
    "scoped_identities",
)
JSON_SCALAR = re.compile(r"(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)\Z")


class UnsupportedYAML(ValueError):
    def __init__(self, reason, mark=None):
        self.reason = reason
        self.line = getattr(mark, "line", None)
        self.column = getattr(mark, "column", None)
        super().__init__(reason)  # Never include raw parser snippets or scalar content.


class DeadlineStream(io.BytesIO):
    def __init__(self, raw, deadline):
        super().__init__(raw)
        self.deadline = deadline

    def read(self, size=-1):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("YAML lexical input deadline reached")
        return super().read(min(size, 4096) if size >= 0 else 4096)


class EventParser(yaml.reader.Reader, yaml.scanner.Scanner, yaml.parser.Parser):
    def __init__(self, stream):
        yaml.reader.Reader.__init__(self, stream)
        yaml.scanner.Scanner.__init__(self)
        yaml.parser.Parser.__init__(self)


def projection(raw, *, deadline, statistics=None):
    if yaml.__version__ != YAML_VERSION or len(raw) > MAX_BYTES:
        raise UnsupportedYAML("parser_version_or_file_size_not_registered")
    parser = None
    count = 0

    def peek():
        if time.monotonic() >= deadline:
            raise TimeoutError("YAML parser-event deadline reached")
        return parser.peek_event()

    def take(expected=None):
        nonlocal count
        event = peek()
        if event is None or (expected is not None and not isinstance(event, expected)):
            raise UnsupportedYAML(
                "unexpected_event_or_complex_key", getattr(event, "start_mark", None)
            )
        event = parser.get_event()
        count += 1
        if count > MAX_EVENTS:
            raise UnsupportedYAML("event_limit_exceeded")
        if isinstance(event, yaml.events.AliasEvent) or getattr(event, "anchor", None) is not None:
            raise UnsupportedYAML("anchors_and_aliases_not_supported", event.start_mark)
        if getattr(event, "tag", None) is not None:
            raise UnsupportedYAML("explicit_tags_not_supported", event.start_mark)
        return event

    def scalar(event):
        if event.style is not None:
            return event.value
        value = event.value
        if JSON_SCALAR.fullmatch(value):
            result = json.loads(value)
            if isinstance(result, float) and not math.isfinite(result):
                raise UnsupportedYAML("nonfinite_implicit_scalar", event.start_mark)
            return result
        # Inspect resolver regex tables, never instantiate a Resolver or
        # construct tagged values. Reject implicit non-JSON YAML coercions.
        for _tag, pattern in yaml.resolver.Resolver.yaml_implicit_resolvers.get(value[:1], []):
            if pattern.match(value):
                raise UnsupportedYAML("ambiguous_non_JSON_implicit_scalar", event.start_mark)
        return value

    def node(key="", *, keep=True, depth=0):
        if depth > MAX_DEPTH:
            raise UnsupportedYAML("nesting_depth_exceeded")
        event = take()
        keep = keep and key not in FORBIDDEN
        if isinstance(event, yaml.events.ScalarEvent):
            return scalar(event) if keep and safe_key(key) else None
        if isinstance(event, yaml.events.SequenceStartEvent):
            result = [] if keep else None
            while not isinstance(peek(), yaml.events.SequenceEndEvent):
                value = node(key, keep=keep, depth=depth + 1)
                if keep:
                    result.append(value)
            take(yaml.events.SequenceEndEvent)
            return result
        if isinstance(event, yaml.events.MappingStartEvent):
            result, keys = {} if keep else None, set()
            while not isinstance(peek(), yaml.events.MappingEndEvent):
                field = take(yaml.events.ScalarEvent)
                if field.value == "<<" or field.value in keys:
                    raise UnsupportedYAML("merge_or_duplicate_key_not_supported", field.start_mark)
                keys.add(field.value)
                value = node(field.value, keep=keep, depth=depth + 1)
                if keep:
                    result[field.value] = value
            take(yaml.events.MappingEndEvent)
            return result
        raise UnsupportedYAML("unexpected_YAML_node", getattr(event, "start_mark", None))

    try:
        parser = EventParser(DeadlineStream(raw, deadline))
        take(yaml.events.StreamStartEvent)
        start = take(yaml.events.DocumentStartEvent)
        if start.tags or start.version:
            raise UnsupportedYAML("YAML_directives_not_supported", start.start_mark)
        result = node()
        take(yaml.events.DocumentEndEvent)
        take(yaml.events.StreamEndEvent)  # A second document is rejected here.
        if parser.peek_event() is not None:
            raise UnsupportedYAML("trailing_YAML_events")
        return result
    except yaml.YAMLError as error:
        raise UnsupportedYAML("malformed_YAML", getattr(error, "problem_mark", None)) from None
    finally:
        if parser is not None:
            parser.dispose()
        if statistics is not None:
            statistics.update(events=count)


def read(path, expected=None):
    return read_json(read_bytes(path, bound=128 * 2**20, expected=expected)[0])


def prepare(root, *, cpu_gate_root, parent, parent_gate_filename=None):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch YAML projection root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(cpu_gate_root, revision)
    gates = ("GATE_freshness_frontier_fork.json", "GATE_freshness_empty_parquet_frontier.json")
    candidates = [name for name in gates if (parent / name).is_file()]
    gate_name = parent_gate_filename or (candidates[0] if len(candidates) == 1 else None)
    if gate_name not in gates:
        raise ValueError("one explicitly reviewed parent gate required")
    gate_path = parent / gate_name
    gate = read(gate_path)
    if gate["missing_shards"]:
        raise ValueError("every parent state required before queue freezing")
    plan_path = parent / "plan.json"
    old = read(plan_path)
    roots = tuple(Path(p).resolve(strict=True) for p in old["roots"])
    bound = {str(gate_path): sha(gate_path), str(plan_path): sha(plan_path)}
    rows = defaultdict(set)
    workers = {
        name: expected
        for name, expected in gate["worker_bindings"].items()
        if Path(name).parent.resolve() == (parent / "workers").resolve()
    }
    if len(workers) != SHARDS:
        raise ValueError("eight exact parent state artifacts required")
    if sorted(int(Path(path).stem.removeprefix("shard-")) for path in workers) != list(
        range(SHARDS)
    ):
        raise ValueError("parent shard indices must cover every registered worker exactly once")
    for path, expected in workers.items():
        record = read(path, expected)
        if (
            record["source_sha"] != old["source_sha"]
            or digest(record["state"]) != record["state_sha256"]
        ):
            raise ValueError("parent state source/hash differs")
        bound[path] = expected
        for error in record["state"]["errors"]:
            if error.get("error") == YAML_ERROR:
                rows[error["path"]].add(digest(error))
    inventory_path = parent / "combined_identity_inventory.json"
    read(inventory_path, gate["identity_inventory_sha256"])
    bound[str(inventory_path)] = gate["identity_inventory_sha256"]
    queue = []
    if len(rows) > 50000:
        raise ValueError("registered YAML queue exceeds finite bound")
    for name, errors in sorted(rows.items()):
        path = Path(name).resolve(strict=True)
        if not any(path.is_relative_to(r) for r in roots) or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("registered YAML error escapes declared historical scope")
        queue.append(
            dict(
                path=str(path),
                original_error_sha256s=sorted(errors),
                signature=signature(path),
                index=int(hashlib.sha256(str(path).encode()).hexdigest(), 16) % SHARDS,
            )
        )
    if len({row["path"] for row in queue}) != len(queue):
        raise ValueError("canonical YAML aliases require separate reviewed deduplication")
    plan = dict(
        source_sha=revision,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        parent=str(parent),
        parent_gate_filename=gate_name,
        parent_source_sha=old["source_sha"],
        bindings=bound,
        roots=list(map(str, roots)),
        queue=queue,
        parser_version=YAML_VERSION,
        seconds_per_worker=SECONDS,
        maximum_bytes=MAX_BYTES,
        maximum_events=MAX_EVENTS,
        maximum_depth=MAX_DEPTH,
        shard_count=SHARDS,
        segment_count=1,
        no_YAML_object_construction=True,
        transient_scalar_events_not_predecode_redaction=True,
        original_errors_preserved=True,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_bounded_YAML_event_projection",
        files=len(queue),
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable YAML projection parent evidence changed")
    return plan


def project_file(row, deadline):
    path = Path(row["path"])
    if signature(path) != row["signature"]:
        raise ValueError("registered YAML signature changed")
    if row["signature"][2] > MAX_BYTES:
        raise UnsupportedYAML("file_size_limit_exceeded")
    raw, before = read_bytes(path, bound=MAX_BYTES)
    statistics = {}
    value = projection(raw, deadline=deadline, statistics=statistics)
    identities = {k: [] for k in IDENTITIES} | {"projected_records": 0}
    collect(value, identities, path)
    if time.monotonic() >= deadline:
        raise TimeoutError("YAML transaction deadline reached")
    _raw, after = read_bytes(path, bound=MAX_BYTES, expected=before)
    if signature(path) != row["signature"] or before != after:
        raise ValueError("YAML source changed across projection")
    return dict(
        path=str(path),
        identities=identities,
        whole_file_sha256_before=before,
        whole_file_sha256_after=after,
        signature=row["signature"],
        statistics=statistics,
        projected_metadata_sha256=digest(value),
        original_error_sha256s=row["original_error_sha256s"],
        old_unbound_historical_version_byte_identity_claimed=False,
    )


def worker(root, index):
    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("registered one-CPU YAML shard required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline = time.monotonic() + plan["seconds_per_worker"]
    queue = [r for r in plan["queue"] if r["index"] == index]
    committed, pending, blocked = [], [], []
    for number, row in enumerate(queue):
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("finite YAML worker deadline")
            committed.append(project_file(row, deadline))
        except TimeoutError:
            pending.extend(queue[number:])
            break
        except UnsupportedYAML as error:
            blocked.append(dict(row=row, reason=error.reason, line=error.line, column=error.column))
        except (OSError, ValueError, RecursionError, UnicodeError) as error:
            blocked.append(
                dict(row=row, reason=type(error).__name__, errno=getattr(error, "errno", None))
            )
    output = dict(
        source_sha=plan["source_sha"],
        index=index,
        committed=committed,
        pending=pending,
        blocked=blocked,
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
    )
    path = root / "workers" / f"shard-{index}.json"
    write_once(path, output)
    return dict(
        index=index,
        projected=len(committed),
        pending=len(pending),
        blocked=len(blocked),
        sha256=sha(path),
    )


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    old = read(Path(plan["parent"]) / "combined_identity_inventory.json")
    parent_gate = read(Path(plan["parent"]) / plan["parent_gate_filename"])
    merged = {
        k: set(tuple(v) if isinstance(v, list) else v for v in old["identities"][k])
        for k in IDENTITIES
    }
    missing, committed, pending, blocked, bound = [], [], [], [], {}
    expected = {r["path"]: r for r in plan["queue"]}
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        result = read(path)
        if result["source_sha"] != plan["source_sha"] or result["index"] != index:
            raise ValueError("YAML worker source/index differs")
        bound[str(path)] = sha(path)
        classified = (
            [r["path"] for r in result["committed"]]
            + [r["path"] for r in result["pending"]]
            + [r["row"]["path"] for r in result["blocked"]]
        )
        if len(classified) != len(set(classified)) or set(classified) != {
            r["path"] for r in plan["queue"] if r["index"] == index
        }:
            raise ValueError("registered YAML files dropped or multiply classified")
        for row in result["committed"]:
            original = expected[row["path"]]
            if (
                row["signature"] != original["signature"]
                or row["original_error_sha256s"] != original["original_error_sha256s"]
                or row["whole_file_sha256_before"] != row["whole_file_sha256_after"]
            ):
                raise ValueError("YAML immutable transaction proof differs")
            for key in IDENTITIES:
                merged[key].update(
                    tuple(v) if isinstance(v, list) else v for v in row["identities"][key]
                )
        if any(row != expected[row["path"]] for row in result["pending"]) or any(
            row["row"] != expected[row["row"]["path"]] for row in result["blocked"]
        ):
            raise ValueError("unresolved YAML queue differs from its immutable registration")
        committed.extend(result["committed"])
        pending.extend(result["pending"])
        blocked.extend(result["blocked"])
    write_once(
        root / "combined_identity_inventory.json",
        dict(
            identities={k: sorted(v) for k, v in merged.items()},
            source_sha=plan["source_sha"],
            worker_bindings=bound,
            complete_historical_id_inventory=False,
            cross_dataset_media_certified=False,
            fresh_confirmation_authorized=False,
        ),
    )
    report = dict(
        status="bounded_YAML_event_projection_complete"
        if not (missing or pending or blocked)
        else "bounded_YAML_event_projection_incomplete",
        source_sha=plan["source_sha"],
        registered_files=len(expected),
        projected_files=len(committed),
        pending_files=len(pending),
        blocked_files=len(blocked),
        missing_worker_files=sum(row["index"] in missing for row in plan["queue"]),
        missing_shards=missing,
        frontier_exhausted=parent_gate["frontier_exhausted"],
        original_and_new_coverage_errors=parent_gate["original_and_new_coverage_errors"],
        original_errors_preserved=True,
        error_waivers=0,
        worker_bindings=bound,
        identity_inventory_sha256=sha(root / "combined_identity_inventory.json"),
        no_YAML_object_construction=True,
        transient_scalar_events_not_predecode_redaction=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "GATE_freshness_yaml_events.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    result = (
        prepare(args.root, cpu_gate_root=args.cpu_gate_root, parent=args.parent)
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    print(json.dumps(result))
