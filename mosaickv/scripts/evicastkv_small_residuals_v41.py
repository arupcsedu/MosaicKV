#!/usr/bin/env python3
"""Bounded additive identity proofs, never freshness or historical quality claims.

YAML is interpreted as an inert parser-event DAG, without YAML constructors.
Credential-named files are never opened. Empty-file hashes follow from stable
regular-file size zero; a nonempty credential fixture deliberately stays blocked.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import evicastkv_freshness_yaml_events as lexical
import evicastkv_review_freshness_stream as scanner
import yaml
from evicastkv_freshness_producer_canary import read_json

SOURCE = Path(__file__).resolve().parents[2]
HISTORY = Path("/scratch/djy8hg/runs/mosaickv").resolve()
MAX_BYTES, MAX_EVENTS, MAX_DEPTH, MAX_EXPANDED = 2 * 2**20, 100000, 40, 100000
NONFINITE = re.compile(r"(?:NaN|-?Infinity)(?![A-Za-z0-9_])")
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
CALLBACK = HISTORY / "bounded-readout-v8-2-20261026-v1"
FIXTURE = CALLBACK / "clean-callback-verification-v1"
EMPTY = frozenset(
    {
        HISTORY / "hf-adapter-tiny-local-20260726.json",
        CALLBACK / "callback-retry-v1/codex/runtime/auth.json",
        FIXTURE / "codex/runtime/auth.json",
        FIXTURE / "pytest-tmp/test_real_runtime_overlay_writ0/callback/codex/runtime/auth.json",
    }
)
CACHE_FIXTURE = FIXTURE / (
    "pytest-tmp/test_real_runtime_overlay_writ0/callback/codex/runtime/cache-fixture.json"
)
NAN_SUMMARIES = {
    HISTORY / "choice-c-v38-bottlenecks-a100-v1/v38_analysis.json": (
        "233dcee945881d5eb44b084d87d88d2652abe3ebaca31098fdc7bff578339df7",
        4,
    ),
    HISTORY / "choice-c-v39-bottlenecks-a100-v1/v38_analysis.json": (
        "6bd56ffbab5f4777df03c907ede5327297cd1ecffbc9c25701080339cac439ea",
        2,
    ),
}
V6_ORIGINAL = HISTORY / "bounded-history-memory-v6-20261025-v1/analysis/report.json"
V6_DERIVED = HISTORY / "bounded-history-v6-derived-analysis-20261025-v1"
V6_SHA = "3eb8dd6c0c0d84bfe01856447e3fb1ee86f576eccebdc4e5914320679d987112"
V6_PROVENANCE_SHA = "991d6fc0f3288e89af51d99e6d05615fb2e25927ad9b9b000118bf2d2a188726"
V6_REPORT_SHA = "8a258d75548aa9c744e224442526241cfcec91592a4644777894c9142b543896"


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("bounded residual proof deadline")


def signature(path):
    value = Path(path).lstat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def canonical(path, roots):
    target = Path(path).resolve(strict=True)
    if not any(target.is_relative_to(Path(root).resolve(strict=True)) for root in roots):
        raise ValueError("residual path outside registered historical roots")
    if not stat.S_ISREG(target.lstat().st_mode):
        raise ValueError("regular historical file required")
    return target


def sensitive(path):
    return ".codex" in path.parts or path.name.lower() in {"auth.json", "credentials.json"}


def read(path, roots, deadline, bindings, expected=None):
    check(deadline)
    if sensitive(Path(path)):
        raise ValueError("credential body reads forbidden")
    path = canonical(path, roots)
    if sensitive(path):
        raise ValueError("credential body reads forbidden")
    before = signature(path)
    if before[2] > MAX_BYTES:
        raise ValueError("small residual byte bound exceeded")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        first = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(first.st_mode)
            or [first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns] != before
        ):
            raise ValueError("residual descriptor differs from registered regular file")
        raw = stream.read(MAX_BYTES + 1)
        final = os.fstat(stream.fileno())
        if [final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns] != before:
            raise ValueError("residual descriptor changed during read")
    check(deadline)
    actual = hashlib.sha256(raw).hexdigest()
    if signature(path) != before or len(raw) != before[2] or (expected and actual != expected):
        raise ValueError("residual artifact signature or immutable hash changed")
    bindings[str(path)] = actual
    return raw


def projection(raw, *, deadline, statistics=None):
    """Resolve bounded backward aliases/merges, then redact by use-site context."""
    if yaml.__version__ != lexical.YAML_VERSION or len(raw) > MAX_BYTES:
        raise ValueError("registered YAML version/size differs")
    parser = lexical.EventParser(lexical.DeadlineStream(raw, deadline))
    anchors, nodes, events = {}, [], 0

    def take(expected=None):
        nonlocal events
        check(deadline)
        value = parser.get_event()
        events += 1
        if events > MAX_EVENTS or value is None:
            raise ValueError("YAML event bound exceeded")
        if expected is not None and not isinstance(value, expected):
            raise ValueError("unexpected YAML event or complex key")
        if getattr(value, "tag", None) is not None:
            raise ValueError("explicit YAML tags forbidden")
        return value

    def parse(depth=0):
        if depth > MAX_DEPTH:
            raise ValueError("YAML depth bound exceeded")
        value = take()
        if isinstance(value, yaml.events.AliasEvent):
            if value.anchor not in anchors:
                raise ValueError("undefined or forward YAML alias")
            return anchors[value.anchor]
        node = {"event": value, "children": []}
        nodes.append(node)
        anchor = getattr(value, "anchor", None)
        if anchor:
            if anchor in anchors:
                raise ValueError("duplicate YAML anchor")
            anchors[anchor] = node
        if isinstance(value, yaml.events.ScalarEvent):
            return node
        if isinstance(value, yaml.events.SequenceStartEvent):
            while not isinstance(parser.peek_event(), yaml.events.SequenceEndEvent):
                node["children"].append(parse(depth + 1))
            take(yaml.events.SequenceEndEvent)
        elif isinstance(value, yaml.events.MappingStartEvent):
            keys = set()
            while not isinstance(parser.peek_event(), yaml.events.MappingEndEvent):
                key = take(yaml.events.ScalarEvent)
                if key.anchor is not None or key.value in keys:
                    raise ValueError("anchored or duplicate explicit YAML key")
                keys.add(key.value)
                node["children"].append((key.value, parse(depth + 1)))
            take(yaml.events.MappingEndEvent)
        else:
            raise ValueError("unsupported YAML node")
        return node

    def mapping(node):
        check(deadline)
        if not isinstance(node["event"], yaml.events.MappingStartEvent):
            raise ValueError("YAML merge requires mapping targets")
        inherited, explicit = {}, {}
        for key, child in node["children"]:
            if key != "<<":
                explicit[key] = child
                continue
            group = (
                child["children"]
                if isinstance(child["event"], yaml.events.SequenceStartEvent)
                else [child]
            )
            for member in group:
                for name, item in mapping(member).items():
                    inherited.setdefault(name, item)  # Earlier merge sequence wins.
        return inherited | explicit  # Explicit fields always override inherited ones.

    visits, active = 0, set()

    def validate(node, depth=0):
        nonlocal visits
        check(deadline)
        visits += 1
        if visits > MAX_EXPANDED or depth > MAX_DEPTH:
            raise ValueError("expanded YAML DAG bound exceeded")
        identity = id(node)
        if identity in active:
            raise ValueError("cyclic YAML alias")
        active.add(identity)
        children = node["children"]
        if isinstance(node["event"], yaml.events.MappingStartEvent):
            children = [child for _key, child in children]
        for child in children:
            validate(child, depth + 1)
        active.remove(identity)

    emitted = 0

    def emit(node, key="", keep=True, depth=0):
        nonlocal emitted
        check(deadline)
        emitted += 1
        if emitted > MAX_EXPANDED or depth > MAX_DEPTH:
            raise ValueError("projected YAML bound exceeded")
        keep = keep and key not in scanner.FORBIDDEN
        event = node["event"]
        if isinstance(event, yaml.events.ScalarEvent):
            if not keep or not scanner.safe_key(key):
                return None
            if event.style is not None:
                return event.value
            if lexical.JSON_SCALAR.fullmatch(event.value):
                result = json.loads(event.value)
                if isinstance(result, float) and not math.isfinite(result):
                    raise ValueError("nonfinite YAML identity scalar")
                return result
            for _tag, pattern in yaml.resolver.Resolver.yaml_implicit_resolvers.get(
                event.value[:1], []
            ):
                if pattern.match(event.value):
                    raise ValueError("ambiguous non-JSON YAML identity scalar")
            return event.value
        if isinstance(event, yaml.events.SequenceStartEvent):
            values = [emit(child, key, keep, depth + 1) for child in node["children"]]
            return values if keep else None
        values = {name: emit(child, name, keep, depth + 1) for name, child in mapping(node).items()}
        return values if keep else None

    try:
        take(yaml.events.StreamStartEvent)
        document = take(yaml.events.DocumentStartEvent)
        if document.tags or document.version:
            raise ValueError("YAML directives forbidden")
        root = parse()
        take(yaml.events.DocumentEndEvent)
        take(yaml.events.StreamEndEvent)
        if parser.peek_event() is not None:
            raise ValueError("multiple YAML documents forbidden")
        validate(root)
        result = emit(root)
        if statistics is not None:
            statistics.update(events=events, unique_nodes=len(nodes), expanded_nodes=visits)
        return result
    except yaml.YAMLError:
        raise ValueError("malformed bounded YAML events") from None
    finally:
        parser.dispose()


def redact_nonfinite(raw, *, deadline):
    """Validate JSON lexically; NaN is permitted only at discarded scalar sites."""
    text = raw.decode("utf-8")
    changes, work = [], 0

    def spaces(pos):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos

    def string(pos):
        match = scanner._STRING.match(text, pos)
        if match is None:
            raise ValueError("malformed bounded JSON string")
        return match.end()

    def value(pos, key="", keep=True, depth=0):
        nonlocal work
        check(deadline)
        work += 1
        if work > MAX_EVENTS or depth > MAX_DEPTH:
            raise ValueError("JSON projection complexity bound exceeded")
        pos = spaces(pos)
        keep = keep and key not in scanner.FORBIDDEN
        if pos >= len(text):
            raise ValueError("truncated JSON")
        if text[pos] == '"':
            return string(pos)
        if text[pos] in "[{":
            object_ = text[pos] == "{"
            end, pos = ("}" if object_ else "]"), spaces(pos + 1)
            keys = set()
            if pos < len(text) and text[pos] == end:
                return pos + 1
            while pos < len(text):
                field = key
                if object_:
                    finish = string(pos)
                    field = json.loads(text[pos:finish])
                    if field in keys:
                        raise ValueError("duplicate JSON key")
                    keys.add(field)
                    pos = spaces(finish)
                    if pos >= len(text) or text[pos] != ":":
                        raise ValueError("JSON object lacks colon")
                    pos += 1
                pos = spaces(value(pos, field, keep, depth + 1))
                if pos < len(text) and text[pos] == end:
                    return pos + 1
                if pos >= len(text) or text[pos] != ",":
                    raise ValueError("invalid JSON delimiter")
                pos = spaces(pos + 1)
                if pos >= len(text) or text[pos] == end:
                    raise ValueError("trailing JSON comma")
            raise ValueError("unterminated JSON container")
        token = NONFINITE.match(text, pos)
        if token:
            if keep and scanner.safe_key(key):
                raise ValueError("nonfinite identity/provenance value forbidden")
            changes.append((pos, token.end()))
            return token.end()
        match = scanner.SCALAR.match(text, pos)
        if match is None:
            raise ValueError("invalid JSON scalar")
        return match.end()

    finish = spaces(value(0))
    if finish != len(text):
        raise ValueError("trailing JSON content")
    for start, end in reversed(changes):
        text = text[:start] + "null" + text[end:]
    return scanner.identity_projection(text), len(changes)


def collected(projected, path):
    result = {key: [] for key in lexical.IDENTITIES} | {"projected_records": 0}
    scanner.collect(projected, result, path)
    return result


def root_metadata(raw, names, *, deadline):
    """Decode only explicit top-level execution-control fields.

    The unchanged identity projector validates the complete JSON while
    redacting benchmark values. A second lexical walk finds requested root
    scalars without constructing unknown arrays, objects or strings.
    """
    if not set(names) <= {"status", "completed_rows", "expected_rows", "plan_sha256", "source_sha"}:
        raise ValueError("only registered execution-control fields may be decoded")
    projected = scanner.identity_projection(raw, budget=SimpleNamespace(deadline=deadline))
    text = raw.decode("utf-8")

    def spaces(pos):
        while pos < len(text) and text[pos] in " \t\r\n":
            pos += 1
        return pos

    def finish(pos):
        pos = spaces(pos)
        if pos >= len(text):
            raise ValueError("truncated reviewed metadata")
        if text[pos] == '"':
            match = scanner._STRING.match(text, pos)
            if match is None:
                raise ValueError("invalid reviewed metadata string")
            return match.end()
        if text[pos] not in "[{":
            match = scanner.SCALAR.match(text, pos)
            if match is None:
                raise ValueError("invalid reviewed metadata scalar")
            return match.end()
        stack, cursor = [], pos
        while cursor < len(text):
            if cursor % 256 == 0:
                check(deadline)
            char = text[cursor]
            if char == '"':
                match = scanner._STRING.match(text, cursor)
                if match is None:
                    raise ValueError("invalid reviewed nested metadata string")
                cursor = match.end()
                continue
            if char in "[{":
                stack.append("}" if char == "{" else "]")
            elif char in "]}":
                if not stack or char != stack.pop():
                    raise ValueError("mismatched reviewed metadata container")
                if not stack:
                    return cursor + 1
            cursor += 1
        raise ValueError("unterminated reviewed metadata container")

    pos, selected = spaces(0), {}
    if pos >= len(text) or text[pos] != "{":
        raise ValueError("reviewed control metadata requires a JSON object")
    pos = spaces(pos + 1)
    while pos < len(text) and text[pos] != "}":
        check(deadline)
        key_end = finish(pos)
        key = json.loads(text[pos:key_end])
        if not isinstance(key, str):
            raise ValueError("reviewed control metadata requires string keys")
        colon = spaces(key_end)
        if colon >= len(text) or text[colon] != ":":
            raise ValueError("reviewed control metadata lacks colon")
        start = spaces(colon + 1)
        end = finish(start)
        if key in names:
            if text[start] in "[{":
                raise ValueError("reviewed execution-control field must be scalar")
            selected[key] = json.loads(text[start:end])
        pos = spaces(end)
        if pos < len(text) and text[pos] == ",":
            pos = spaces(pos + 1)
        elif pos >= len(text) or text[pos] != "}":
            raise ValueError("reviewed control metadata delimiter differs")
    check(deadline)
    return selected, projected


def fixture_proof(path, roots, deadline, bindings):
    spec = read_json(
        read(
            FIXTURE / "spec.json",
            roots,
            deadline,
            bindings,
            "8246a4508993446d4ad511fd36ffaf637e6629b92c061fd0583019dcc574d8a9",
        )
    )
    status = read_json(
        read(
            FIXTURE / "tests-status.json",
            roots,
            deadline,
            bindings,
            "4b680fcb7a6c15f918e535ad5750ae2e6da57536ec56dae3b8b89f11edee0d8a",
        )
    )
    junit = read(
        FIXTURE / "pytest.xml",
        roots,
        deadline,
        bindings,
        "4f4bf96a2d219150184b7062b206336ea97a4684f4518fd75da682b6870888f9",
    )
    if (
        spec["verification_source_sha"] != "01a350b8d4e0148356a78c673a5caf2e3b204a2e"
        or status["verification_source_sha"] != spec["verification_source_sha"]
        or status["status"] != "completed"
        or status["returncode"] != 0
        or status["junit_sha256"] != hashlib.sha256(junit).hexdigest()
    ):
        raise ValueError("actual fixture test provenance differs")
    source_hashes = {
        "tests/unit/test_readout_completion.py": (
            "5e42d3ff1237fbc6d48bdbd1f188d9a9bdb639b00b4cef1dc57947a999757eb2"
        ),
        "scripts/complete_readout_v8_2.py": (
            "73daf80650bd6fe9418f14f550c3991d36a98723fefa73ba2e0663aec4475618"
        ),
    }
    for name, expected in source_hashes.items():
        if not any(
            key.endswith("/mosaickv/" + name) and val == expected
            for key, val in spec["bound_files"].items()
        ):
            raise ValueError("fixture spec omits exact reviewed producer")
        # Current byte-identical source is separate from the historical producer.
        source = SOURCE / "mosaickv" / name
        check(deadline)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("reviewed closed fixture writer changed")
        bindings[str(source)] = expected
    cases = [
        node
        for node in ET.fromstring(junit).iter("testcase")
        if node.get("name")
        == "test_real_runtime_overlay_writes_only_scratch_and_cannot_modify_auth"
    ]
    if len(cases) != 1 or list(cases[0]):
        raise ValueError("fixture test missing, skipped or failed")
    raw = read(path, roots, deadline, bindings, hashlib.sha256(b"scratch-only").hexdigest())
    if raw != b"scratch-only":
        raise ValueError("closed noncredential fixture bytes differ")
    return "actual_passed_test_closed_scalar_fixture", None


def recovered_v6(path, roots, deadline, bindings):
    read(path, roots, deadline, bindings, V6_SHA)
    provenance = read_json(
        read(V6_DERIVED / "analysis_provenance.json", roots, deadline, bindings, V6_PROVENANCE_SHA)
    )
    report_raw = read(V6_DERIVED / "report.json", roots, deadline, bindings, V6_REPORT_SHA)
    report, projected_report = root_metadata(
        report_raw,
        {"status", "completed_rows", "expected_rows", "plan_sha256"},
        deadline=deadline,
    )
    parent = V6_ORIGINAL.parent.parent
    if (
        provenance["parent_root"] != str(parent)
        or provenance["original_analysis_artifacts"].get(str(path)) != V6_SHA
        or provenance["gpu_reruns"] != 0
        or provenance["original_analysis_preserved"] is not True
        or provenance["producer_artifacts_verified"] is not True
        or provenance["producer_git_sha"] != "799ca3c6c3eccadadb52688ebc3823cfbe66522f"
        or report["status"] != "passed_execution"
        or report["completed_rows"] != report["expected_rows"]
        or report["completed_rows"] != 64
        or report["plan_sha256"] != provenance["parent_plan_sha256"]
    ):
        raise ValueError("existing V6 derived-recovery provenance differs")
    plan, _projected_plan = root_metadata(
        read(parent / "plan.json", roots, deadline, bindings, provenance["parent_plan_sha256"]),
        {"source_sha"},
        deadline=deadline,
    )
    if plan["source_sha"] != provenance["producer_git_sha"]:
        raise ValueError("original V6 GPU producer differs from the recovery")
    read(
        V6_DERIVED / "analysis-source.patch",
        roots,
        deadline,
        bindings,
        provenance["analyzer_patch_sha256"],
    )
    for name in ("samples.parquet", "summary.parquet"):
        expected = provenance["original_analysis_artifacts"][str(parent / "analysis" / name)]
        read(parent / "analysis" / name, roots, deadline, bindings, expected)
        read(V6_DERIVED / name, roots, deadline, bindings, expected)
    return "existing_derived_recovery_original_truncation_preserved", collected(
        projected_report, path
    )


def prove(row, roots, deadline):
    """Return an additive proof or raise; callers preserve every failed occurrence."""
    check(deadline)
    original = Path(row["path"])
    path = canonical(original, roots)
    before, bindings, identities = signature(path), {}, None
    if row.get("signature") is not None and list(row["signature"]) != before:
        raise ValueError("registered residual signature changed")
    is_sensitive = sensitive(original) or sensitive(path)
    if path in EMPTY:
        if (
            not stat.S_ISREG(original.lstat().st_mode)
            or before[2] != 0
            or signature(path) != before
        ):
            raise ValueError("registered zero-byte placeholder changed")
        kind, actual = "stable_zero_byte_file_no_body_read", EMPTY_SHA
        bindings[str(path)] = actual
    elif is_sensitive:
        raise ValueError("nonempty credential body remains outside proof scope")
    elif path.suffix in {".yaml", ".yml"}:
        raw = read(path, roots, deadline, bindings)
        stats = {}
        identities = collected(projection(raw, deadline=deadline, statistics=stats), path)
        kind, actual = (
            "bounded_parser_event_DAG_identity_projection",
            hashlib.sha256(raw).hexdigest(),
        )
    elif path == CACHE_FIXTURE:
        kind, identities = fixture_proof(path, roots, deadline, bindings)
        actual = bindings[str(path)]
    elif path in NAN_SUMMARIES:
        expected, count = NAN_SUMMARIES[path]
        raw = read(path, roots, deadline, bindings, expected)
        projected, observed = redact_nonfinite(raw, deadline=deadline)
        if observed != count:
            raise ValueError("registered aggregate nonfinite count differs")
        identities = collected(projected, path)
        kind, actual = "nonfinite_aggregate_redacted_metadata_only", expected
    elif path == V6_ORIGINAL:
        kind, identities = recovered_v6(path, roots, deadline, bindings)
        actual = V6_SHA
    else:
        raise ValueError("no reviewed small-residual proof for this path")
    expected = row.get("current_file_sha256", row.get("sha256"))
    if expected is not None and expected != actual:
        raise ValueError("registered residual full-file hash changed")
    if path not in EMPTY:
        read(path, roots, deadline, bindings, actual)  # Independent closing whole-file hash.
    if signature(path) != before:
        raise ValueError("residual file changed during proof")
    return dict(
        proof_kind=kind,
        path=str(original),
        canonical_path=str(path),
        signature=before,
        current_file_sha256=actual,
        full_file_sha256_before=actual,
        full_file_sha256_after=actual,
        file_hash_basis=(
            "stable_regular_size_zero" if path in EMPTY else "two_complete_bounded_reads"
        ),
        credential_bodies_read=False,
        identities=identities,
        bindings=bindings,
        source_bindings={
            str(Path(module).resolve()): hashlib.sha256(Path(module).read_bytes()).hexdigest()
            for module in (__file__, lexical.__file__, scanner.__file__)
        },
        original_error_sha256s=row.get("original_error_sha256s", []),
        historical_source_executed=False,
        historical_numeric_validity_claimed=False,
        old_historical_byte_claim=False,
        error_waived=False,
        fresh_confirmation_authorized=False,
        historical_media_exposure_resolved=False,
        source_group_disjointness_verified=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
    )
