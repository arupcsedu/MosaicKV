"""Pure, pinned provenance checks for six original negative freeze jobs.

The caller verifies Git commit/tree/blob identities and the original seal before
passing its small execution projection. This module performs no I/O or historical
code execution. It identifies the file writer, not experiment correctness, model
dependencies, installed environments, sample identities, or scientific admission.
The critical source map is deliberately NOT a complete runtime import closure.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
from pathlib import PurePosixPath
from typing import Any

MAXIMUM_RECORD_BYTES = 32768
MAXIMUM_LIVE_BYTES = 8 * 1024**2
MAXIMUM_NODES = 100000
MAXIMUM_DEPTH = 64
MAXIMUM_SCALAR_BYTES = 1024**2
BASE_WRITER = "mosaickv/scripts/analyze_choice_c_v74_label_blind_calibration.py"
ROBUST_WRITER = "mosaickv/scripts/analyze_choice_c_v86_robust_fidelity.py"
CLASSIFICATION = "sealed_negative_label_blind_calibration_result"
PROJECT = "/sfs/weka/scratch/djy8hg/workdir/MosaicKV"
SPEC_FIELDS = {
    "version",
    "job_id",
    "source_sha",
    "repository_tree_oid",
    "seal_path",
    "seal_sha256",
    "seal_commit",
    "seal_blob_oid",
    "protocol_path",
    "protocol_sha256",
    "run_root",
    "frozen_output",
    "expected_state",
    "expected_exit_code",
    "accounting_sha256",
    "startup_sha256",
    "batch_record_sha256",
    "wrapper_path",
    "wrapper_sha256",
    "entrypoint_path",
    "source_blob_sha256s",
    "outputs",
}
SEAL_FIELDS = {
    "classification",
    "run_root",
    "freeze_job",
    "protocol_path",
    "protocol_sha256",
    "frozen_candidate_created",
}
# Canonical JSON digests of the six reviewed registry specifications. A caller
# cannot self-authorize altered commands/source by replacing their expected hash.
REGISTERED_SPEC_SHA256 = {
    76: "0dd3ebad32eb1a18fc1e0df2f5711afd5c679c741cebe445e09f5946d5ed71d6",
    78: "f487253e52698ee41b38e4ba57d1bc183f9e815595321a6a2ea7a035239ab5fb",
    79: "e8b8d46a803bc5c31c236e7d68db0a02ffcaae8f24ec8a0730477782fd12bfa0",
    80: "6a5b4bbdb0230cf81a76df8ff0f5ffe8d8842d2cecccb712e21bd0155f91cde0",
    81: "363c2b294399e855545783c56267f7d49c1df88b6bf124525c4e44b7d3cad674",
    88: "9907ccd71d4d368a3e2340feb5ab2ecab01892264c0971778c5a38dafc53328e",
}
REGISTERED_FACTS_SHA256 = {
    76: "effb44fafd359237d7b748140d3590f3fec75623daf9fbaed232d80d2c5c903e",
    78: "20e31fd7cc2ca6f226a0bce9ca02a3a52a9d1554db5e9b05b0a1df002dcd6082",
    79: "4bc312364bd605494a7f4940f5a08b7df6fa86c48406d89c5d1fa50fd3dcb76b",
    80: "48d66915ba06b0fdf46883182fdac76536f95b630fc5c109c0f3c8112132c4c2",
    81: "f8b3cd4908bacdccc2de80eae0b189322261590ee0d4a531f28396d8f808806d",
    88: "61bdd51507c05149fd94449c407916d3938c6a229e4f4a1d4a49da2aa7182d81",
}


def _bounded(value: Any) -> None:
    pending = [(value, 0)]
    nodes = size = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAXIMUM_NODES or depth > MAXIMUM_DEPTH:
            raise ValueError("metadata node limit or depth limit exceeded")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("string metadata keys required")
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            scalar_bytes = len(item.encode("utf-8"))
            if scalar_bytes > MAXIMUM_SCALAR_BYTES:
                raise ValueError("metadata string scalar byte limit exceeded")
            size += scalar_bytes
        elif item is not None and type(item) not in (bool, int):
            raise ValueError("plain bounded JSON metadata required")
        if size > MAXIMUM_LIVE_BYTES:
            raise ValueError("live metadata byte limit exceeded")


def _digest(value: Any) -> str:
    _bounded(value)
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (RecursionError, ValueError, TypeError):
        raise ValueError("bounded canonical JSON metadata required") from None
    if len(encoded) > MAXIMUM_LIVE_BYTES:
        raise ValueError("live metadata byte limit exceeded")
    return hashlib.sha256(encoded).hexdigest()


def _path(value: Any, *, relative: bool = False) -> str:
    if (
        not isinstance(value, str)
        or not 0 < len(value.encode()) <= 4096
        or "\\" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("bounded canonical noncredential path required")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute() == relative
        or str(parsed) != value
        or value == "."
        or ".." in parsed.parts
        or ".codex" in parsed.parts
        or parsed.name.casefold() in {"auth.json", "credentials.json", ".env", ".netrc"}
    ):
        raise ValueError("bounded canonical noncredential path required")
    if relative:
        return value
    if value.startswith("/scratch/djy8hg/"):
        return "/sfs/weka" + value
    if not value.startswith("/sfs/weka/scratch/djy8hg/"):
        raise ValueError("only the registered scratch alias is supported")
    return value


def _raw(raw: bytes, expected: str, label: str, maximum: int) -> None:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= maximum:
        raise ValueError(f"bounded nonempty {label} bytes required")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"registered {label} hash differs")


def validate_registered_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Reject unregistered Git paths/hashes before the caller performs any I/O."""
    if not isinstance(spec, dict) or set(spec) != SPEC_FIELDS:
        raise ValueError("exact registered freeze specification required")
    version = spec["version"]
    if type(version) is not int or REGISTERED_SPEC_SHA256.get(version) != _digest(spec):
        raise ValueError("freeze specification is not one of the six pinned jobs")
    return json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))


def validate_runtime_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Check the exact pinned sanitized projection used by lineage consumers.

    The evidence digest hashes all other fields using canonical JSON, not an
    assertion boolean. This projection is not a substitute for the raw-record
    checks in ``validate_freeze_runtime`` or the caller's immutable Git checks.
    """
    if not isinstance(facts, dict) or "runtime_evidence_sha256" not in facts:
        raise ValueError("explicit validated freeze-runtime fact projection required")
    payload = {key: value for key, value in facts.items() if key != "runtime_evidence_sha256"}
    version = payload.get("version")
    expected = REGISTERED_FACTS_SHA256.get(version) if type(version) is int else None
    if (
        expected is None
        or _digest(payload) != expected
        or facts["runtime_evidence_sha256"] != expected
    ):
        raise ValueError("freeze-runtime facts differ from the six pinned projections")
    # JSON round-trip returns an independent plain-data object without file I/O.
    return json.loads(json.dumps(facts, sort_keys=True, allow_nan=False))


def _accounting(raw: bytes, spec: dict[str, Any]) -> None:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError("UTF-8 original accounting required") from None
    if not text.endswith("\n") or text.count("\n") != 1 or "\r" in text:
        raise ValueError("one complete original accounting line required")
    parts = text[:-1].split("|")
    if len(parts) != 6 or parts[:4] != [
        spec["job_id"],
        "djy8hg",
        spec["expected_state"],
        spec["expected_exit_code"],
    ]:
        raise ValueError("original accounting identity/state/exit differs")
    if _path(parts[4]) != PROJECT + "/mosaickv":
        raise ValueError("original accounting workdir differs")
    if any(c in parts[5] for c in ("`", "$", "\x00", "#", ";", "|", "&", "<", ">")):
        raise ValueError("dynamic original submission is forbidden")
    try:
        argv = shlex.split(parts[5], posix=True)
    except ValueError:
        raise ValueError("complete original submission quoting required") from None
    if (
        not argv
        or argv[0] != "sbatch"
        or argv[-1] != spec["wrapper_path"].removeprefix("mosaickv/")
    ):
        raise ValueError("exact submitted freeze wrapper required")
    options = argv[1:-1]
    expected_flags = {"--parsable"}
    if spec["version"] in (78, 81):
        expected_flags |= {"--nodes=2", "--ntasks=2"}
    flags = {arg for arg in options if arg in expected_flags}
    dependencies = [arg for arg in options if arg.startswith("--dependency=")]
    exports = [arg for arg in options if arg.startswith("--export=")]
    if (
        len(set(options)) != len(options)
        or flags != expected_flags
        or len(dependencies) != 1
        or len(exports) != 1
        or re.fullmatch(r"--dependency=afterok:[1-9][0-9]{0,15}", dependencies[0]) is None
        or len(options) != len(flags) + 2
    ):
        raise ValueError("exact bounded original sbatch options required")
    supplied = exports[0].removeprefix("--export=").split(",")
    if len(supplied) != 4 or supplied[0] != "ALL":
        raise ValueError("exact original export fieldmap required")
    values = {}
    for item in supplied[1:]:
        key, separator, value = item.partition("=")
        if not separator or key in values:
            raise ValueError("unique explicit original export values required")
        values[key] = _path(value)
    prefix = f"MOSAICKV_V{spec['version']}_"
    expected = {
        prefix + "OUTPUT_ROOT": _path(spec["run_root"]),
        prefix + "SPEC": PROJECT + "/" + spec["protocol_path"],
        prefix + "FROZEN": _path(spec["frozen_output"]),
    }
    if values != expected:
        raise ValueError("original export/output/protocol/frozen scope differs")


def _wrapper(source: bytes, spec: dict[str, Any]) -> None:
    # The entire wrapper is already hash-pinned. This additional reviewed
    # template proves guard ordering, fixed complete argv and exit semantics;
    # it is intentionally not a general shell interpreter.
    text = source.decode("utf-8", errors="strict")
    common = """set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V@V@_OUTPUT_ROOT:-@ROOT@}
SPEC=${MOSAICKV_V@V@_SPEC:-${PACKAGE_ROOT}/@PROTOCOL@}
FROZEN=${MOSAICKV_V@V@_FROZEN:-${OUTPUT_ROOT}/analysis/frozen_candidate.yaml}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
"""
    command = """"${PYTHON}" @ENTRY@ \\
  --spec "${SPEC}" \\
  --output-root "${OUTPUT_ROOT}" \\
  --frozen-output "${FROZEN}" \\
  --run-prefix v@V@
"""
    if spec["version"] == 88:
        tail = (
            """
set +e
"""
            + command
            + """ANALYSIS_STATUS=$?
set -e

if [[ "${ANALYSIS_STATUS}" -eq 2 && ! -e "${FROZEN}" ]]; then
  printf 'V88 completed as a negative calibration: no candidate passed every unchanged gate.\\n'
  exit 0
fi
exit "${ANALYSIS_STATUS}"
"""
        )
    else:
        tail = command
    expected = common + tail
    for key, value in {
        "@V@": str(spec["version"]),
        "@ROOT@": spec["run_root"],
        "@PROTOCOL@": spec["protocol_path"].removeprefix("mosaickv/"),
        "@ENTRY@": spec["entrypoint_path"].removeprefix("mosaickv/"),
    }.items():
        expected = expected.replace(key, value)
    before, marker, body = text.partition("set -euo pipefail\n")
    if not marker or marker + body != expected or not before.startswith("#!/bin/bash\n"):
        raise ValueError("reviewed guarded freeze wrapper template differs")


def _tree(source: bytes) -> ast.Module:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        raise ValueError("bounded parsed writer source required") from None
    for count, _node in enumerate(ast.walk(tree), start=1):
        if count > MAXIMUM_NODES:
            raise ValueError("writer AST node limit exceeded")
    return tree


def _writer(source: bytes, *, robust: bool) -> None:
    tree = _tree(source)
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not {"analyze", "main"}.issubset(definitions):
        raise ValueError("reviewed analyze/main writer functions required")
    default = "v86" if robust else "v74"
    sorted_print = ", sort_keys=True" if robust else ""
    expected_main = ast.parse(f'''def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frozen-output", required=True, type=Path)
    parser.add_argument("--run-prefix", default="{default}")
    args = parser.parse_args()
    result = analyze(
        args.spec.resolve(), args.output_root.resolve(), args.frozen_output.resolve(),
        run_prefix=str(args.run_prefix),
    )
    print(json.dumps(
        {{"selected_candidate": result["selected_candidate"], "labels_accessed": False}}
        {sorted_print}
    ))
    return 0 if result["selected_candidate"] is not None else 2
''').body[0]
    if ast.dump(definitions["main"]) != ast.dump(expected_main):
        raise ValueError("reviewed full argv and deliberate negative return differ")
    footer = ast.parse('if __name__ == "__main__":\n    raise SystemExit(main())').body[0]
    if ast.dump(tree.body[-1]) != ast.dump(footer):
        raise ValueError("reviewed Python main footer differs")
    analyze = definitions["analyze"]
    statements = [ast.dump(node) for node in analyze.body]
    root = ast.parse('analysis_root = output_root / "analysis"').body[0]
    ranking = "robust_ranking" if robust else "candidate_ranking"
    filename = (
        "robust_fidelity_candidate_ranking.parquet"
        if robust
        else ("label_blind_candidate_ranking.parquet")
    )
    write = ast.parse(
        f'pd.DataFrame({ranking}).to_parquet(analysis_root / "{filename}", index=False)'
    ).body[0]
    end = ast.parse("return result").body[0]
    try:
        positions = [statements.index(ast.dump(node)) for node in (root, write, end)]
    except ValueError:
        raise ValueError("unconditional registered ranking write and return required") from None
    if positions != sorted(positions) or statements.count(ast.dump(write)) != 1:
        raise ValueError("ranking must be written before the normal analysis return")
    if robust:
        imports = ast.parse(
            "from analyze_choice_c_v74_label_blind_calibration import analyze as analyze_base"
        ).body[0]
        call = ast.parse(
            "base = analyze_base(spec_path, output_root, provisional_path, run_prefix=run_prefix)"
        ).body[0]
        if ast.dump(imports) not in [ast.dump(node) for node in tree.body] or not any(
            ast.dump(node) == ast.dump(call) for node in ast.walk(analyze)
        ):
            raise ValueError("robust entrypoint must invoke the original base writer")


def validate_freeze_runtime(
    accounting: bytes,
    startup: bytes,
    batch_record: bytes,
    seal_execution: dict[str, Any],
    spec: dict[str, Any],
    source_blobs: dict[str, bytes],
    repository_tree_oid: str,
) -> dict[str, Any]:
    """Return sanitized writer facts for one exact registered historical job.

    Git verification of the seal/commit/tree and blob membership belongs to the
    bounded I/O caller. A negative scheduler status is not accepted on its own.
    """
    spec = validate_registered_spec(spec)
    version = spec["version"]
    if repository_tree_oid != spec["repository_tree_oid"]:
        raise ValueError("original committed repository tree differs")
    _raw(accounting, spec["accounting_sha256"], "accounting", MAXIMUM_RECORD_BYTES)
    _raw(startup, spec["startup_sha256"], "startup", 55)
    _raw(batch_record, spec["batch_record_sha256"], "batch record", MAXIMUM_RECORD_BYTES)
    if startup != f"clean_git_sha={spec['source_sha']}\n".encode("ascii"):
        raise ValueError("exact original 55-byte clean source prefix required")
    expected_sources = spec["source_blob_sha256s"]
    if not isinstance(source_blobs, dict) or set(source_blobs) != set(expected_sources):
        raise ValueError("exact critical writer source fieldmap required")
    live_bytes = len(accounting) + len(startup) + len(batch_record)
    for path, expected in expected_sources.items():
        _path(path, relative=True)
        source = source_blobs[path]
        _raw(source, expected, "critical source", MAXIMUM_LIVE_BYTES)
        live_bytes += len(source)
        if live_bytes > MAXIMUM_LIVE_BYTES:
            raise ValueError("live critical-source byte limit exceeded")
    expected_seal_fields = SEAL_FIELDS | ({"freeze_state"} if version == 88 else set())
    if not isinstance(seal_execution, dict) or set(seal_execution) != expected_seal_fields:
        raise ValueError("exact original seal execution projection required")
    _bounded(seal_execution)
    protocol = _path(seal_execution["protocol_path"], relative=True)
    if not protocol.startswith("mosaickv/"):
        protocol = "mosaickv/" + protocol
    job = seal_execution["freeze_job"]
    if (
        type(job) not in (str, int)
        or str(job) != spec["job_id"]
        or seal_execution["classification"] != CLASSIFICATION
        or seal_execution["frozen_candidate_created"] is not False
        or _path(seal_execution["run_root"]) != _path(spec["run_root"])
        or protocol != spec["protocol_path"]
        or seal_execution["protocol_sha256"] != spec["protocol_sha256"]
        or (
            version == 88
            and seal_execution["freeze_state"] != ("completed_failed_closed_no_eligible_candidate")
        )
    ):
        raise ValueError("original negative seal/job/output/protocol link differs")
    _accounting(accounting, spec)
    wrapper = source_blobs[spec["wrapper_path"]]
    header = f"Batch Script for {spec['job_id']}\n" + "-" * 80 + "\n"
    if batch_record != header.encode("ascii") + wrapper:
        raise ValueError("original batch header/body differs from actual source wrapper")
    _wrapper(wrapper, spec)
    _writer(source_blobs[BASE_WRITER], robust=False)
    if version == 88:
        _writer(source_blobs[ROBUST_WRITER], robust=True)
    facts = {
        key: spec[key]
        for key in (
            "version",
            "job_id",
            "source_sha",
            "repository_tree_oid",
            "frozen_output",
            "protocol_path",
            "protocol_sha256",
            "seal_path",
            "seal_sha256",
            "expected_state",
            "expected_exit_code",
            "accounting_sha256",
            "startup_sha256",
            "batch_record_sha256",
            "wrapper_path",
            "wrapper_sha256",
            "entrypoint_path",
        )
    }
    facts.update(
        schema="historical_freeze_runtime_facts_v54",
        run_root=_path(spec["run_root"]),
        frozen_output=_path(spec["frozen_output"]),
        source_blob_sha256s=dict(expected_sources),
        outputs=[dict(output, path=_path(output["path"])) for output in spec["outputs"]],
        critical_source_closure_sha256=_digest(expected_sources),
        negative_result_preserved=True,
        quality_gate_pass=False,
        scientific_dependency_validation=False,
        canonical_eligible=False,
    )
    facts["runtime_evidence_sha256"] = _digest(facts)
    return facts
