"""Pure, hash-bound validation of one original scheduler/startup record.

The accounting record is corroborating startup provenance, not proof of a
writer's outputs or identities. Full commands are never returned or included
in error messages. No file, scheduler, repository, or credential is accessed.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import PurePosixPath
from typing import Any

from mosaickv.evaluation import history_occurrence_reconciliation as contract

MAXIMUM_BYTES = 8 * 1024**2
SPEC_FIELDS = {
    "job_id",
    "accounting_sha256",
    "startup_sha256",
    "source_sha",
    "output_path",
    "stage",
}
_SHELL_SEPARATORS = {";", "&&"}


def _path(value: Any, *, relative: bool = False) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 4096
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("bounded canonical path required")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute() == relative
        or str(parsed) != value
        or ".." in parsed.parts
        or value == "."
        or ".codex" in parsed.parts
        or parsed.name.casefold() in {"auth.json", "credentials.json", ".env", ".netrc"}
    ):
        raise ValueError("noncredential canonical path required")
    if relative:
        return value
    prefix = "/scratch/djy8hg"
    if value == prefix or value.startswith(prefix + "/"):
        return "/sfs/weka" + value
    prefix = "/sfs/weka/scratch/djy8hg"
    if value != prefix and not value.startswith(prefix + "/"):
        raise ValueError("registered scratch path required")
    return value


def _tokens(command: str) -> list[str]:
    if "`" in command or "$(" in command or "\x00" in command or "#" in command:
        raise ValueError("unsupported dynamic shell command")
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        values = list(lexer)
    except ValueError:
        raise ValueError("invalid or truncated scheduler shell quoting") from None
    for value in values:
        if "HF_TOKEN=" in value:
            supplied = value.split("HF_TOKEN=", 1)[1]
            if supplied not in {"$HF_TOKEN", "${HF_TOKEN}"}:
                raise ValueError("literal credential assignment is forbidden")
    return values


def _wrapped_command(submit_line: str) -> str:
    # sacct SubmitLine commonly removes the outer --wrap quoting. Everything
    # after its single marker therefore belongs to the shell body; tokenizing
    # the entire submission first would silently discard almost all of it.
    matches = list(re.finditer(r"(?:^|\s)--wrap=", submit_line))
    if len(matches) != 1:
        raise ValueError("one complete original scheduler wrap is required")
    match = matches[0]
    prefix, body = submit_line[: match.start()], submit_line[match.end() :].strip()
    try:
        tokens = shlex.split(prefix, posix=True)
    except ValueError:
        raise ValueError("invalid or truncated scheduler submit line") from None
    if not tokens or PurePosixPath(tokens[0]).name != "sbatch":
        raise ValueError("original sbatch submission required")
    for token in tokens[1:]:
        if token == "--wrap":
            raise ValueError("duplicate scheduler wrap argument")
        if "HF_TOKEN=" in token:
            # Outer scheduler --export can also persist a literal secret.
            supplied = token.split("HF_TOKEN=", 1)[1]
            if supplied not in {"$HF_TOKEN", "${HF_TOKEN}"}:
                raise ValueError("literal credential assignment is forbidden")
    if not body:
        raise ValueError("one complete original scheduler wrap is required")
    if body[0] in {"'", '"'}:
        try:
            wrapped = shlex.split(body, posix=True)
        except ValueError:
            raise ValueError("invalid or truncated scheduler wrap quoting") from None
        if len(wrapped) != 1:
            raise ValueError("unexpected arguments after quoted scheduler wrap")
        body = wrapped[0]
    return body


def _arguments(segment: list[str], expected_stage: str, expected_output: str) -> None:
    options = {}
    index = 0
    while index < len(segment):
        value = segment[index]
        if value in {"--stage", "--output"}:
            key = value
            index += 1
            if index == len(segment) or segment[index].startswith("--"):
                raise ValueError("analyzer argument value missing")
            supplied = segment[index]
        elif value.startswith("--stage=") or value.startswith("--output="):
            key, supplied = value.split("=", 1)
        else:
            index += 1
            continue
        if key in options:
            raise ValueError("duplicate analyzer target or stage")
        options[key] = supplied
        index += 1
    if set(options) != {"--stage", "--output"}:
        raise ValueError("explicit analyzer output and stage are required")
    if options["--stage"] != expected_stage or _path(options["--output"]) != expected_output:
        raise ValueError("analyzer output or stage differs from registered scope")


def _command_scope(command: str, expected_stage: str, expected_output: str) -> None:
    tokens = _tokens(command)
    if any(
        token
        and all(character in ";&|()<>" for character in token)
        and token not in _SHELL_SEPARATORS
        for token in tokens
    ):
        raise ValueError("unsupported shell operator in registered scheduler wrap")
    segments = []
    current = []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            if not current:
                raise ValueError("empty scheduler command segment")
            segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    guards, analyzers = [], []
    for index, segment in enumerate(segments):
        names = [PurePosixPath(value).name for value in segment]
        if "assert_clean_worktree.sh" in names:
            if names[0] not in {"assert_clean_worktree.sh", "bash", "sh"}:
                raise ValueError("clean-worktree guard is not an executed command")
            if names[0] in {"bash", "sh"} and (
                len(names) != 2 or names[1] != "assert_clean_worktree.sh"
            ):
                raise ValueError("unsupported clean-worktree guard invocation")
            guards.append(index)
        if "analyze_milebench_shared_bytes.py" in names:
            # A final shell exec replaces the wrapper with this exact analyzer;
            # it is not a second interpreter or an arbitrary wrapper program.
            executed = segment[1:] if names[0] == "exec" else segment
            executed_names = [PurePosixPath(value).name for value in executed]
            if not executed_names or executed_names[0] not in {
                "analyze_milebench_shared_bytes.py",
                "python",
                "python3",
            }:
                raise ValueError("analyzer is not an executed command")
            analyzer_index = executed_names.index("analyze_milebench_shared_bytes.py")
            if analyzer_index != (
                0 if executed_names[0] == "analyze_milebench_shared_bytes.py" else 1
            ):
                raise ValueError("unsupported analyzer invocation")
            _arguments(executed[analyzer_index + 1 :], expected_stage, expected_output)
            analyzers.append(index)
    if len(guards) != 1 or len(analyzers) != 1 or guards[0] >= analyzers[0]:
        raise ValueError("one clean-worktree guard must precede the registered analyzer")


def validate_startup(accounting: bytes, startup: bytes, spec: dict[str, Any]) -> dict[str, str]:
    """Validate immutable raw records and return only six sanitized scope facts."""
    if not isinstance(spec, dict) or set(spec) != SPEC_FIELDS:
        raise ValueError("exact static startup specification required")
    job_id = spec["job_id"]
    if not isinstance(job_id, str) or re.fullmatch(r"[1-9][0-9]{0,15}", job_id) is None:
        raise ValueError("explicit original numeric scheduler job ID required")
    source = spec["source_sha"]
    if not isinstance(source, str) or contract.SHA.fullmatch(source) is None:
        raise ValueError("explicit immutable original source SHA required")
    if not isinstance(spec["stage"], str) or spec["stage"] not in {"validation", "larger"}:
        raise ValueError("only registered validation or larger stage is allowed")
    output = _path(spec["output_path"])
    for raw, key in ((accounting, "accounting_sha256"), (startup, "startup_sha256")):
        if not isinstance(raw, bytes) or not 0 < len(raw) <= MAXIMUM_BYTES:
            raise ValueError("bounded nonempty raw scheduler/startup bytes required")
        expected = contract._hash(spec[key], "registered original scheduler/startup hash")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("raw scheduler/startup hash differs from static binding")
    if startup != f"clean_git_sha={source}\n".encode("ascii"):
        raise ValueError("exact clean source startup record required")
    try:
        text = accounting.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError("UTF-8 scheduler accounting required") from None
    if not text.endswith("\n") or text.count("\n") != 1 or "\r" in text:
        raise ValueError("one complete original scheduler accounting line required")
    parts = text[:-1].split("|", 5)
    if len(parts) != 6:
        raise ValueError("six original scheduler accounting fields required")
    actual_job, user, state, exit_code, workdir, submit_line = parts
    if (actual_job, user, state, exit_code) != (job_id, "djy8hg", "COMPLETED", "0:0"):
        raise ValueError("scheduler identity, completion state, or exit code differs")
    _path(workdir)
    _command_scope(_wrapped_command(submit_line), spec["stage"], output)
    return dict(
        job_id=job_id,
        source_sha=source,
        output_path=output,
        stage=spec["stage"],
        accounting_sha256=spec["accounting_sha256"],
        startup_sha256=spec["startup_sha256"],
    )
