#!/usr/bin/env python3
"""Event-driven completion: validate, decide, then one bounded Codex review.

No queue/log polling, GPU submission, recursive continuation, or gate changes.
The deterministic decision is authoritative; an LLM review cannot promote it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from audit_readout_v8_2 import DOWNSTREAM, load_plan
from bounded_history_pilot import SOURCE, digest, write_once

from mosaickv.evaluation.readout_audit import PROSPECTIVE_SCORES


def decide(report):
    useful = report.get("useful_prospective_signals", [])
    invalid = (
        report.get("status") != "completed"
        or not report.get("original_guard_unchanged")
        or any(report.get(k, True) for k in DOWNSTREAM)
        or len(report.get("tasks", [])) != 2
        or any(name not in PROSPECTIVE_SCORES[1:] for name in useful)
        or any(
            not all(
                t.get("screening", {}).get(name, {}).get("descriptive_signal") is True
                for t in report["tasks"]
            )
            for name in useful
        )
    )
    if invalid:
        return dict(
            action="stop_integrity_failure",
            reason="Audit failed or authorization/guard mismatch",
            cpu_candidate_design_authorized=False,
            **dict.fromkeys(DOWNSTREAM, False),
            gpu_submission_authorized=False,
        )
    return dict(
        action="cpu_candidate_design_only" if useful else "stop_no_prospective_signal",
        reason="Descriptive pre-probe signal needs bounded CPU design tests, not promotion"
        if useful
        else "No fixed pre-probe score passed the descriptive screen on both consumed inputs",
        eligible_scores=useful,
        cpu_candidate_design_authorized=bool(useful),
        registered_accepted_proposals=sum(
            t["registered_accepted_proposals"] for t in report["tasks"]
        ),
        **dict.fromkeys(DOWNSTREAM, False),
        gpu_submission_authorized=False,
    )


def codex_command(executable, root, schema):
    """No credentials are copied/persisted, no shell tools or mutable source."""
    return [
        executable,
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--disable",
        "shell_snapshot",
        "-c",
        "features.shell_tool=false",
        "-c",
        'web_search="disabled"',
        "-c",
        'history.persistence="none"',
        "-c",
        f"log_dir={json.dumps(str(root / 'codex/log'))}",
        "-c",
        f"sqlite_home={json.dumps(str(root / 'codex/state'))}",
        "--cd",
        str(SOURCE),
        "--color",
        "never",
        "--json",
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(root / "codex/review.json"),
        "-",
    ]


def protected_command(bwrap, executable, root, cache, application_home, schema, auth_fd):
    """All application runtime writes land on scratch, not the home filesystem.

    Bind an already-open, read-only auth file descriptor after overlaying the
    application runtime. Do not copy or persist credential contents. HOME and
    CODEX_HOME are not reassigned. A descriptor avoids resolving a mount alias
    before Bubblewrap has installed that alias.
    """
    runtime = root / "codex/runtime"
    return [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        str(root),
        str(root),
        "--bind",
        str(cache),
        str(cache),
        "--bind",
        str(runtime),
        str(application_home),
        "--ro-bind-fd",
        str(auth_fd),
        str(application_home / "auth.json"),
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--",
        *codex_command(executable, root, schema),
    ]


def application_directory():
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).resolve()


def review_with_codex(root, executable, decision, report):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return dict(
            status="unavailable",
            error="read_only_home_mount_unavailable",
            automatic_code_or_GPU_execution=False,
        )
    cache = Path(os.environ["MOSAICKV_CACHE_ROOT"]).resolve()
    if (
        not cache.is_absolute()
        or Path.home().resolve() == cache
        or Path.home().resolve() in cache.parents
    ):
        raise ValueError("Codex callback cache cannot be in home")
    destination = root / "codex"
    destination.mkdir()
    application_home = application_directory()
    if not (application_home / "auth.json").is_file():
        return dict(
            status="unavailable",
            error="existing_read_only_Codex_auth_not_found",
            automatic_code_or_GPU_execution=False,
        )
    (destination / "runtime").mkdir()
    schema = destination / "review-schema.json"
    write_once(
        schema,
        dict(
            type="object",
            properties=dict(
                summary=dict(type="string"),
                action=dict(type="string", enum=[decision["action"]]),
                gpu_submission_authorized=dict(type="boolean", enum=[False]),
                paper_campaign_authorized=dict(type="boolean", enum=[False]),
            ),
            required=[
                "summary",
                "action",
                "gpu_submission_authorized",
                "paper_campaign_authorized",
            ],
            additionalProperties=False,
        ),
    )
    prompt = (
        "You are the one-shot post-Slurm MosaicKV analysis reviewer. Do not use tools, edit files, "
        "run commands, launch jobs, change gates, or continue recursively. Interpret only the "
        "validated aggregate diagnostic supplied below. It contains two consumed inputs, not "
        "benchmark-quality evidence. Distinguish pre-probe scores "
        "from candidate-forward explanations. "
        "Honor the deterministic action exactly; the next authorized action is already recorded. "
        "Write a concise review in the required JSON schema.\n"
        + json.dumps(
            dict(
                decision=decision,
                tasks=report.get("tasks", []),
                limitations=report.get("limitations", []),
            )
        )
    )
    with (
        (destination / "events.jsonl").open("x") as out,
        (destination / "stderr.log").open("x") as err,
    ):
        auth_fd = None
        try:
            auth_fd = os.open(application_home / "auth.json", os.O_RDONLY)
            command = protected_command(
                bwrap, executable, root, cache, application_home, schema, auth_fd
            )
            result = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=out,
                stderr=err,
                timeout=180,
                cwd=SOURCE,
                check=False,
                pass_fds=(auth_fd,),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return dict(
                status="unavailable",
                error=type(error).__name__,
                automatic_code_or_GPU_execution=False,
            )
        finally:
            if auth_fd is not None:
                os.close(auth_fd)
    if result.returncode != 0 or not (destination / "review.json").is_file():
        return dict(
            status="failed", returncode=result.returncode, automatic_code_or_GPU_execution=False
        )
    try:
        review = json.loads((destination / "review.json").read_text())
    except ValueError:
        return dict(status="invalid_output", automatic_code_or_GPU_execution=False)
    if (
        review["action"] != decision["action"]
        or review["gpu_submission_authorized"]
        or review["paper_campaign_authorized"]
    ):
        return dict(status="rejected_scope_mismatch", automatic_code_or_GPU_execution=False)
    return dict(
        status="completed",
        review_sha256=digest(destination / "review.json"),
        independent_ephemeral_review_not_UI_session_resume=True,
        application_runtime_physical_storage="scratch_bind_mount",
        credentials_copied=False,
        original_auth_mounted_read_only=True,
        automatic_code_or_GPU_execution=False,
    )


def complete(root, plan_sha, executable=None):
    plan = load_plan(root, plan_sha)
    if (root / "decision.json").exists():
        raise FileExistsError("completion decision is write-once; duplicate callback refused")
    report = dict(status="failed", **dict.fromkeys(DOWNSTREAM, False))
    try:
        completion = json.loads((root / "worker-completed.json").read_text())
        if (
            completion["source_sha"] != plan["source_sha"]
            or completion["report_sha256"] != digest(root / "audit/report.json")
            or completion["artifacts_sha256"] != digest(root / "audit/artifact-hashes.json")
        ):
            raise ValueError("CPU completion evidence changed")
        for path, sha in json.loads((root / "audit/artifact-hashes.json").read_text()).items():
            if digest(path) != sha:
                raise ValueError("CPU audit artifact changed")
        report = json.loads((root / "audit/report.json").read_text())
    except (OSError, ValueError, KeyError) as error:
        report.update(error=f"{type(error).__name__}: {error}")
    decision = decide(report)
    decision.update(
        source_sha=plan["source_sha"],
        plan_sha256=plan_sha,
        log_polling_performed=False,
        report_sha256=digest(root / "audit/report.json")
        if (root / "audit/report.json").is_file()
        else None,
    )
    write_once(root / "decision.json", decision)
    # This is the automatically taken next action. A useful signal gets only
    # a bounded CPU design specification; no new inference or generalization.
    if decision["cpu_candidate_design_authorized"]:
        write_once(
            root / "bounded-design-request.json",
            dict(
                scope="CPU_only_bounded_proposer_design_on_same_consumed_inputs",
                allowed_scores=decision["eligible_scores"],
                max_proposals=16,
                substitutions="one_head_one_token_equal_capacity",
                original_guard_unchanged=True,
                **dict.fromkeys(DOWNSTREAM, False),
                gpu_submission_authorized=False,
            ),
        )
    else:
        write_once(
            root / "negative-audit-seal.json",
            dict(
                classification="negative_or_integrity_blocked_consumed_input_diagnostic",
                decision_sha256=digest(root / "decision.json"),
                preserve_all_artifacts=True,
                no_new_inference=True,
                **dict.fromkeys(DOWNSTREAM, False),
            ),
        )
    review = (
        review_with_codex(root, executable, decision, report)
        if executable
        else dict(status="not_configured")
    )
    write_once(root / "codex-review-status.json", review)
    write_once(
        root / "completion.json",
        dict(
            status="completed",
            action=decision["action"],
            decision_sha256=digest(root / "decision.json"),
            codex_review=review,
            gpu_submission_authorized=False,
            **dict.fromkeys(DOWNSTREAM, False),
        ),
    )
    print(json.dumps(dict(decision=decision, codex_review=review)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--codex-executable")
    args = parser.parse_args()
    complete(args.root.resolve(), args.plan_sha, args.codex_executable)
