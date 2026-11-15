"""Bounded reuse of four completed certificates; never experiment admission.

Only explicitly registered small JSON capsules are read. Their raw-data links
are deliberately NOT followed. A generated readiness packet cannot authorize
scanning, materialization, GPU work, or a scientific superiority claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Any

ANCHOR = Path("/sfs/weka/scratch/djy8hg/workdir")
BASE_SOURCE = "5494e684aab0c1ca5e6d60e32ac8c3e237fdb7b3"
MAX_BYTES = 8 * 1024**2
MAX_INPUT = 2 * 1024**3
MAX_SCALAR = 1024**2
MAX_NODES = 100000
AUTHORIZATION_KEYS = (
    "exclusion_scan_authorized",
    "materialization_authorized",
    "fresh_confirmation_authorized",
    "GPU_execution_authorized",
    "experiment_submission_authorized",
    "paper_admission_authorized",
)
CERTIFICATES = {
    "engineering": (
        "fox-review-v52/run-20261113-4b2cf0b-engineering-retry1/engineering_review.json",
        "9237f9f562f29e814d7cdd637a2141958034eb471e0f17b3b975b6d4dac8dc4d",
    ),
    "cpu": (
        "fox-review-v58/run-20261114-5494e68-1/review_decision.json",
        "61c680c73f244e0c79ef84d0fdc7de42dc7f08e05c4ce495875fbc053aa70ef9",
    ),
    "target_recovery": (
        "fox-review-v61/discovery-20261114-1/followup_summary.json",
        "78198fb937e9105583625f5239200c0b490ff09c114bb03f87a368ccd9269c8f",
    ),
    "backup_recovery": (
        "fox-review-v62/discovery-20261114-1/followup_summary.json",
        "cbe1eba4fff185292ff25080e5d3de24052f9f9af6bfcb76105f96ccf38e9b6d",
    ),
}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if len(key.encode()) > MAX_SCALAR:
            raise ValueError("unchanged scalar key limit exceeded")
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("nonfinite JSON constant")


class CapsuleReader:
    """Read a registered leaf via no-follow directory descriptors, within bounds."""

    def __init__(self, anchor: Path = ANCHOR):
        self.anchor = Path(anchor)
        self.started = time.monotonic()
        self.input_bytes = self.nodes = self.projected_bytes = 0
        self.bindings: list[dict[str, Any]] = []

    def _check(self):
        if time.monotonic() - self.started > 210:
            raise TimeoutError("unchanged 210-second metadata deadline")

    def read(self, relative: str, expected: str | None) -> dict[str, Any]:
        self._check()
        path = Path(relative)
        if path.is_absolute() or not path.parts or any(p in (".", "..") for p in path.parts):
            raise ValueError("registered relative capsule path required")
        forbidden = ("credential", "secret", "token", ".codex", ".ssh", ".env", "netrc")
        if any(term in relative.lower() for term in forbidden):
            raise ValueError("protected path is never a capsule")
        descriptor = os.open(self.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parts[:-1]:
                next_fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                os.close(descriptor)
                descriptor = next_fd
            leaf = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
            )
            with os.fdopen(leaf, "rb") as stream:
                before = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or not 0 < before.st_size <= MAX_BYTES
                    or self.input_bytes + before.st_size > MAX_INPUT
                    or self.projected_bytes + before.st_size > MAX_BYTES
                ):
                    raise ValueError("unchanged regular-file/byte limits exceeded")
                raw = stream.read(before.st_size + 1)
                after = os.fstat(stream.fileno())
                current = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        finally:
            os.close(descriptor)

        def signature(info):
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

        if len(raw) != before.st_size or any(
            signature(info) != signature(before) for info in (after, current)
        ):
            raise ValueError("capsule changed during bounded read")
        self.input_bytes += len(raw)
        observed = hashlib.sha256(raw).hexdigest()
        if expected is not None and observed != expected:
            raise ValueError("registered capsule hash differs")
        try:
            value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
        except (RecursionError, UnicodeError) as error:
            raise ValueError("bounded UTF-8 JSON required") from error
        if not isinstance(value, dict):
            raise ValueError("capsule object required")
        pending = [(value, 0)]
        self.projected_bytes += len(raw)
        while pending:
            self._check()
            item, depth = pending.pop()
            self.nodes += 1
            self.projected_bytes += 128  # conservative per-node allowance
            if self.nodes > MAX_NODES or depth > 64 or self.projected_bytes > MAX_BYTES:
                raise ValueError("unchanged live metadata/node/depth limits exceeded")
            if isinstance(item, dict):
                pending.extend((key, depth + 1) for key in item)
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
            elif isinstance(item, str) and len(item.encode()) > MAX_SCALAR:
                raise ValueError("unchanged scalar limit exceeded")
            elif isinstance(item, float) and not math.isfinite(item):
                raise ValueError("nonfinite metadata number")
        self.bindings.append(
            dict(
                path=str(self.anchor / path),
                sha256=observed,
                bytes=len(raw),
                computed_git_blob_oid=hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
            )
        )
        return value


def disabled_authorization(value: Any) -> None:
    if not isinstance(value, dict) or any(value.get(k) is not False for k in AUTHORIZATION_KEYS):
        raise ValueError("all six downstream permissions must be explicitly false")
    if any(v is not False for v in value.values()):
        raise ValueError("this packet never grants any execution permission")


def build_decision(evidence, protocol, baselines, claims):
    """Combine reused certificates and source audits without promoting evidence."""
    if set(evidence) != set(CERTIFICATES):
        raise ValueError("exact four completed certificates required")
    for asset in (protocol, baselines, claims):
        disabled_authorization(asset.get("authorization"))
    if protocol.get("status") != "NO_CONFIG_FROZEN":
        raise ValueError("prospective protocol must retain its unfrozen state")
    candidate = protocol.get("candidate")
    if not isinstance(candidate, dict) or any(
        candidate.get(k, "missing") is not None
        for k in ("selected_config", "selected_config_path", "config_path", "config_sha256")
    ):
        raise ValueError("no selected candidate may be inferred from source inspection")
    cpu = evidence["cpu"]
    counts = cpu.get("CPU", {}).get("counts", {})
    if (
        cpu.get("status") != "passed"
        or cpu.get("source_sha") != BASE_SOURCE
        or type(counts.get("tests")) is not int
        or counts["tests"] != 10761
        or any(
            type(counts.get(k)) is not int or counts[k] != 0
            for k in ("failures", "errors", "skipped")
        )
    ):
        raise ValueError("exact completed CPU certificate required")
    recovery = evidence["target_recovery"]
    expected_counts = {
        "total_artifacts": 134,
        "runtime_verified": 20,
        "partial_total": 47,
        "unbound": 67,
        "semantic_exposure_unresolved": 135,
    }
    observed_counts = recovery.get("counts_before_and_after")
    expected_backup_counts = {k: v for k, v in expected_counts.items() if k != "total_artifacts"}
    backup_counts = evidence["backup_recovery"].get("counts_before_and_after")
    if (
        not isinstance(observed_counts, dict)
        or any(type(v) is not int for v in observed_counts.values())
        or observed_counts != expected_counts
        or not isinstance(backup_counts, dict)
        or any(type(v) is not int for v in backup_counts.values())
        or backup_counts != expected_backup_counts
        or type(evidence["backup_recovery"].get("source_commits_recovered")) is not int
        or evidence["backup_recovery"]["source_commits_recovered"] != 0
    ):
        raise ValueError("historical recovery counts require separate review")
    for capsule in evidence.values():
        for key in (
            "fresh_confirmation_authorized",
            "materialization_authorized",
            "paper_campaign_authorized",
        ):
            if capsule.get(key) is not False:
                raise ValueError("reused capsules must not authorize new work")
    engineering = evidence["engineering"]
    fidelity = engineering.get("source_fidelity", {})
    if (
        fidelity.get("status") != "passed_in_registered_lossless_BF16_MHA_scope"
        or any(
            type(fidelity.get(key)) is not int or fidelity[key] != expected
            for key, expected in (
                ("constructed_arms", 4),
                ("consumed_inputs", 2),
                ("predictions_per_input", 128),
            )
        )
        or type(fidelity.get("maximum_logit_difference")) not in (int, float)
        or fidelity["maximum_logit_difference"] != 0
        or engineering.get("whole_request_peak", {}).get("whole_device_GPU_peak_certified")
        is not False
        or engineering.get("independent_quality_confirmation") is not False
        or engineering.get("latency_superiority_established") is not False
        or engineering.get("six_component_benefit_established") is not False
    ):
        raise ValueError("scope changes require separate empirical review")
    requirements = claims.get("requirements")
    if (
        not isinstance(requirements, list)
        or any(not isinstance(r, dict) for r in requirements)
        or {r.get("id") for r in requirements} != {f"R{i:02d}" for i in range(1, 11)}
        or len(requirements) != 10
    ):
        raise ValueError("exact ten Fox review requirements required")
    return dict(
        schema="fox_paper_readiness_packet_v63",
        status="BLOCKED_FOR_NEW_EXPERIMENTS",
        classification="reused_evidence_and_source_preparation_not_new_measurements",
        reusable_CPU_tests=10761,
        source_fidelity_scope=engineering["source_fidelity"],
        whole_request_peak_scope=engineering["whole_request_peak"],
        history=expected_counts,
        prepared_work=["metric_baseline_audit", "disabled_COCO_protocol", "claim_response_draft"],
        blockers=[
            "original_producer_and_media_source_provenance_incomplete",
            "no_qualified_selected_configuration_frozen",
            "candidate_specific_baseline_scoring_and_capacity_evidence_required",
            "independent_quality_confirmation_not_established",
            "strict_end_to_end_latency_superiority_not_established",
            "six_mechanism_benefit_not_established",
        ],
        reviewer_requirements=requirements,
        authorization={k: False for k in AUTHORIZATION_KEYS},
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
        superiority_established=False,
        GPU_jobs=0,
        credential_fixture_read=False,
        completed_proofs_replayed=0,
    )
