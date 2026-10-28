"""Fox-review campaign contracts. No labels enter selection or subset ordering.

This module is orchestration/measurement, not a replacement compression policy.
Historical artifacts are read-only; all outputs use exclusive, atomic publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SEED = 20271014
TASKS = ("ActionPrediction", "ActionSequence", "CLEVR-Change", "Spot-the-Diff")
HISTORICAL = {
    "coco": Path("/scratch/djy8hg/runs/mosaickv/coco-shared-32-edb6edf-retry1-20261014"),
    "milebench": Path("/scratch/djy8hg/runs/mosaickv/milebench-shared-minimum-89ecd50-20261018"),
}
COCO_POOL = Path(
    "/scratch/djy8hg/runs/mosaickv/coco-shared-pool-edb6edf-retry1-20261014/ledger.json"
)
MILE_POOL = Path("/scratch/djy8hg/runs/mosaickv/milebench-representative-89ecd50-20261018/ledgers")
BUDGETS = {"coco": (20, 40), "milebench": (16, 40)}
METHODS = {
    "coco": ("full_kv", "prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact"),
    "milebench": ("full_kv", "prompt_attention_topk", "lookm_reimpl", "mosaickv_exact"),
}
STAGES = (
    "smoke",
    "correctness",
    "scoring",
    "measurement",
    "optimization",
    "mechanisms",
    "confirmation",
    "batching",
    "export",
    "audit",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_once(path: Path, value: Any) -> None:
    """Atomic create, never replace an existing artifact (including race winners)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def clean_source(source: Path, expected: str | None = None) -> str:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=source, text=True
    )
    if dirty or (expected is not None and actual != expected):
        raise ValueError("canonical execution requires the registered clean source SHA")
    return actual


def development_subset(samples: list[dict], *, count: int, task: str) -> list[dict]:
    """Length-stratified input-only round robin; never read reference/score fields."""
    if len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("duplicate source identities")
    if len(samples) < count:
        raise ValueError(f"{task}: insufficient development inputs")
    ordered = sorted(
        samples, key=lambda s: (len(s["prompt"]), digest([task, s["sample_id"], SEED]))
    )
    buckets = [ordered[i * len(ordered) // 4 : (i + 1) * len(ordered) // 4] for i in range(4)]
    for bucket in buckets:
        bucket.sort(key=lambda s: digest([SEED, task, s["sample_id"]]))
    result = []
    for index in range(max(map(len, buckets))):
        for bucket in buckets:
            if index < len(bucket):
                result.append(bucket[index])
                if len(result) == count:
                    return result
    raise AssertionError("unreachable")


def source_config(workload: str) -> dict:
    return json.loads(
        (HISTORICAL[workload] / "configs/validation/mosaickv_exact-40.json").read_text()
    )


def validate_locked_policy(config: dict) -> None:
    required = {
        ("model", "precision"): "fp16",
        ("cache", "block_size"): 1,
        ("forecasting", "prompt_window"): 16,
        ("forecasting", "draft_steps"): 0,
        ("graph", "candidate_nonmandatory_fraction"): 0.08,
        ("graph", "max_neighbors"): 8,
        ("graph", "uncertain_head_fraction"): 1 / 256,
        ("selection", "coverage_correction_rounds"): 2,
        ("utility", "lambda_q"): 1.0,
        ("utility", "lambda_v"): -0.3,
        ("utility", "lambda_o"): -0.3,
        ("selection", "lambda_g"): -0.25,
        ("selection", "lambda_m"): -0.25,
    }
    # Field spellings are source-defined; missing fields are blockers.
    for (section, key), expected in required.items():
        if config.get(section, {}).get(key) != expected:
            raise ValueError(f"historical policy mismatch: {section}.{key} != {expected!r}")
    for name in ("prototypes", "residual", "repair"):
        if config[name]["enabled"]:
            raise ValueError(f"{name} must be disabled")
    if config.get("bounded_decode") is not None or config.get("output_preserving") is not None:
        raise ValueError("campaign prohibits bounded-history/draft experimental variants")


@dataclass(frozen=True)
class CampaignRow:
    run_id: str
    sample_id: str
    media_group_ids: tuple[str, ...]
    workload: str
    subtask: str | None
    method: str
    effective_method: str | None
    budget_bytes: int | None
    source_sha: str
    config_sha: str
    manifest_path: str
    input_sha: str
    processor_sha: str
    scorer_sha: str
    attempt: str
    repeat: int
    status: str
    error: str | None = None
    answer: str | None = None
    token_ids: tuple[int, ...] = ()
    emitted_tokens: int | None = None
    forward_steps: int | None = None
    cache_append_steps: int | None = None
    initial_kv_bytes: int | None = None
    final_kv_bytes: int | None = None
    peak_kv_bytes: int | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    e2e_wall_seconds: float | None = None
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    trace_path: str | None = None
    gpu_uuid: str | None = None
    slurm_job_id: str | None = None
    score: float | None = None
    measurement_type: str = "measured_development"

    def __post_init__(self):
        if self.status not in {"completed", "failed", "unsupported"}:
            raise ValueError("invalid row status")
        if self.status == "completed":
            required = (
                self.answer,
                self.emitted_tokens,
                self.forward_steps,
                self.initial_kv_bytes,
                self.final_kv_bytes,
                self.peak_kv_bytes,
                self.e2e_wall_seconds,
                self.trace_path,
            )
            if any(value is None for value in required):
                raise ValueError("completed row has missing measurements")
            if self.effective_method is None or "fallback" in self.effective_method:
                raise ValueError("unexpected fallback")
        if self.status == "failed" and not self.error:
            raise ValueError("failed samples need an explicit error")
        for value in (self.e2e_wall_seconds, self.score):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("invalid measurement")


def unique_storage_bytes(tensors) -> int:
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storages[identity] = storage.nbytes()
    return sum(storages.values())


def independent_masked_attention(query, keys, values, positions, *, mandatory=()):
    """Original-layout float64 oracle. Does not call production packer/masks."""
    import torch

    if (
        query.dtype != keys.dtype
        or keys.dtype != values.dtype
        or query.device != keys.device
        or keys.device != values.device
    ):
        raise ValueError("dtype/device mismatch")
    if query.ndim != 4 or keys.ndim != 4 or keys.shape != values.shape:
        raise ValueError("expected B,H,N,D layout")
    batch, kv_heads, source_length, dimension = keys.shape
    if query.shape[0] != batch or query.shape[-1] != dimension or query.shape[1] % kv_heads:
        raise ValueError("invalid GQA geometry")
    if len(positions) != kv_heads:
        raise ValueError("missing head positions")
    allowed = torch.zeros((kv_heads, source_length), device=keys.device, dtype=torch.bool)
    for head, ids in enumerate(positions):
        if not ids or len(set(ids)) != len(ids) or tuple(sorted(ids)) != tuple(ids):
            raise ValueError("positions must be unique, nonempty and monotonic")
        if min(ids) < 0 or max(ids) >= source_length or not set(mandatory).issubset(ids):
            raise ValueError("invalid positions or infeasible mandatory state")
        allowed[head, list(ids)] = True
    groups = query.shape[1] // kv_heads
    key = keys.repeat_interleave(groups, dim=1)
    value = values.repeat_interleave(groups, dim=1)
    logits = query @ key.transpose(-1, -2) / math.sqrt(dimension)
    logits = logits.masked_fill(
        ~allowed.repeat_interleave(groups, dim=0)[None, :, None, :], -torch.inf
    )
    return logits.softmax(-1) @ value


def slurm_seconds(value: str) -> int:
    """Parse Slurm [days-]HH:MM:SS, MM:SS and integer minutes."""
    day, clock = value.split("-", 1) if "-" in value else ("0", value)
    components = [int(x) for x in clock.split(":")]
    if len(components) == 1:
        seconds = components[0] * 60
    elif len(components) == 2:
        seconds = components[0] * 60 + components[1]
    elif len(components) == 3:
        seconds = components[0] * 3600 + components[1] * 60 + components[2]
    else:
        raise ValueError("invalid Slurm duration")
    return int(day) * 86400 + seconds


def gate_payload(stage, *, source_sha, protocol_sha, passed, tests, blockers=()):
    return dict(
        schema_version=SCHEMA_VERSION,
        stage=stage,
        source_sha=source_sha,
        protocol_sha256=protocol_sha,
        status="passed" if passed else "blocked",
        tests_actually_executed=tests,
        blockers=list(blockers),
    )
