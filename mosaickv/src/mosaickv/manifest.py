"""Immutable run-manifest construction and atomic writing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.types import Backend, JsonObject, MeasurementType


class ManifestError(RuntimeError):
    """Raised when complete, immutable manifest provenance cannot be produced."""


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    """Hash UTF-8 text."""

    return sha256_bytes(payload.encode("utf-8"))


def _validate_digest(value: str, path: str, *, allow_not_applicable: bool = False) -> None:
    if allow_not_applicable and value == "not_applicable":
        return
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        suffix = " or 'not_applicable'" if allow_not_applicable else ""
        raise ManifestError(f"{path} must be a lowercase SHA-256 digest{suffix}")


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """Hashes that establish controlled-comparison input identity."""

    prompt_set_sha: str
    media_set_sha: str
    preprocessing_sha: str
    tokenization_sha: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _validate_digest(str(value), f"inputs.{name}")


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Content hashes for immutable outputs."""

    raw_output_sha: str
    metrics_sha: str
    log_sha: str

    def __post_init__(self) -> None:
        _validate_digest(self.raw_output_sha, "artifacts.raw_output_sha")
        _validate_digest(self.metrics_sha, "artifacts.metrics_sha", allow_not_applicable=True)
        _validate_digest(self.log_sha, "artifacts.log_sha")


def _run(command: list[str], cwd: Path, *, check: bool = True) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ManifestError(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout


def _find_repo_root(start: Path | None = None) -> Path:
    candidates = [start or Path.cwd(), Path(__file__).resolve()]
    for candidate in candidates:
        directory = candidate if candidate.is_dir() else candidate.parent
        try:
            output = _run(["git", "rev-parse", "--show-toplevel"], directory)
        except (ManifestError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
        root = Path(output.strip())
        if root.is_dir():
            return root
    raise ManifestError("cannot locate a git repository for source provenance")


def _worktree_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    status = _run(["git", "status", "--porcelain=v1", "-z"], repo_root)
    digest.update(status.encode("utf-8", errors="surrogateescape"))
    for command in (
        ["git", "diff", "--binary", "HEAD"],
        ["git", "diff", "--binary", "--cached", "HEAD"],
    ):
        digest.update(_run(command, repo_root).encode("utf-8", errors="surrogateescape"))
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard", "-z"], repo_root)
    for relative in sorted(item for item in untracked.split("\0") if item):
        path = repo_root / relative
        digest.update(relative.encode("utf-8"))
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif path.is_symlink():
            digest.update(os.readlink(path).encode("utf-8"))
    return digest.hexdigest()


def _source_provenance(repo_root: Path) -> JsonObject:
    git_sha = _run(["git", "rev-parse", "HEAD"], repo_root).strip()
    if len(git_sha) != 40:
        raise ManifestError(f"git SHA is not a 40-character commit: {git_sha!r}")
    dirty = bool(_run(["git", "status", "--porcelain=v1"], repo_root).strip())
    return {
        "git_sha": git_sha,
        "git_dirty": dirty,
        "patch_sha": _worktree_fingerprint(repo_root) if dirty else "not_applicable",
        "canonical_eligible": not dirty,
        "config_sha": "filled_by_manifest_builder",
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _nvidia_hardware() -> tuple[str, int, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "not_used", 0, "not_used"
    if completed.returncode != 0:
        return "not_used", 0, "not_used"
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    names: set[str] = set()
    drivers: set[str] = set()
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) == 2:
            names.add(parts[0])
            drivers.add(parts[1])
    return (
        ",".join(sorted(names)) if names else "not_used",
        len(rows),
        ",".join(sorted(drivers)) if drivers else "not_used",
    )


def _torch_cuda_version() -> str:
    if _version("torch") == "not_installed":
        return "not_used"
    script = "import torch; print(torch.version.cuda or 'not_used')"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return "not_used"
    return completed.stdout.strip() if completed.returncode == 0 else "not_used"


def _software_provenance(backend: Backend, gpu_count: int, driver: str) -> JsonObject:
    uses_torch = backend in {Backend.HUGGINGFACE, Backend.VLLM, Backend.SGLANG}
    return {
        "cuda": _torch_cuda_version() if gpu_count > 0 and uses_torch else "not_used",
        "driver": driver if gpu_count > 0 else "not_used",
        "pytorch": _version("torch") if uses_torch else "not_used",
        # Both pinned engine backends use Transformers for checkpoint config,
        # tokenization, and multimodal processor construction even though they
        # do not use the Transformers generation loop.
        "transformers": _version("transformers") if uses_torch else "not_used",
        "vllm": _version("vllm") if backend == Backend.VLLM else "not_used",
        "sglang": _version("sglang") if backend == Backend.SGLANG else "not_used",
        "python": platform.python_version(),
        "numpy": _version("numpy"),
    }


def _generation_sha(config: RunConfig) -> str:
    payload = json.dumps(asdict(config.generation), sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


class RunManifestWriter:
    """Build and atomically write complete schema-v1 run manifests."""

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = _find_repo_root(Path(repo_root) if repo_root is not None else None)

    def build(
        self,
        config: RunConfig,
        measurement_type: MeasurementType,
        inputs: InputProvenance,
        artifacts: ArtifactProvenance,
        *,
        run_id: str | None = None,
        started_at_utc: str | None = None,
        execution_metadata: JsonObject | None = None,
        attention_implementation_override: str | None = None,
    ) -> JsonObject:
        source = _source_provenance(self.repo_root)
        source["config_sha"] = config_sha256(config)
        lock_path = self.repo_root / "mosaickv" / "env" / "common" / "requirements.lock"
        if not lock_path.is_file():
            raise ManifestError(f"common environment lock is missing: {lock_path}")
        patch_path = (
            self.repo_root
            / "mosaickv"
            / "env"
            / "patches"
            / "sglang-0.4.3.post4-transformers-4.49.patch"
        )
        if not patch_path.is_file():
            raise ManifestError(f"common environment patch is missing: {patch_path}")
        gpu_type, gpu_count, driver = _nvidia_hardware()
        execution: JsonObject = {
            "backend": config.execution.backend.value,
            "attention_implementation": (
                attention_implementation_override or config.execution.attention_implementation
            ),
            "seed": config.execution.seed,
            "deterministic_algorithms": config.execution.deterministic_algorithms,
        }
        for key, value in (execution_metadata or {}).items():
            if key in execution:
                raise ManifestError(f"execution metadata cannot replace required field: {key}")
            execution[key] = value
        return {
            "schema_version": 1,
            "run_id": run_id or uuid.uuid4().hex,
            "measurement_type": measurement_type.value,
            "started_at_utc": started_at_utc or datetime.now(UTC).isoformat(),
            "source": source,
            "model": {
                "id": config.model.id,
                "revision": config.model.revision,
                "precision": config.model.precision.value,
            },
            "dataset": {
                "id": config.dataset.id,
                "revision": config.dataset.revision,
                "split": config.dataset.split,
            },
            "software": _software_provenance(config.execution.backend, gpu_count, driver),
            "hardware": {"gpu_type": gpu_type, "gpu_count": gpu_count},
            "execution": execution,
            "inputs": cast("JsonObject", asdict(inputs)),
            "generation": {
                "parameters_sha": _generation_sha(config),
                "output_length_policy": config.generation.output_length_policy.value,
            },
            "cache": {
                "budget_value": config.cache.budget_value,
                "budget_unit": config.cache.budget_unit.value,
                "retention_ratio": config.cache.retention_ratio,
                "accounting_spec_sha": config.cache.accounting_spec_sha,
            },
            "artifacts": cast("JsonObject", asdict(artifacts)),
            "resolved_config": canonical_config(config),
            "environment": {
                "name": "common",
                "lock_path": str(lock_path.relative_to(self.repo_root)),
                "lock_sha256": sha256_bytes(lock_path.read_bytes()),
                "patches": {
                    "sglang_transformers_compat": {
                        "path": str(patch_path.relative_to(self.repo_root)),
                        "sha256": sha256_bytes(patch_path.read_bytes()),
                    }
                },
                "cache_root": os.environ.get("MOSAICKV_CACHE_ROOT", "not_set"),
                "platform": platform.platform(),
                "hostname": platform.node(),
                "python_executable": sys.executable,
                "detected_packages": {
                    name: _version(name)
                    for name in ("torch", "transformers", "vllm", "sglang", "numpy")
                },
            },
        }

    def write(
        self,
        path: str | Path,
        config: RunConfig,
        measurement_type: MeasurementType,
        inputs: InputProvenance,
        artifacts: ArtifactProvenance,
        *,
        run_id: str | None = None,
        started_at_utc: str | None = None,
        execution_metadata: JsonObject | None = None,
        attention_implementation_override: str | None = None,
    ) -> Path:
        """Write a new manifest atomically and refuse to overwrite an artifact."""

        output_path = Path(path)
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite immutable manifest: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.build(
            config,
            measurement_type,
            inputs,
            artifacts,
            run_id=run_id,
            started_at_utc=started_at_utc,
            execution_metadata=execution_metadata,
            attention_implementation_override=attention_implementation_override,
        )
        serialized = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, output_path)
        except BaseException:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise
        return output_path


__all__ = [
    "ArtifactProvenance",
    "InputProvenance",
    "ManifestError",
    "RunManifestWriter",
    "sha256_bytes",
    "sha256_text",
]
