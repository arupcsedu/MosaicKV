#!/usr/bin/env python3
"""Fail-closed launcher for pinned, unmodified published baseline code.

The launcher is a control-plane wrapper: it validates source identity,
licenses, environment compatibility, scratch-only cache/output paths, and the
exact command.  It never relabels a MosaicKV reimplementation as official.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class OfficialBaseline:
    name: str
    relative_root: str
    revision: str
    license_name: str
    required_versions: dict[str, str]
    python_paths: tuple[str, ...]
    required_modules: tuple[str, ...]


BASELINES = {
    "prefixkv": OfficialBaseline(
        "PrefixKV",
        "third_party/PrefixKV",
        "597f1ab032704951550f93bcc8a23f1454b80aa4",
        "MIT",
        {
            "torch": "2.1.2",
            "transformers": "4.31.0",
            "accelerate": "0.21.0",
        },
        (".",),
        ("llava", "rouge"),
    ),
    "lookm": OfficialBaseline(
        "LOOK-M",
        "third_party/LOOK-M",
        "ecf0f51a9c416c2d85e47faf2638502f01a6d748",
        "MIT",
        {
            "torch": "2.1.1",
            "transformers": "4.37.0",
            "accelerate": "0.24.0",
        },
        (".", "LLaVA-mix_merge_v1"),
        ("llava", "omegaconf", "shortuuid", "einops_exts"),
    ),
    "elastic_cache": OfficialBaseline(
        "Elastic Cache",
        "third_party/ElasticCache",
        "3138e03fc8894231d98a2e5353f27a59af854cc3",
        "MIT",
        {
            "torch": "2.1.2",
            "transformers": "4.31.0",
            "accelerate": "0.21.0",
            "tokenizers": "0.13.1",
        },
        (".",),
        ("llava", "rouge"),
    ),
}


def _git(path: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outside_home(path: Path, label: str) -> Path:
    resolved = path.resolve()
    home = Path.home().resolve()
    if resolved == home or home in resolved.parents:
        raise RuntimeError(f"{label} cannot be inside the home directory: {resolved}")
    return resolved


def _python_path(repository: Path, baseline: OfficialBaseline) -> tuple[Path, ...]:
    paths = tuple(
        (repository / relative).resolve(strict=True) for relative in baseline.python_paths
    )
    if any(path != repository and repository not in path.parents for path in paths):
        raise RuntimeError("official Python path escapes the pinned repository")
    return paths


def _environment_versions(
    python: Path,
    python_paths: tuple[Path, ...],
    required_modules: tuple[str, ...],
) -> dict[str, str]:
    program = """
import importlib.util
import importlib.metadata as metadata
import json
import sys
names = ['torch', 'transformers', 'accelerate', 'tokenizers']
missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError('missing official modules: ' + ', '.join(missing))
print(json.dumps({name: metadata.version(name) for name in names}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (*map(str, python_paths), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [str(python), "-c", program, *required_modules],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return dict(json.loads(completed.stdout))


def _validate_source(root: Path, baseline: OfficialBaseline) -> tuple[Path, str]:
    repository = (root / baseline.relative_root).resolve(strict=True)
    observed = _git(repository, "rev-parse", "HEAD")
    if observed != baseline.revision:
        raise RuntimeError(
            f"{baseline.name} revision mismatch: expected {baseline.revision}, observed {observed}"
        )
    dirty = _git(repository, "status", "--porcelain")
    if dirty:
        raise RuntimeError(f"official {baseline.name} checkout is modified")
    license_path = repository / "LICENSE"
    if not license_path.is_file():
        raise RuntimeError(f"official {baseline.name} checkout has no preserved LICENSE")
    return repository, _sha256(license_path)


def _validate_versions(observed: dict[str, str], required: dict[str, str]) -> None:
    errors = []
    for package, expected in required.items():
        actual = observed.get(package, "missing")
        if not (actual == expected or actual.startswith(f"{expected}+")):
            errors.append(f"{package}: expected {expected}, observed {actual}")
    if errors:
        raise RuntimeError(
            "official baseline environment is incompatible:\n- " + "\n- ".join(errors)
        )


def _prefixkv_command(args: argparse.Namespace, root: Path, python: Path) -> list[str]:
    required = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "media_root": args.media_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "PrefixKV launch requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    return [
        str(python),
        str(root / "mosaickv/scripts/run_prefixkv_llava_parity.py"),
        "--repository-root",
        str(root),
        "--model-path",
        str(args.model_path),
        "--data-path",
        str(args.data_path),
        "--media-root",
        str(args.media_root),
        "--output-directory",
        str(args.output_directory),
        "--retention-ratio",
        str(args.retention_ratio),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
    ]


def _lookm_command(args: argparse.Namespace, repository: Path, python: Path) -> list[str]:
    required = {
        "data_path": args.data_path,
        "model_name": args.model_name,
        "model_configs": args.model_configs,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "LOOK-M launch requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    return [
        str(python),
        str(repository / "generate.py"),
        "--data_dir",
        str(args.data_path),
        "--dataset_name",
        str(args.dataset_name),
        "--model_name",
        str(args.model_name),
        "--model_configs",
        str(args.model_configs),
        "--output_dir",
        str(args.output_directory),
        "--kv_mode",
        str(args.lookm_kv_mode),
        "--hh_ratio",
        str(args.lookm_important_ratio),
        "--recent_ratio",
        str(args.lookm_recent_ratio),
    ]


def _elastic_cache_command(
    args: argparse.Namespace, repository: Path, python: Path
) -> list[str]:
    required = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "media_root": args.media_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Elastic Cache launch requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if not 0 < args.retention_ratio <= 1:
        raise ValueError("Elastic Cache retention ratio must be in (0, 1]")
    return [
        str(python),
        str(repository / "eval_generate.py"),
        "--model-path",
        str(args.model_path),
        "--data-path",
        str(args.data_path),
        "--image-path",
        str(args.media_root),
        "--temperature",
        "0",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--eval-samples",
        "1",
        "--method",
        "elastic",
        "--ratio",
        str(1.0 - args.retention_ratio),
        "--exp-name",
        f"official-elastic-cache-seed-{args.seed}",
    ]


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite official launch manifest: {path}")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", choices=sorted(BASELINES))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--model-configs", type=Path)
    parser.add_argument("--dataset-name", default="data/sample.json")
    parser.add_argument("--retention-ratio", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookm-kv-mode", default="text_prior_pivot_merge")
    parser.add_argument("--lookm-important-ratio", type=float, default=0.1)
    parser.add_argument("--lookm-recent-ratio", type=float, default=0.1)
    args = parser.parse_args()

    root = args.repository_root.resolve(strict=True)
    python = args.python.resolve(strict=True)
    output = _outside_home(args.output_directory, "official output")
    cache_root = _outside_home(
        Path(os.environ.get("MOSAICKV_CACHE_ROOT", "/scratch/djy8hg/cache/mosaickv")),
        "cache root",
    )
    baseline = BASELINES[args.baseline]
    repository, license_sha = _validate_source(root, baseline)
    python_paths = _python_path(repository, baseline)
    versions = _environment_versions(python, python_paths, baseline.required_modules)
    _validate_versions(versions, baseline.required_versions)
    if args.baseline == "prefixkv":
        command = _prefixkv_command(args, root, python)
    elif args.baseline == "lookm":
        command = _lookm_command(args, repository, python)
    else:
        command = _elastic_cache_command(args, repository, python)
    payload: dict[str, Any] = {
        "baseline": baseline.name,
        "measurement_type": "baseline_official_measured",
        "official_code": True,
        "local_reimplementation": False,
        "source": asdict(baseline),
        "source_license_sha256": license_sha,
        "python": str(python),
        "environment_versions": versions,
        "cache_root": str(cache_root),
        "output_directory": str(output),
        "command": command,
        "check_only": bool(args.check_only),
    }
    if args.check_only:
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    _write_manifest(output / "official_launch_manifest.json", payload)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (*map(str, python_paths), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        command,
        cwd=repository if args.baseline in {"lookm", "elastic_cache"} else root,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
