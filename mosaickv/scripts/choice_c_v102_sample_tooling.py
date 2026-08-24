#!/usr/bin/env python3
"""Run fail-closed, configuration-aware V102 sample-protocol tooling."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.evaluation.disjoint_samples import (
    DatasetContract,
    SampleProtocolError,
    check_disjointness,
    construct_id_only_catalog_from_dataset,
    materialize_deterministic_ledger,
    require_protocol_authorization,
    scan_exclusion_sources,
    sha256_file,
    write_once,
)

EXPERIMENT_ID = "choice_c_v102_v100_fresh_cross_dataset_calibration"
PROJECTED_CATALOG_TOOLING_SHA = "67beb0af20022228643eb211d2c5c06395b98db4"
CANONICAL_SPEC_RELATIVE_PATH = Path(
    "configs/experiments/choice_c_v102_v100_fresh_cross_dataset_calibration.yaml"
)


def _git_provenance(repository: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def _load_protocol(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V102 protocol must contain an object")
    protocol = cast("dict[str, Any]", value)
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected protocol experiment ID")
    return protocol


def _contract(task: dict[str, Any]) -> DatasetContract:
    raw_config = task.get("dataset_config")
    if not isinstance(raw_config, str) or not raw_config.strip():
        raise ValueError("V102 requires a nonempty dataset_config for every task")
    return DatasetContract(
        task=str(task["task"]),
        dataset_id=str(task["dataset_id"]),
        dataset_revision=str(task["dataset_revision"]),
        split=str(task["split"]),
        dataset_config=raw_config,
    )


def _contracts(protocol: dict[str, Any]) -> tuple[DatasetContract, ...]:
    return tuple(_contract(task) for task in cast("list[dict[str, Any]]", protocol["tasks"]))


def _resolve_protocol_path(package_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (package_root / path).resolve()


def _require_execution_snapshot(
    protocol: dict[str, Any], *, repository: Path, action: str
) -> tuple[str, bool]:
    sha, dirty = _git_provenance(repository)
    source = cast("dict[str, Any]", protocol["candidate_source"])
    expected_field = (
        "catalog_constructor_implementation_git_sha"
        if action == "catalog"
        else "tooling_implementation_git_sha"
    )
    expected = str(source.get(expected_field, ""))
    if expected in {"", "pending_tooling_commit", "pending_tooling_seal"}:
        raise PermissionError("V102 tooling SHA remains pending its immutable commit and seal")
    if action == "catalog":
        _require_catalog_authorization_seal(protocol)
    ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", expected, sha],
        check=False,
    )
    candidate = str(source["candidate_implementation_git_sha"])
    changed = subprocess.run(
        ["git", "-C", str(repository), "diff", "--name-only", f"{candidate}..{sha}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    allowlist = set(cast("list[str]", source["execution_diff_allowlist"]))
    unexpected = sorted(set(changed) - allowlist)
    if dirty or ancestor.returncode != 0 or unexpected:
        raise RuntimeError(
            "V102 requires a clean descendant of its sealed tooling commit and permits only "
            f"allowlisted changes after candidate SHA {candidate}; observed sha={sha}, "
            f"dirty={dirty}, unexpected={unexpected}"
        )
    return sha, dirty


def _require_catalog_authorization_seal(protocol: dict[str, Any]) -> None:
    seal = protocol.get("replacement_projected_catalog_tooling")
    authorization = protocol.get("candidate_catalog_construction_authorization")
    if (
        not isinstance(seal, dict)
        or seal.get("status")
        != "committed_cpu_tested_sealed_projected_construction_authorized_not_run"
        or seal.get("implementation_git_sha") != PROJECTED_CATALOG_TOOLING_SHA
        or seal.get("implementation_reviewed_after_commit") is not True
        or seal.get("tooling_sealed") is not True
        or seal.get("replacement_construction_authorized") is not True
        or not isinstance(authorization, dict)
        or authorization.get("status")
        != "projected_id_only_catalog_construction_authorized_not_executed"
        or authorization.get("reviewed_projected_tooling_git_sha") != PROJECTED_CATALOG_TOOLING_SHA
        or authorization.get("catalog_construction_performed") is not False
    ):
        raise PermissionError("V102 projected ID-only catalog authorization is not fully sealed")


def _require_canonical_spec_path(spec_path: Path, *, package_root: Path) -> None:
    expected = (package_root / CANONICAL_SPEC_RELATIVE_PATH).resolve()
    if spec_path.resolve() != expected:
        raise PermissionError(f"V102 catalog construction requires canonical spec path: {expected}")


def _require_disjointness_authorization(
    protocol: dict[str, Any],
    *,
    exclusion_manifest: Path,
    ledger_paths: tuple[Path, ...],
    output: Path,
) -> None:
    authorization = protocol.get("disjointness_check_authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("status")
        != "configuration_aware_cpu_disjointness_check_authorized_not_executed"
        or authorization.get("disjointness_check_performed") is not False
        or authorization.get("report_reviewed") is not False
    ):
        raise PermissionError("V102 disjointness-check authorization is not active")
    expected_manifest = Path(str(authorization.get("exclusion_manifest_path", ""))).resolve()
    expected_ledgers = tuple(
        Path(str(path)).resolve() for path in authorization.get("ledger_paths", ())
    )
    expected_output = Path(str(authorization.get("output_path", ""))).resolve()
    if exclusion_manifest.resolve() != expected_manifest:
        raise PermissionError("V102 disjointness check requires the sealed exclusion manifest")
    if tuple(path.resolve() for path in ledger_paths) != expected_ledgers:
        raise PermissionError("V102 disjointness check requires the two reviewed ledgers in order")
    if output.resolve() != expected_output:
        raise PermissionError("V102 disjointness report path differs from the authorized path")
    expected_hashes = tuple(str(value) for value in authorization.get("ledger_sha256", ()))
    if len(expected_hashes) != len(expected_ledgers):
        raise PermissionError("V102 disjointness authorization has invalid ledger hashes")
    if sha256_file(expected_manifest) != authorization.get("exclusion_manifest_sha256"):
        raise SampleProtocolError("sealed V102 exclusion manifest differs")
    if any(
        sha256_file(path) != expected
        for path, expected in zip(expected_ledgers, expected_hashes, strict=True)
    ):
        raise SampleProtocolError("reviewed V102 ledger differs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")


def _task_contract(
    protocol: dict[str, Any], task_name: str
) -> tuple[dict[str, Any], DatasetContract]:
    tasks = cast("list[dict[str, Any]]", protocol["tasks"])
    matches = [task for task in tasks if task.get("task") == task_name]
    if len(matches) != 1:
        raise ValueError(f"V102 has no unique task contract for {task_name!r}")
    return matches[0], _contract(matches[0])


def _require_scratch_cache_environment() -> Path:
    scratch_root = Path("/scratch/djy8hg").resolve()
    required = (
        "MOSAICKV_CACHE_ROOT",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "TMPDIR",
    )
    for variable in required:
        raw = os.environ.get(variable)
        if not raw:
            raise RuntimeError(f"{variable} is unset; source mosaickv/scripts/cache_env.sh first")
        resolved = Path(raw).resolve()
        if not resolved.is_relative_to(scratch_root):
            raise RuntimeError(f"{variable} must resolve below {scratch_root}: {resolved}")
    return Path(os.environ["HF_DATASETS_CACHE"]).resolve()


class _ProjectedIdDataset:
    """Minimal dataset facade that cannot expose anything except one ID column."""

    def __init__(self, *, id_field: str, raw_ids: tuple[object, ...]) -> None:
        self.column_names = [id_field]
        self._id_field = id_field
        self._raw_ids = raw_ids

    def __len__(self) -> int:
        return len(self._raw_ids)

    def select_columns(self, columns: list[str]) -> _ProjectedIdDataset:
        if columns != [self._id_field]:
            raise SampleProtocolError("projected source exposes only its audited ID column")
        return self

    def __getitem__(self, key: str) -> tuple[object, ...]:
        if key != self._id_field:
            raise SampleProtocolError("projected source exposes only its audited ID column")
        return self._raw_ids


def _select_pinned_parquet_shards(
    *,
    repo_files: list[str],
    parquet_glob: str,
    expected_shard_count: int,
) -> tuple[str, ...]:
    if (
        not parquet_glob.endswith(".parquet")
        or parquet_glob.startswith("/")
        or ".." in Path(parquet_glob).parts
        or expected_shard_count < 1
    ):
        raise SampleProtocolError("invalid pinned Parquet shard contract")
    matches = tuple(
        sorted(
            path
            for path in repo_files
            if fnmatchcase(path, parquet_glob)
            and path.endswith(".parquet")
            and not path.startswith("/")
            and ".." not in Path(path).parts
        )
    )
    if len(matches) != expected_shard_count or len(set(matches)) != len(matches):
        raise SampleProtocolError(
            f"pinned Parquet contract matched {len(matches)} shards; "
            f"expected {expected_shard_count}"
        )
    return matches


def _read_projected_parquet_ids(
    *,
    shard_paths: tuple[str, ...],
    id_field: str,
    open_shard: Callable[[str], Any],
    read_table: Callable[..., Any],
) -> tuple[object, ...]:
    raw_ids: list[object] = []
    for shard_path in shard_paths:
        with open_shard(shard_path) as handle:
            table = read_table(
                handle,
                columns=[id_field],
                use_threads=False,
                pre_buffer=False,
            )
        if list(getattr(table, "column_names", ())) != [id_field]:
            raise SampleProtocolError("Parquet projection returned an unapproved column")
        column = table[id_field]
        to_pylist = getattr(column, "to_pylist", None)
        if not callable(to_pylist):
            raise SampleProtocolError("projected Parquet ID column cannot be enumerated")
        values = tuple(to_pylist())
        if len(values) != int(getattr(table, "num_rows", -1)):
            raise SampleProtocolError("projected Parquet shard row count differs")
        raw_ids.extend(values)
    return tuple(raw_ids)


def _load_projected_id_dataset(
    *,
    dataset_id: str,
    dataset_config: str,
    dataset_revision: str,
    split: str,
    cache_dir: Path,
    id_field: str,
    parquet_glob: str,
    expected_shard_count: int,
) -> Any:
    del dataset_config, split, cache_dir
    from huggingface_hub import HfApi, HfFileSystem
    from pyarrow import parquet as pq

    token = os.environ.get("HF_TOKEN") or False
    repo_files = HfApi(token=token).list_repo_files(
        dataset_id,
        repo_type="dataset",
        revision=dataset_revision,
    )
    shards = _select_pinned_parquet_shards(
        repo_files=repo_files,
        parquet_glob=parquet_glob,
        expected_shard_count=expected_shard_count,
    )
    filesystem = HfFileSystem(token=token)
    remote_root = f"datasets/{dataset_id}@{dataset_revision}"

    def open_shard(path: str) -> Any:
        return filesystem.open(f"{remote_root}/{path}", mode="rb", block_size=1 << 20)

    raw_ids = _read_projected_parquet_ids(
        shard_paths=shards,
        id_field=id_field,
        open_shard=open_shard,
        read_table=pq.read_table,
    )
    return _ProjectedIdDataset(id_field=id_field, raw_ids=raw_ids)


def _catalog_authorization(
    protocol: dict[str, Any], *, task: dict[str, Any], output: Path
) -> dict[str, Any]:
    authorization = cast("dict[str, Any]", protocol["candidate_catalog_construction_authorization"])
    catalogs = cast("list[dict[str, Any]]", authorization.get("catalogs"))
    matches = [record for record in catalogs if record.get("task") == task.get("task")]
    if len(matches) != 1:
        raise ValueError("V102 has no unique catalog authorization for the selected task")
    record = matches[0]
    pinned_fields = (
        "dataset_id",
        "dataset_config",
        "dataset_revision",
        "split",
    )
    if any(record.get(field) != task.get(field) for field in pinned_fields):
        raise ValueError("candidate-catalog authorization differs from the task contract")
    if record.get("expected_id_count") != task.get("candidate_catalog_expected_id_count"):
        raise ValueError("candidate-catalog cardinality authorization differs")
    expected_output = Path(str(record.get("output_path", ""))).resolve()
    if output.resolve() != expected_output:
        raise PermissionError(
            f"catalog output must equal the authorized write-once path: {expected_output}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    id_field = record.get("id_field")
    if not isinstance(id_field, str) or id_field != task.get("candidate_catalog_id_field"):
        raise ValueError("candidate-catalog ID field is absent or differs from the task contract")
    if record.get("parquet_glob") != task.get("candidate_catalog_parquet_glob"):
        raise ValueError("candidate-catalog Parquet glob differs from the task contract")
    if record.get("expected_shard_count") != task.get("candidate_catalog_expected_shard_count"):
        raise ValueError("candidate-catalog shard count differs from the task contract")
    return record


def construct_catalog_for_task(
    *,
    protocol: dict[str, Any],
    task_name: str,
    output: Path,
    git_sha: str,
    git_dirty: bool,
    dataset_loader: Callable[..., Any] = _load_projected_id_dataset,
) -> dict[str, Any]:
    """Build one authorized ID-only payload without writing it."""

    require_protocol_authorization(protocol, "catalog")
    if git_dirty or re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise SampleProtocolError("canonical catalog construction requires a clean full Git SHA")
    task, contract = _task_contract(protocol, task_name)
    authorization = _catalog_authorization(protocol, task=task, output=output)
    cache_dir = _require_scratch_cache_environment()
    dataset = dataset_loader(
        dataset_id=contract.dataset_id,
        dataset_config=str(contract.dataset_config),
        dataset_revision=contract.dataset_revision,
        split=contract.split,
        cache_dir=cache_dir,
        id_field=str(authorization["id_field"]),
        parquet_glob=str(authorization["parquet_glob"]),
        expected_shard_count=int(authorization["expected_shard_count"]),
    )
    return construct_id_only_catalog_from_dataset(
        dataset=dataset,
        contract=contract,
        id_field=str(authorization["id_field"]),
        expected_count=int(authorization["expected_id_count"]),
        construction_git_sha=git_sha,
        construction_git_dirty=git_dirty,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)
    scan = subparsers.add_parser("scan")
    scan.add_argument("--output", required=True, type=Path)
    catalog = subparsers.add_parser("catalog")
    catalog.add_argument("--task", required=True)
    catalog.add_argument("--output", required=True, type=Path)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--task", required=True)
    materialize.add_argument("--catalog", required=True, type=Path)
    materialize.add_argument("--exclusion-manifest", required=True, type=Path)
    materialize.add_argument("--output", required=True, type=Path)
    check = subparsers.add_parser("check")
    check.add_argument("--exclusion-manifest", required=True, type=Path)
    check.add_argument("--ledger", required=True, action="append", type=Path)
    check.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    repository = package_root.parent
    spec_path = args.spec.resolve()
    protocol = _load_protocol(spec_path)
    action = str(args.action)
    # Authorization is deliberately checked before Git, filesystem, or dataset work.
    require_protocol_authorization(protocol, action)
    if action == "catalog":
        _require_canonical_spec_path(spec_path, package_root=package_root)
    git_sha, git_dirty = _require_execution_snapshot(protocol, repository=repository, action=action)
    if action == "check":
        _require_disjointness_authorization(
            protocol,
            exclusion_manifest=Path(str(args.exclusion_manifest)),
            ledger_paths=tuple(Path(str(path)) for path in args.ledger),
            output=Path(str(args.output)),
        )

    if action == "scan":
        exclusion = cast("dict[str, Any]", protocol["required_exclusion_snapshot"])
        roots = tuple(
            _resolve_protocol_path(package_root, str(value))
            for value in cast("list[str]", exclusion["recursively_scan_roots"])
        )
        ignored = {
            Path(str(args.output)).resolve(),
            _resolve_protocol_path(package_root, str(exclusion["disjointness_report_path"])),
            *(
                _resolve_protocol_path(package_root, str(task["ledger_path"]))
                for task in cast("list[dict[str, Any]]", protocol["tasks"])
            ),
        }
        payload = scan_exclusion_sources(
            roots=roots,
            contracts=_contracts(protocol),
            source_git_sha=git_sha,
            source_git_dirty=git_dirty,
            ignored_paths=ignored,
        )
        artifact_sha = write_once(Path(str(args.output)).resolve(), payload)
        print(json.dumps({"status": "created", "sha256": artifact_sha}, sort_keys=True))
        return 0

    if action == "catalog":
        output = Path(str(args.output)).resolve()
        payload = construct_catalog_for_task(
            protocol=protocol,
            task_name=str(args.task),
            output=output,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        final_sha, final_dirty = _git_provenance(repository)
        if (final_sha, final_dirty) != (git_sha, git_dirty):
            raise RuntimeError("source snapshot changed during candidate-catalog construction")
        artifact_sha = write_once(output, payload)
        print(
            json.dumps(
                {
                    "id_count": len(cast("list[str]", payload["sample_ids"])),
                    "sha256": artifact_sha,
                    "status": "created",
                },
                sort_keys=True,
            )
        )
        return 0

    if action == "materialize":
        task, contract = _task_contract(protocol, str(args.task))
        rule = cast("dict[str, Any]", protocol["future_materialization_rule"])
        source = cast("dict[str, Any]", protocol["candidate_source"])
        payload = materialize_deterministic_ledger(
            catalog_path=Path(str(args.catalog)).resolve(),
            exclusion_manifest_path=Path(str(args.exclusion_manifest)).resolve(),
            contract=contract,
            selection_seed=int(task["selection_seed"]),
            selection_salt=str(rule["hash_preimage"]).split("_null_", maxsplit=1)[0],
            sample_count=int(task["sample_count"]),
            candidate_implementation_git_sha=str(source["candidate_implementation_git_sha"]),
            materialization_git_sha=git_sha,
            materialization_git_dirty=git_dirty,
            protocol_sha256=sha256_file(spec_path),
        )
        artifact_sha = write_once(Path(str(args.output)).resolve(), payload)
        print(json.dumps({"status": "created", "sha256": artifact_sha}, sort_keys=True))
        return 0

    report = check_disjointness(
        exclusion_manifest_path=Path(str(args.exclusion_manifest)).resolve(),
        ledger_paths=tuple(Path(str(path)).resolve() for path in args.ledger),
        validation_git_sha=git_sha,
        validation_git_dirty=git_dirty,
    )
    artifact_sha = write_once(Path(str(args.output)).resolve(), report)
    print(json.dumps({"status": report["status"], "sha256": artifact_sha}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
