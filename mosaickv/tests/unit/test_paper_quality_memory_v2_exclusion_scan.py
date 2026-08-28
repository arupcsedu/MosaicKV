from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml  # type: ignore[import-untyped]

from mosaickv.evaluation.disjoint_samples import DatasetContract, SampleProtocolError
from mosaickv.evaluation.identity_exclusion_scan import (
    scan_identity_sources,
    write_once_exclusive,
)

TARGET = DatasetContract(
    task="realworldqa",
    dataset_id="lmms-lab/RealWorldQA",
    dataset_config="default",
    dataset_revision="907c4e5228fd1703c710ed937601cb5f89ab8d5c",
    split="test",
)


def _scanner(package: Path) -> ModuleType:
    path = package / "scripts/scan_paper_quality_memory_v2_exclusions.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v2_scan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_id": TARGET.dataset_id,
                "dataset_config": TARGET.dataset_config,
                "dataset_revision": TARGET.dataset_revision,
                "split": TARGET.split,
            }
        ),
        encoding="utf-8",
    )


def test_identity_scan_projects_ledgers_results_and_parquet(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    ledger_root = tmp_path / "ledgers"
    run_root = tmp_path / "runs"
    result_root = run_root / "run-a"
    ledger_root.mkdir()
    result_root.mkdir(parents=True)
    (ledger_root / "selected.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset_id": TARGET.dataset_id,
                "dataset_config": TARGET.dataset_config,
                "dataset_revision": TARGET.dataset_revision,
                "split": TARGET.split,
                "sample_ids": ["realworldqa:2", "realworldqa:1"],
            }
        ),
        encoding="utf-8",
    )
    _manifest(result_root / "manifest.json")
    (result_root / "raw.jsonl").write_text(
        json.dumps(
            {
                "task": "realworldqa",
                "sample_id": "realworldqa:3",
                "answer": "must-not-enter-manifest",
                "reference": "must-not-enter-manifest",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    parquet.write_table(
        pyarrow.table(
            {
                "dataset_id": [TARGET.dataset_id],
                "dataset_config": [TARGET.dataset_config],
                "dataset_revision": [TARGET.dataset_revision],
                "split": [TARGET.split],
                "task": [TARGET.task],
                "sample_id": ["realworldqa:4"],
                "answer": ["not-projected"],
            }
        ),
        result_root / "aggregate.parquet",
    )
    (run_root / "candidate_catalog.json").write_text("not valid json", encoding="utf-8")

    payload = scan_identity_sources(
        roots=(ledger_root, run_root),
        target=TARGET,
        source_git_sha="a" * 40,
        source_git_dirty=False,
    )

    assert [row["sample_id"] for row in payload["excluded_identities"]] == [
        "realworldqa:1",
        "realworldqa:2",
        "realworldqa:3",
        "realworldqa:4",
    ]
    assert payload["excluded_unique_sample_count"] == 4
    assert payload["forbidden_content_fields_accessed"] is False
    encoded = json.dumps(payload)
    assert "must-not-enter-manifest" not in encoded
    assert "not-projected" not in encoded
    assert "candidate_catalog" not in encoded
    assert all(len(source["sha256"]) == 64 for source in payload["sources"])


@pytest.mark.parametrize(
    "record",
    (
        {"task": "realworldqa", "sample_id": "realworldqa:1"},
        {
            "dataset_id": TARGET.dataset_id,
            "dataset_config": TARGET.dataset_config,
            "dataset_revision": "0" * 40,
            "split": TARGET.split,
            "sample_id": "realworldqa:1",
        },
    ),
)
def test_identity_scan_fails_closed_on_incomplete_or_conflicting_provenance(
    tmp_path: Path,
    record: dict[str, object],
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    (root / "raw.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(SampleProtocolError, match="provenance"):
        scan_identity_sources(
            roots=(root,),
            target=TARGET,
            source_git_sha="a" * 40,
            source_git_dirty=False,
        )


def test_write_once_exclusive_is_read_only_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "inputs/exclusion.json"
    first = write_once_exclusive(output, {"schema_version": 2})

    assert len(first) == 64
    assert output.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError, match="overwrite"):
        write_once_exclusive(output, {"schema_version": 3})


def test_canonical_scope_cannot_authorize_execution(tmp_path: Path) -> None:
    package = Path(__file__).parents[2]
    scanner = _scanner(package)
    scope = scanner.load_canonical_scope(package)
    output = tmp_path / "exclusion.json"
    scanner.__dict__["EXPECTED_OUTPUT"] = output
    fake_package = tmp_path / "package"
    authorization = fake_package / "configs/ledgers/authorization.yaml"
    authorization.parent.mkdir(parents=True)
    authorization.write_text(
        yaml.safe_dump(
            {
                "record_type": scanner.AUTHORIZATION_RECORD_TYPE,
                "scope_commit_git_sha": scanner.SCOPE_COMMIT,
                "scope_file_sha256": scanner.SCOPE_SHA256,
                "output": str(output),
                "reviewed_by_user": False,
                "authorization": {"exclusion_scan_execution_authorized": False},
            }
        ),
        encoding="utf-8",
    )

    assert scope["authorization"]["exclusion_scan_execution_authorized"] is False
    with pytest.raises(PermissionError, match="not explicitly authorized"):
        scanner.require_execution_authorization(
            authorization_path=authorization,
            scope=scope,
            output=output,
            package_root=fake_package,
        )
    assert not output.exists()
