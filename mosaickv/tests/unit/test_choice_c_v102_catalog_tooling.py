from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _wrapper(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v102_sample_tooling.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v102_catalog_tooling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _SyntheticDataset:
    def __init__(self, accesses: list[object], *, projected: bool = False) -> None:
        self._accesses = accesses
        self._projected = projected
        self.column_names = (
            ["question_id"] if projected else ["question_id", "question", "answers", "image"]
        )

    def __len__(self) -> int:
        return 3

    def select_columns(self, columns: list[str]) -> _SyntheticDataset:
        self._accesses.append(("select_columns", tuple(columns)))
        assert columns == ["question_id"]
        return _SyntheticDataset(self._accesses, projected=True)

    def __getitem__(self, key: str) -> list[int]:
        self._accesses.append(("column", key))
        assert self._projected and key == "question_id"
        return [3, 1, 2]


def _protocol(output: Path) -> dict:
    return {
        "experiment_id": "choice_c_v102_v100_fresh_cross_dataset_calibration",
        "candidate_catalog_construction_authorized": True,
        "tasks": [
            {
                "task": "synthetic_id_catalog",
                "dataset_id": "synthetic/catalog",
                "dataset_config": "default",
                "dataset_revision": "1" * 40,
                "split": "validation",
                "candidate_catalog_expected_id_count": 3,
                "candidate_catalog_id_field": "question_id",
                "candidate_catalog_parquet_glob": "data/validation-*.parquet",
                "candidate_catalog_expected_shard_count": 2,
            }
        ],
        "candidate_catalog_construction_authorization": {
            "catalogs": [
                {
                    "task": "synthetic_id_catalog",
                    "dataset_id": "synthetic/catalog",
                    "dataset_config": "default",
                    "dataset_revision": "1" * 40,
                    "split": "validation",
                    "expected_id_count": 3,
                    "id_field": "question_id",
                    "parquet_glob": "data/validation-*.parquet",
                    "expected_shard_count": 2,
                    "output_path": str(output),
                }
            ]
        },
    }


def _scratch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    root = "/scratch/djy8hg/cache/mosaickv-v102-synthetic-test"
    values = {
        "MOSAICKV_CACHE_ROOT": root,
        "PIP_CACHE_DIR": f"{root}/pip",
        "XDG_CACHE_HOME": f"{root}/xdg",
        "HF_HOME": f"{root}/huggingface",
        "HF_HUB_CACHE": f"{root}/huggingface/hub",
        "HF_DATASETS_CACHE": f"{root}/datasets",
        "TRANSFORMERS_CACHE": f"{root}/transformers",
        "TORCH_HOME": f"{root}/torch",
        "TMPDIR": f"{root}/tmp",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_v102_catalog_wrapper_passes_only_pinned_id_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = Path(__file__).parents[2]
    wrapper = _wrapper(package)
    output = tmp_path / "catalog.json"
    accesses: list[object] = []
    loader_calls: list[dict[str, object]] = []

    def loader(**kwargs: object) -> _SyntheticDataset:
        loader_calls.append(kwargs)
        return _SyntheticDataset(accesses)

    _scratch_environment(monkeypatch)
    payload = wrapper.construct_catalog_for_task(
        protocol=_protocol(output),
        task_name="synthetic_id_catalog",
        output=output,
        git_sha="a" * 40,
        git_dirty=False,
        dataset_loader=loader,
    )

    assert payload["sample_ids"] == [
        "synthetic_id_catalog:1",
        "synthetic_id_catalog:2",
        "synthetic_id_catalog:3",
    ]
    assert set(payload) == {
        "schema_version",
        "dataset_id",
        "dataset_config",
        "dataset_revision",
        "split",
        "sample_ids",
    }
    assert loader_calls == [
        {
            "dataset_id": "synthetic/catalog",
            "dataset_config": "default",
            "dataset_revision": "1" * 40,
            "split": "validation",
            "cache_dir": Path(
                "/scratch/djy8hg/cache/mosaickv-v102-synthetic-test/datasets"
            ).resolve(),
            "id_field": "question_id",
            "parquet_glob": "data/validation-*.parquet",
            "expected_shard_count": 2,
        }
    ]
    assert accesses == [("select_columns", ("question_id",)), ("column", "question_id")]
    assert not output.exists()


def test_v102_catalog_wrapper_fails_before_dataset_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = Path(__file__).parents[2]
    wrapper = _wrapper(package)
    output = tmp_path / "catalog.json"
    loader_called = False

    def loader(**_kwargs: object) -> _SyntheticDataset:
        nonlocal loader_called
        loader_called = True
        return _SyntheticDataset([])

    _scratch_environment(monkeypatch)
    with pytest.raises(PermissionError, match="authorized write-once path"):
        wrapper.construct_catalog_for_task(
            protocol=_protocol(output),
            task_name="synthetic_id_catalog",
            output=tmp_path / "wrong.json",
            git_sha="a" * 40,
            git_dirty=False,
            dataset_loader=loader,
        )
    assert loader_called is False

    with pytest.raises(ValueError, match="clean full Git SHA"):
        wrapper.construct_catalog_for_task(
            protocol=_protocol(output),
            task_name="synthetic_id_catalog",
            output=output,
            git_sha="a" * 40,
            git_dirty=True,
            dataset_loader=loader,
        )
    assert loader_called is False


def test_projected_loader_selects_only_requested_split_and_id_column() -> None:
    package = Path(__file__).parents[2]
    wrapper = _wrapper(package)
    repo_files = [
        "data/test-00000-of-00001.parquet",
        "data/validation-00001-of-00002.parquet",
        "README.md",
        "data/validation-00000-of-00002.parquet",
        "data/train-00000-of-00001.parquet",
    ]

    shards = wrapper._select_pinned_parquet_shards(
        repo_files=repo_files,
        parquet_glob="data/validation-*.parquet",
        expected_shard_count=2,
    )

    opened: list[str] = []
    reads: list[dict[str, object]] = []

    class Column:
        def __init__(self, values: list[int]) -> None:
            self._values = values

        def to_pylist(self) -> list[int]:
            return self._values

    class Table:
        def __init__(self, values: list[int]) -> None:
            self._values = values
            self.num_rows = len(values)
            self.column_names = ["question_id"]

        def __getitem__(self, key: str) -> Column:
            assert key == "question_id"
            return Column(self._values)

    def open_shard(path: str) -> io.BytesIO:
        opened.append(path)
        return io.BytesIO(path.encode())

    def read_table(_handle: object, **kwargs: object) -> Table:
        reads.append(kwargs)
        return Table([len(reads) * 10 + 1, len(reads) * 10 + 2])

    raw_ids = wrapper._read_projected_parquet_ids(
        shard_paths=shards,
        id_field="question_id",
        open_shard=open_shard,
        read_table=read_table,
    )

    assert shards == (
        "data/validation-00000-of-00002.parquet",
        "data/validation-00001-of-00002.parquet",
    )
    assert opened == list(shards)
    assert raw_ids == (11, 12, 21, 22)
    assert reads == [
        {"columns": ["question_id"], "use_threads": False, "pre_buffer": False},
        {"columns": ["question_id"], "use_threads": False, "pre_buffer": False},
    ]


def test_projected_loader_fails_closed_on_shard_drift() -> None:
    package = Path(__file__).parents[2]
    wrapper = _wrapper(package)

    with pytest.raises(wrapper.SampleProtocolError, match="matched 1 shards; expected 2"):
        wrapper._select_pinned_parquet_shards(
            repo_files=["data/validation-00000-of-00001.parquet"],
            parquet_glob="data/validation-*.parquet",
            expected_shard_count=2,
        )


def test_v102_catalog_wrapper_rejects_home_cache_before_dataset_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = Path(__file__).parents[2]
    wrapper = _wrapper(package)
    output = tmp_path / "catalog.json"
    loader_called = False

    def loader(**_kwargs: object) -> _SyntheticDataset:
        nonlocal loader_called
        loader_called = True
        return _SyntheticDataset([])

    _scratch_environment(monkeypatch)
    monkeypatch.setenv("HF_HOME", "/home/test/.cache/huggingface")
    with pytest.raises(RuntimeError, match="HF_HOME must resolve below"):
        wrapper.construct_catalog_for_task(
            protocol=_protocol(output),
            task_name="synthetic_id_catalog",
            output=output,
            git_sha="a" * 40,
            git_dirty=False,
            dataset_loader=loader,
        )
    assert loader_called is False
