from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from mosaickv.cli import build_parser
from mosaickv.config import synthetic_smoke_config
from mosaickv.evaluation.paper_native_lanes import resolve_protocol
from mosaickv.evaluation.published_protocols import (
    PublishedBaselineProtocol,
    load_published_protocols,
)
from mosaickv.hf_cli import (
    _direct_config,
    _load_sample_id_file,
    _model_kwargs,
    _paper_native_runtime_profile,
)
from mosaickv.types import BudgetUnit


def test_common_transformers_lock_uses_torch_dtype_keyword() -> None:
    kwargs = _model_kwargs(synthetic_smoke_config(), local_files_only=True)

    assert "torch_dtype" in kwargs
    assert "dtype" not in kwargs
    assert kwargs["local_files_only"] is True


def _paper_native_args(*, method: str, maximum_tokens: int = 16) -> Namespace:
    arguments = [
        "evaluate",
        "--task",
        "llava_description",
        "--model",
        "llava-hf/llava-1.5-7b-hf",
        "--model-revision",
        "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
        "--dataset-revision",
        "407c2d2f8fc13611d13f72a5e18b8c33bf9c44e6e05278bf0b4da41deeacbe6a",
        "--method",
        method,
        "--retention-ratio",
        "0.2",
        "--block-size",
        "1",
        "--max-new-tokens",
        str(maximum_tokens),
    ]
    if method == "prefixkv_reimpl":
        arguments.extend(("--prefixkv-profile-mode", "fixed_global"))
    return build_parser().parse_args(arguments)


def _prefix_protocol() -> PublishedBaselineProtocol:
    root = Path(__file__).resolve().parents[2]
    suite = load_published_protocols(
        root / "configs/experiments/published_paper_native_metrics_v1.yaml"
    )
    return resolve_protocol(suite, "prefixkv_hf_reimpl_native_metrics_v1")


def test_prefix_native_profile_covers_observed_teacher_forcing_target() -> None:
    config = _direct_config(_paper_native_args(method="prefixkv_reimpl"))
    profiled = _paper_native_runtime_profile(config, _prefix_protocol())

    assert profiled.generation.max_new_tokens == 144
    assert profiled.cache.budget_unit is BudgetUnit.BLOCKS


def test_block_one_mosaickv_profile_bounds_graph_and_dense_physical_capacity() -> None:
    config = _direct_config(_paper_native_args(method="mosaickv_exact"))
    profiled = _paper_native_runtime_profile(config, _prefix_protocol())

    assert profiled.generation.max_new_tokens == 144
    assert profiled.cache.block_size == 1
    assert profiled.cache.budget_unit is BudgetUnit.BYTES
    assert profiled.graph.candidate_graph_mode == "shortlist_rerank"
    assert profiled.graph.candidate_descriptor_pooling
    assert profiled.graph.uncertain_head_fraction == pytest.approx(0.015625)
    assert profiled.graph.candidate_nonmandatory_fraction == pytest.approx(0.03)
    assert profiled.selection.routed_head_only_planning
    assert profiled.selection.stable_head_prompt_routing
    assert profiled.selection.stable_head_uniform_fill
    assert profiled.selection.physical_capacity_fraction == pytest.approx(0.69)
    assert profiled.prototypes.uncertain_head_only
    assert profiled.prototypes.uncertain_head_fraction == pytest.approx(0.015625)


def test_sample_id_file_requires_exact_content_and_sha(tmp_path: Path) -> None:
    config = synthetic_smoke_config()
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "task": "synthetic_ci",
                "dataset_id": config.dataset.id,
                "dataset_revision": config.dataset.revision,
                "split": config.dataset.split,
                "sample_ids": ["synthetic:1", "synthetic:2"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    path = tmp_path / "shard.json"
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    assert _load_sample_id_file(str(path), sha, task_name="synthetic_ci", config=config) == (
        "synthetic:1",
        "synthetic:2",
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        _load_sample_id_file(str(path), "0" * 64, task_name="synthetic_ci", config=config)


def test_sample_id_file_accepts_reviewed_schema_version_two_ledger(tmp_path: Path) -> None:
    config = synthetic_smoke_config()
    record = {
        "schema_version": 2,
        "classification": "fresh_disjoint_label_blind_calibration",
        "task": "synthetic_ci",
        "dataset_id": config.dataset.id,
        "dataset_revision": config.dataset.revision,
        "split": config.dataset.split,
        "sample_count": 2,
        "sample_ids": ["synthetic:1", "synthetic:2"],
    }
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = tmp_path / "reviewed-ledger.json"
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    assert _load_sample_id_file(str(path), sha, task_name="synthetic_ci", config=config) == (
        "synthetic:1",
        "synthetic:2",
    )

    record["sample_count"] = 1
    invalid = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(invalid)
    with pytest.raises(ValueError, match="inconsistent sample_count"):
        _load_sample_id_file(
            str(path),
            hashlib.sha256(invalid).hexdigest(),
            task_name="synthetic_ci",
            config=config,
        )
