from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from mosaickv.cli import build_parser, main
from mosaickv.vllm_cli import _resolve_sample_selection, resolve_vllm_config


def test_evaluate_parser_accepts_vllm_measurement_controls() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--backend",
            "vllm",
            "--model",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--task",
            "synthetic_smoke",
            "--enable-mosaickv",
            "--cache-probe-repeats",
            "3",
            "--enable-vllm-prefix-caching",
        ]
    )
    assert args.backend == "vllm"
    assert args.enable_mosaickv
    assert args.cache_probe_repeats == 3
    assert args.enable_vllm_prefix_caching


def test_native_flag_fails_closed_before_cache_or_weight_loading(capsys: object) -> None:
    status = main(
        [
            "evaluate",
            "--backend",
            "vllm",
            "--model",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--model-revision",
            "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "--dataset-revision",
            "schema-v1",
            "--task",
            "synthetic_smoke",
            "--attention-backend",
            "sdpa",
            "--method",
            "mosaickv_exact",
            "--retention-ratio",
            "0.5",
            "--enable-mosaickv",
        ]
    )
    assert status == 2


def _llava_args(*extra: str) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "evaluate",
            "--backend",
            "vllm",
            "--model",
            "llava-hf/llava-1.5-7b-hf",
            "--model-revision",
            "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            "--dataset-revision",
            "bc98d668301da7b14f648724866e57302778ab27",
            "--task",
            "mmstar",
            *extra,
        ]
    )


def test_vllm_selection_forwards_subset_offset_and_defaults() -> None:
    args = _llava_args("--subset-size", "3", "--sample-offset", "7")
    config = resolve_vllm_config(args)
    assert config is not None

    selection = _resolve_sample_selection(
        args,
        config=config,
        task_name="mmstar",
        local_sample_count=None,
    )

    assert selection.subset_size == 3
    assert selection.sample_offset == 7
    assert selection.explicit_sample_ids is None
    assert selection.expected_rows == 3


def test_vllm_selection_accepts_hash_bound_explicit_sample_ids(tmp_path: Path) -> None:
    sample_file = tmp_path / "samples.json"
    sample_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task": "mmstar",
                "dataset_id": "Lin-Chen/MMStar",
                "dataset_revision": "bc98d668301da7b14f648724866e57302778ab27",
                "split": "val",
                "sample_count": 2,
                "sample_ids": ["mmstar:1", "mmstar:2"],
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(sample_file.read_bytes()).hexdigest()
    args = _llava_args("--sample-id-file", str(sample_file), "--sample-id-file-sha", digest)
    config = resolve_vllm_config(args)
    assert config is not None

    selection = _resolve_sample_selection(
        args,
        config=config,
        task_name="mmstar",
        local_sample_count=None,
    )

    assert selection.subset_size is None
    assert selection.explicit_sample_ids == ("mmstar:1", "mmstar:2")
    assert selection.expected_rows == 2


def test_vllm_selection_rejects_conflicting_full_dataset_controls() -> None:
    args = _llava_args("--full-dataset", "--subset-size", "2")
    config = resolve_vllm_config(args)
    assert config is not None
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_sample_selection(
            args,
            config=config,
            task_name="mmstar",
            local_sample_count=None,
        )
