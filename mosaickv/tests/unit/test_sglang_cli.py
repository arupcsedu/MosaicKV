from __future__ import annotations

from pathlib import Path

from mosaickv.cli import build_parser, main
from mosaickv.sglang_cli import resolve_sglang_config
from mosaickv.types import Backend


def test_sglang_cli_options_parse() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--backend",
            "sglang",
            "--model",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--model-revision",
            "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "--dataset-revision",
            "schema-v1",
            "--task",
            "synthetic_smoke",
            "--attention-backend",
            "triton",
            "--sglang-mem-fraction-static",
            "0.75",
            "--sglang-port",
            "31000",
            "--cache-probe-repeats",
            "3",
        ]
    )
    config = resolve_sglang_config(args)
    assert config is not None
    assert config.execution.backend is Backend.SGLANG
    assert config.execution.attention_implementation == "triton"
    assert args.sglang_mem_fraction_static == 0.75
    assert args.sglang_port == 31000
    assert args.cache_probe_repeats == 3


def test_sglang_yaml_resolves_without_server_import() -> None:
    root = Path(__file__).parents[2]
    args = build_parser().parse_args(
        ["evaluate", "--config", str(root / "configs" / "sglang_fullkv_3b.yaml")]
    )
    config = resolve_sglang_config(args)
    assert config is not None
    assert config.execution.backend is Backend.SGLANG


def test_native_flag_fails_closed_before_cache_or_weight_loading() -> None:
    status = main(
        [
            "evaluate",
            "--backend",
            "sglang",
            "--model",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--model-revision",
            "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "--dataset-revision",
            "schema-v1",
            "--task",
            "synthetic_smoke",
            "--attention-backend",
            "triton",
            "--method",
            "mosaickv_exact",
            "--retention-ratio",
            "0.5",
            "--enable-mosaickv",
        ]
    )
    assert status == 2
