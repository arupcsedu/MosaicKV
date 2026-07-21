from __future__ import annotations

from mosaickv.cli import build_parser, main


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
        ]
    )
    assert args.backend == "vllm"
    assert args.enable_mosaickv
    assert args.cache_probe_repeats == 3


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
            "--method",
            "mosaickv_exact",
            "--retention-ratio",
            "0.5",
            "--enable-mosaickv",
        ]
    )
    assert status == 2
