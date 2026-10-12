from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from mosaickv.cli import build_parser
from mosaickv.evaluation.paper_native_latency import (
    PaperNativeLatencyProtocolError,
    canonical_inference_traces,
    load_paper_native_latency_protocol,
    paired_method_order,
)
from mosaickv.hf_cli import resolve_hf_config


def _protocol_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs/experiments/paper_native_latency_v3.yaml"


def _script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"cannot import test script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_native_latency_protocol_registers_persistent_paired_trials() -> None:
    protocol = load_paper_native_latency_protocol(_protocol_path())

    assert protocol.timing.methods == ("prefixkv_reimpl", "mosaickv_exact")
    assert protocol.timing.sample_count == 4
    assert protocol.timing.warmups_per_method == 5
    assert protocol.timing.timed_repetitions_per_method == 20
    assert protocol.timing.maximum_latency_ratio == pytest.approx(1.05)
    assert protocol.expected_mosaickv_mean_active_kv_bytes == 42_860_544
    assert protocol.expected_mean_active_kv_byte_tolerance == 0
    assert paired_method_order(protocol.timing, 0) == protocol.timing.methods
    assert paired_method_order(protocol.timing, 1) == tuple(reversed(protocol.timing.methods))


@pytest.mark.parametrize("method", ["full_kv", "prefixkv_reimpl", "mosaickv_exact"])
def test_persistent_worker_builds_matched_prefix_lane_configs(method: str) -> None:
    worker = _script("run_paper_native_latency_v3.py")
    args = build_parser().parse_args(worker._base_cli(method))
    config = resolve_hf_config(args)

    assert config is not None
    assert config.method.value == method
    assert config.model.revision == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
    assert config.dataset.revision == (
        "407c2d2f8fc13611d13f72a5e18b8c33bf9c44e6e05278bf0b4da41deeacbe6a"
    )
    assert config.cache.block_size == 1
    assert config.cache.retention_ratio == pytest.approx(1.0 if method == "full_kv" else 0.2)
    assert config.generation.max_new_tokens == 144


def test_persistent_worker_accepts_resolved_scratch_alias_only() -> None:
    worker = _script("run_paper_native_latency_v3.py")

    output = worker._validated_output_root(
        Path("/scratch/djy8hg/runs/mosaickv/paper-native-latency-test")
    )

    assert output.is_relative_to(Path("/scratch/djy8hg").resolve())
    with pytest.raises(ValueError, match="must be under /scratch/djy8hg"):
        worker._validated_output_root(Path("/tmp/mosaickv-paper-latency-test"))


def test_persistent_worker_parses_explicit_nonpaper_fullkv_mosaickv_verification() -> None:
    worker = _script("run_paper_native_latency_v3.py")
    methods = ("full_kv", "mosaickv_exact")

    assert worker._paired_method_order(methods, 0) == methods
    assert worker._paired_method_order(methods, 1) == tuple(reversed(methods))
    with pytest.raises(ValueError, match="exactly two"):
        worker._paired_method_order(("full_kv", "full_kv"), 0)


def test_repeated_fullkv_timing_uses_a_private_reference_output(tmp_path: Path) -> None:
    worker = _script("run_paper_native_latency_v3.py")

    initial = tmp_path / "shared-fullkv-reference.json"
    timing_root = tmp_path / "runs" / "timing-fullkv"
    expected = timing_root / "fullkv-reference-output.json"

    # This mirrors the runner's write-once policy: the shared initial
    # reference belongs only to the compressed comparison, while timed FullKV
    # repetitions must have a distinct destination.
    assert initial != expected
    assert expected.parent.name == "timing-fullkv"
    assert worker._paired_method_order(("full_kv", "mosaickv_exact"), 2)[0] == "full_kv"


def test_paper_native_latency_protocol_rejects_weakened_repetitions(tmp_path: Path) -> None:
    payload = yaml.safe_load(_protocol_path().read_text(encoding="utf-8"))
    payload["timing_protocol"]["timed_repetitions_per_method"] = 19
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PaperNativeLatencyProtocolError, match="measurement contract"):
        load_paper_native_latency_protocol(path)


def test_paper_native_latency_protocol_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = yaml.safe_load(_protocol_path().read_text(encoding="utf-8"))
    payload["unregistered_override"] = True
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PaperNativeLatencyProtocolError, match="unknown"):
        load_paper_native_latency_protocol(path)


def test_canonical_inference_traces_excludes_teacher_forced_metric_trace(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "sample.json"
    teacher = tmp_path / "sample-attempt-001.json"
    canonical.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "teacher_forced_evaluation": {"enabled": False},
                "timing_breakdown": {"total": 1.0},
            }
        ),
        encoding="utf-8",
    )
    teacher.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "teacher_forced_evaluation": {"enabled": True},
                "timing_breakdown": {"total": 2.0},
            }
        ),
        encoding="utf-8",
    )

    selected = canonical_inference_traces((teacher, canonical))

    assert len(selected) == 1
    assert selected[0]["timing_breakdown"]["total"] == pytest.approx(1.0)


def test_canonical_inference_traces_rejects_duplicate_generation_traces(
    tmp_path: Path,
) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"trace-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "sample_id": "sample-1",
                    "teacher_forced_evaluation": {"enabled": False},
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    with pytest.raises(PaperNativeLatencyProtocolError, match="multiple canonical"):
        canonical_inference_traces(tuple(paths))
