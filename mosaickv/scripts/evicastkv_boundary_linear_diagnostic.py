#!/usr/bin/env python3
"""Isolate shape-dependent layer-zero query projection on two consumed inputs.

The model runs only with its original native FP16 arithmetic. Alternative GEMM
settings apply to isolated operator replays, never to a model forward. This is
not a precision repair, quality experiment, timing benchmark, or new cohort.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import evicastkv_boundary_operators as boundary
from evicastkv_campaign import SOURCE, environment, resolved_config

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

STAGE = "boundary_linear_diagnostic"
CAPTURE_LIMIT = 128 * 2**20
ORIGINAL_FLAGS = {
    "allow_fp16_reduced_precision_reduction": True,
    "allow_fp16_accumulation": False,
    "allow_tf32": False,
}
VARIANTS = (
    ("native_fp16", ORIGINAL_FLAGS),
    (
        "fp16_no_reduced_reduction",
        {**ORIGINAL_FLAGS, "allow_fp16_reduced_precision_reduction": False},
    ),
    ("fp16_accumulation_optional", {**ORIGINAL_FLAGS, "allow_fp16_accumulation": True}),
)


def arithmetic_flags(torch):
    return {key: getattr(torch.backends.cuda.matmul, key) for key in ORIGINAL_FLAGS}


@contextmanager
def isolated_flags(torch, requested):
    """Restore all original flags even if a diagnostic kernel is unsupported."""
    original = arithmetic_flags(torch)
    if set(requested) != set(ORIGINAL_FLAGS) or requested["allow_tf32"] is not False:
        raise ValueError("only the registered complete, no-TF32 settings are permitted")
    try:
        for key, value in requested.items():
            setattr(torch.backends.cuda.matmul, key, value)
        yield arithmetic_flags(torch)
    finally:
        for key, value in original.items():
            setattr(torch.backends.cuda.matmul, key, value)


class LinearInputCapture(boundary.FinalPositionCapture):
    """Save actual complete normalized inputs, with no changes to forward tensors."""

    def __init__(self, target, maximum_bytes=CAPTURE_LIMIT):
        super().__init__((target,), maximum_bytes=maximum_bytes)
        self.complete_inputs = {}
        self.complete_input_layouts = {}
        self.complete_input_bytes = 0

    def _record(self, label, tensor):
        if self.active is not None and label.endswith(":input"):
            if self.active in self.complete_inputs:
                raise ValueError("query projection executed twice in one captured model path")
            size = tensor.numel() * tensor.element_size()
            if size + self.complete_input_bytes > self.maximum_bytes:
                raise ValueError("complete normalized inputs exceeded registered capture limit")
            self.complete_inputs[self.active] = tensor.detach().to("cpu", copy=True).contiguous()
            self.complete_input_layouts[self.active] = {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "contiguous": tensor.is_contiguous(),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
            self.complete_input_bytes += size
        super()._record(label, tensor)


def _prior_bindings(prior, inputs):
    protocol = boundary._load_protocol(prior)
    gate_path = prior / "GATE_boundary_operators.json"
    gate = json.loads(gate_path.read_text())
    if (
        gate.get("status") != "passed_diagnostic_capture"
        or gate.get("attempted") != 2
        or gate.get("completed") != 2
        or gate.get("original_absolute_logit_tolerance") != boundary.LOGIT_ATOL
        or gate.get("thresholds_changed") is not False
        or gate.get("fullkv_chunked_logit_control_all_passed") is not False
    ):
        raise ValueError("two complete original negative numerical diagnoses are required")
    if json.loads((prior / "inputs.json").read_text()) != inputs:
        raise ValueError("diagnosis must reuse the same two consumed inputs")
    reports_path = prior / "operator_comparisons.json"
    if sha(reports_path) != gate["operator_comparisons_sha256"]:
        raise ValueError("original operator diagnosis hash changed")
    reports = json.loads(reports_path.read_text())
    if len(reports) != 2 or any(
        row.get("first_nonbitwise_operator") != "layer.0.attention.q_proj:output"
        or row.get("fullkv_chunked_logit_control_passed") is not False
        for row in reports
    ):
        raise ValueError("original earliest divergence is not the registered query projection")
    for row in reports:
        query_input = next(
            op for op in row["operators"] if op["operator"] == "layer.0.attention.q_proj:input"
        )
        if query_input["bitwise_equal"] is not True:
            raise ValueError("original query projection input already diverged")
    return {
        str(path): sha(path)
        for path in (prior / "protocol.json", prior / "inputs.json", gate_path, reports_path)
    }, protocol


def prepare(parent, root, *, prior_boundary):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("linear diagnosis requires a new immutable output root")
    root.resolve().relative_to(Path("/scratch/djy8hg").resolve())
    lock, inputs, bindings = boundary.registered_plan(parent)
    previous, _ = _prior_bindings(prior_boundary, inputs)
    bindings.update(previous)
    cfg = resolved_config(lock, "coco", "full_kv", None)
    from mosaickv.config import canonical_config, config_sha256

    if (
        cfg.model.precision.value != "fp16"
        or cfg.execution.attention_implementation != "eager"
        or not cfg.method.is_full_cache
    ):
        raise ValueError("only the unchanged native FP16 eager FullKV path is permitted")
    write_once(root / "resolved_config.json", canonical_config(cfg))
    write_once(root / "inputs.json", inputs)
    write_once(
        root / "protocol.json",
        {
            "stage": STAGE,
            "source_root": str(SOURCE),
            "source_sha": revision,
            "parent": str(parent),
            "prior_boundary": str(prior_boundary),
            "bindings": bindings,
            "inputs_sha256": sha(root / "inputs.json"),
            "resolved_config_sha256": sha(root / "resolved_config.json"),
            "config_sha256": config_sha256(cfg),
            "sample_count": 2,
            "seed": lock["seed"],
            "arithmetic": lock["arithmetic"],
            "original_model_arithmetic": ORIGINAL_FLAGS,
            "operator_only_variants": [{"name": n, "flags": f} for n, f in VARIANTS],
            "operator_shapes": ["full_sequence", "last_row", "shape_matched_zero_prefix"],
            "references": ["operator_GPU_FP32_no_TF32", "operator_CPU_FP64_last_row"],
            "maximum_capture_bytes_per_input": CAPTURE_LIMIT,
            "maximum_weight_capture_bytes": CAPTURE_LIMIT,
            "fullkv_chunked_first_logit_absolute_tolerance": boundary.LOGIT_ATOL,
            "model_precision_or_arithmetic_changed": False,
            "original_numerical_control_unchanged": True,
            "benchmark_labels_accessed": False,
            "consumed_development_only": True,
            "scoring": False,
            "headline_timing": False,
            "fresh_confirmation_authorized": False,
            "superiority_claim_allowed": False,
            "maximum_aggregate_A100s": 6,
            "worker_gpus": 1,
            "worker_cpus": 12,
            "worker_wall_minutes": 20,
        },
    )


def _last(output):
    return output[:, -1:, :].detach().to("cpu", copy=True).contiguous()


def cpu_fp64_reference(inputs, weight, bias, *, torch):
    """Fix CPU reducer threads for exact write-once worker/reanalysis agreement."""
    from torch.nn.functional import linear

    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return linear(
            inputs.detach().cpu().double(),
            weight.detach().cpu().double(),
            None if bias is None else bias.detach().cpu().double(),
        )
    finally:
        torch.set_num_threads(original_threads)


def operator_replays(inputs, weight, bias, *, torch):
    """Evaluate identical last-row operands at three shapes; save only final rows."""
    from torch.nn.functional import linear

    if inputs.ndim != 3 or inputs.shape[0] != 1 or inputs.shape[1] < 2:
        raise ValueError("a real batch-one prompt with at least two positions is required")
    if inputs.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("captured native operands must remain FP16")
    if inputs.shape[-1] != weight.shape[-1] or weight.ndim != 2:
        raise ValueError("query projection operand dimensions do not match")
    last = inputs[:, -1:, :].contiguous()
    padded = torch.zeros_like(inputs)
    padded[:, -1:, :].copy_(last)
    operands = {"full_sequence": inputs, "last_row": last, "shape_matched_zero_prefix": padded}
    variants = []
    for name, flags in VARIANTS:
        outputs = {}
        try:
            with isolated_flags(torch, flags) as active:
                for shape, operand in operands.items():
                    outputs[shape] = _last(linear(operand, weight, bias))
            variants.append(
                {"name": name, "status": "completed", "flags": active, "outputs": outputs}
            )
        except RuntimeError as error:
            if name != "fp16_accumulation_optional" or not any(
                reason in str(error).lower()
                for reason in ("unsupported", "not supported", "cublas_status_not_supported")
            ):
                raise
            variants.append({"name": name, "status": "unsupported", "error": str(error)})
        if arithmetic_flags(torch) != ORIGINAL_FLAGS:
            raise AssertionError("operator replay leaked arithmetic flags")
    # CPU FP64 is a descriptive high-precision last-row reference, not a new
    # model arithmetic policy or replacement for the registered model-logit test.
    reference64 = cpu_fp64_reference(last, weight, bias, torch=torch)
    with isolated_flags(torch, ORIGINAL_FLAGS):
        reference32 = {
            shape: _last(
                linear(operand.float(), weight.float(), None if bias is None else bias.float())
            )
            for shape, operand in operands.items()
        }
    return {
        "variants": variants,
        "GPU_FP32_no_TF32": reference32,
        "CPU_FP64_last_row": reference64,
        "actual_input_shape": list(inputs.shape),
        "actual_weight_shape": list(weight.shape),
        "arithmetic_flags_restored": arithmetic_flags(torch) == ORIGINAL_FLAGS,
    }


def compare_packet(packet, weight_packet):
    """CPU-only reanalysis; a passing diagnostic is not a passing Boundary-B test."""
    import torch

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _compare_packet(packet, weight_packet)
    finally:
        torch.set_num_threads(threads)


def _compare_packet(packet, weight_packet):
    import torch

    paths, replay = packet["model_paths"], packet["operator_replays"]
    full_input, held_input = (packet["complete_inputs"][p] for p in ("A", "B"))
    if full_input.device.type != "cpu" or held_input.device.type != "cpu":
        raise ValueError("captured inputs must be CPU tensors")
    input_comparison = boundary.compare_vectors(full_input[:, -1:, :], held_input)
    layouts = packet["complete_input_layouts"]
    if any(
        layouts[path]["shape"] != list(tensor.shape)
        or layouts[path]["stride"] != list(tensor.stride())
        or layouts[path]["contiguous"] is not True
        for path, tensor in (("A", full_input), ("B", held_input))
    ):
        raise ValueError("native contiguous operand layout was not preserved by capture")
    expected = cpu_fp64_reference(
        full_input[:, -1:, :], weight_packet["weight"], weight_packet["bias"], torch=torch
    )
    if not torch.equal(expected, replay["CPU_FP64_last_row"]):
        raise ValueError("FP64 operator reference does not reproduce from saved actual operands")
    model_report = boundary.compare_capture(paths)
    native = next(v for v in replay["variants"] if v["name"] == "native_fp16")
    if native["status"] != "completed" or not replay["arithmetic_flags_restored"]:
        raise ValueError("native isolated operator replay is incomplete or leaks settings")
    required_shapes = {"full_sequence", "last_row", "shape_matched_zero_prefix"}
    if [v["name"] for v in replay["variants"]] != [name for name, _ in VARIANTS]:
        raise ValueError("operator variant set/order differs from preregistration")
    if set(replay["GPU_FP32_no_TF32"]) != required_shapes:
        raise ValueError("FP32 reference shape set differs")
    report = []
    for variant, (_name, flags) in zip(replay["variants"], VARIANTS, strict=True):
        if variant["status"] == "unsupported":
            if variant["name"] != "fp16_accumulation_optional":
                raise ValueError(
                    "only the explicitly optional accumulation setting may be unsupported"
                )
            report.append({key: value for key, value in variant.items() if key != "outputs"})
            continue
        if set(variant["outputs"]) != required_shapes:
            raise ValueError("missing registered operator shape")
        if variant["flags"] != flags:
            raise ValueError("registered isolated operator settings differ")
        outputs = variant["outputs"]
        report.append(
            {
                "name": variant["name"],
                "status": variant["status"],
                "flags": variant["flags"],
                "full_vs_last_row": boundary.compare_vectors(
                    outputs["full_sequence"], outputs["last_row"]
                ),
                "full_vs_same_shape_zero_prefix": boundary.compare_vectors(
                    outputs["full_sequence"], outputs["shape_matched_zero_prefix"]
                ),
                "errors_vs_CPU_FP64": {
                    shape: boundary.compare_vectors(expected, output.double())
                    for shape, output in outputs.items()
                },
            }
        )
    output_label = "layer.0.attention.q_proj:output"
    replay_checks = {
        "native_full_matches_model_A": boundary.compare_vectors(
            paths["captures"]["A"][output_label]["vector"], native["outputs"]["full_sequence"]
        )["bitwise_equal"],
        "native_last_matches_model_B": boundary.compare_vectors(
            paths["captures"]["B"][output_label]["vector"], native["outputs"]["last_row"]
        )["bitwise_equal"],
    }
    return {
        "normalized_last_inputs": input_comparison,
        "native_input_layouts": layouts,
        "native_operator_replay_checks": replay_checks,
        "operator_variants": report,
        "FP32_errors_vs_CPU_FP64": {
            shape: boundary.compare_vectors(expected, output.double())
            for shape, output in replay["GPU_FP32_no_TF32"].items()
        },
        "original_model_logit_comparison": model_report["first_logit_comparison"],
        "original_fullkv_chunked_logit_control_passed": model_report[
            "fullkv_chunked_logit_control_passed"
        ],
        "original_absolute_logit_tolerance": boundary.LOGIT_ATOL,
        "original_control_unchanged": True,
        "operator_checks_do_not_pass_model_numerical_control": True,
        "interpretation": (
            "Shape/operand-controlled projection replay localizes numerical behavior; "
            "no causal attribution beyond this isolated operator, production precision "
            "fix, or quality claim follows."
        ),
    }


def worker(root):
    import torch
    from PIL import Image

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import config_sha256, load_config
    from mosaickv.evaluation.coco_shared_bytes import processor_signature
    from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    protocol = boundary._load_protocol(root)
    if protocol["stage"] != STAGE or (root / "result.json").exists():
        raise ValueError("wrong protocol or immutable worker result already exists")
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one allocated Slurm GPU is required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic cuBLAS setting must precede Python startup")
    if arithmetic_flags(torch) != ORIGINAL_FLAGS:
        raise RuntimeError("original native FP16 model arithmetic flags differ")
    torch.set_num_threads(12)
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    cfg = load_config(root / "resolved_config.json")
    if config_sha256(cfg) != protocol["config_sha256"]:
        raise ValueError("resolved configuration changed")
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    if any(p.device.type != "cuda" or p.dtype != torch.float16 for p in adapter.model.parameters()):
        raise RuntimeError("model must remain single-GPU native FP16, without offload")
    gpu = asdict(capture_gpu_environment(torch))
    devices = [d for d in gpu["devices"] if d["visible_to_process"]]
    if (
        len(devices) != 1
        or "A100" not in devices[0]["name"]
        or gpu["background_processes"]
        or gpu["query_error"]
    ):
        raise RuntimeError("single-A100 isolation could not be established")
    target = next(
        t for t in boundary.operator_targets(adapter) if t[0] == "layer.0.attention.q_proj"
    )
    module = target[2]
    if type(module) is not torch.nn.Linear:
        raise NotImplementedError("actual query projection is not an unmodified torch.nn.Linear")
    weight_packet = {
        "weight": module.weight.detach().cpu().clone(),
        "bias": None if module.bias is None else module.bias.detach().cpu().clone(),
    }
    weight_bytes = sum(
        t.numel() * t.element_size() for t in weight_packet.values() if t is not None
    )
    if weight_bytes > protocol["maximum_weight_capture_bytes"]:
        raise ValueError("query projection exceeds registered weight capture limit")
    boundary._save_tensor_once(root / "query_weight.pt", weight_packet)
    write_once(
        root / "environment.json",
        {
            "environment": environment(),
            "gpu": gpu,
            **scheduler_identity({}),
            "source_sha": protocol["source_sha"],
            "model": asdict(cfg.model),
            "dataset": asdict(cfg.dataset),
            "config_sha256": protocol["config_sha256"],
            "processor_sha256": digest(processor_signature(adapter.processor)),
            "actual_model_arithmetic_flags": arithmetic_flags(torch),
            "seed": protocol["seed"],
            "measurement_type": "isolated_linear_operator_diagnostic_not_timing_or_scoring",
            "backend": "huggingface",
            "attention": "eager",
            "vllm": "not_used",
            "sglang": "not_used",
            "installed_query_module_path": target[1],
            "installed_query_module_class": (
                f"{type(module).__module__}.{type(module).__qualname__}"
            ),
            "query_weight_sha256": sha(root / "query_weight.pt"),
        },
    )
    inputs = json.loads((root / "inputs.json").read_text())["samples"]
    if len(inputs) != 2:
        raise ValueError("exactly two consumed inputs required")
    records = []
    for index, sample in enumerate(inputs):
        start = time.perf_counter()
        try:
            if arithmetic_flags(torch) != ORIGINAL_FLAGS:
                raise AssertionError("operator setting leaked into next model forward")
            media = []
            for item in sample["media"]:
                path = Path(item["path"]).resolve(strict=True)
                path.relative_to(Path("/scratch/djy8hg").resolve())
                if item["kind"] != "image" or sha(path) != item["sha256"]:
                    raise ValueError("input media changed or is unsupported")
                with Image.open(path) as image:
                    media.append(MediaItem(MediaKind.IMAGE, image.convert("RGB").copy()))
            prepared = adapter.prepare_inputs(
                build_multimodal_messages(sample["prompt"], tuple(media))
            )
            capture = LinearInputCapture(target, protocol["maximum_capture_bytes_per_input"])
            paths = boundary.run_paths(adapter, prepared, capture)
            if not all(row["contiguous"] for row in capture.complete_input_layouts.values()):
                raise NotImplementedError("isolated replay does not support strided native inputs")
            normalized = capture.complete_inputs["A"].to(adapter.device)
            with torch.inference_mode():
                replay = operator_replays(normalized, module.weight, module.bias, torch=torch)
            torch.cuda.synchronize(adapter.device)
            packet = {
                "model_paths": paths,
                "complete_inputs": capture.complete_inputs,
                "complete_input_layouts": capture.complete_input_layouts,
                "operator_replays": replay,
            }
            report = compare_packet(packet, weight_packet)
            path = root / "captures" / f"{index:02d}.pt"
            boundary._save_tensor_once(path, packet)
            records.append(
                {
                    "index": index,
                    "sample_id": sample["sample_id"],
                    "status": "completed",
                    "capture_path": str(path),
                    "capture_sha256": sha(path),
                    "input_sha256": digest(sample),
                    "instrumented_wall_seconds": time.perf_counter() - start,
                    "complete_normalized_input_bytes": capture.complete_input_bytes,
                    "report": report,
                }
            )
            del normalized, packet, replay, paths, capture, prepared
        except Exception as error:
            records.append(
                {
                    "index": index,
                    "sample_id": sample["sample_id"],
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        gc.collect()
        torch.cuda.empty_cache()
    write_once(
        root / "result.json",
        {
            "source_sha": protocol["source_sha"],
            "protocol_sha256": sha(root / "protocol.json"),
            "environment_sha256": sha(root / "environment.json"),
            "query_weight_sha256": sha(root / "query_weight.pt"),
            "actual_final_arithmetic_flags": arithmetic_flags(torch),
            "records": records,
        },
    )


def analyze(root):
    import torch

    protocol = boundary._load_protocol(root)
    result_path = root / "result.json"
    if not result_path.exists():
        write_once(
            root / "GATE_boundary_linear.json",
            {"status": "blocked_missing_worker_result", "B_quality_claim_allowed": False},
        )
        return
    result = json.loads(result_path.read_text())
    if (
        result["protocol_sha256"] != sha(root / "protocol.json")
        or result["source_sha"] != protocol["source_sha"]
        or result["environment_sha256"] != sha(root / "environment.json")
        or result["query_weight_sha256"] != sha(root / "query_weight.pt")
        or result["actual_final_arithmetic_flags"] != ORIGINAL_FLAGS
    ):
        raise ValueError("worker provenance or restored arithmetic differs")
    expected = json.loads((root / "inputs.json").read_text())["samples"]
    records = result["records"]
    if (
        len(records) != 2
        or [r["index"] for r in records] != [0, 1]
        or [r["sample_id"] for r in records] != [s["sample_id"] for s in expected]
    ):
        raise ValueError("ordered complete two-input denominator required")
    weight = torch.load(root / "query_weight.pt", map_location="cpu", weights_only=True)
    reports = []
    for record in records:
        if record["status"] != "completed":
            continue
        path = Path(record["capture_path"])
        path.resolve(strict=True).relative_to(root.resolve())
        if sha(path) != record["capture_sha256"]:
            raise ValueError("captured operator operands changed")
        packet = torch.load(path, map_location="cpu", weights_only=True)
        report = compare_packet(packet, weight)
        if report != record["report"] or record["input_sha256"] != digest(
            expected[record["index"]]
        ):
            raise ValueError("reanalysis or input binding differs")
        reports.append({"index": record["index"], **report})
    complete = len(reports) == 2
    write_once(root / "linear_comparisons.json", reports)
    write_once(
        root / "GATE_boundary_linear.json",
        {
            "status": "passed_operator_diagnostic" if complete else "blocked_incomplete_diagnostic",
            "attempted": 2,
            "completed": len(reports),
            "failed": 2 - len(reports),
            "source_sha": protocol["source_sha"],
            "protocol_sha256": sha(root / "protocol.json"),
            "result_sha256": sha(result_path),
            "linear_comparisons_sha256": sha(root / "linear_comparisons.json"),
            "native_operator_replay_all_matched": complete
            and all(all(r["native_operator_replay_checks"].values()) for r in reports),
            "normalized_last_inputs_all_bitwise_equal": complete
            and all(r["normalized_last_inputs"]["bitwise_equal"] for r in reports),
            "original_FullKV_logit_control_all_passed": complete
            and all(r["original_fullkv_chunked_logit_control_passed"] for r in reports),
            "original_absolute_logit_tolerance": boundary.LOGIT_ATOL,
            "thresholds_changed": False,
            "production_model_arithmetic_changed": False,
            "operator_replay_not_a_model_fix": True,
            "B_quality_claim_allowed": False,
            "fresh_confirmation_authorized": False,
            "superiority_established": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--prior-boundary", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.parent is None or args.prior_boundary is None:
            parser.error("prepare requires --parent and --prior-boundary")
        prepare(args.parent, args.root, prior_boundary=args.prior_boundary)
    elif args.mode == "worker":
        worker(args.root)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
