#!/usr/bin/env python3
"""Bounded FullKV A/B final-position operator diagnosis on consumed inputs.

No candidate, scoring, precision change or numerical-tolerance repair is made.
Hooks copy only the final token's vectors to CPU, never full prompt activations.
This entry point submits no jobs. Its capture-completeness gate is explicitly
separate from the unchanged 1e-4 FullKV chunked-prefill logit control.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import re
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from evicastkv_campaign import SOURCE, environment, resolved_config
from evicastkv_development import runtime_files

from mosaickv.evaluation.fox_boundary import hold_back_last_text_token
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

STAGE = "boundary_operators"
LOGIT_ATOL = 1e-4
MAX_CAPTURE_BYTES = 64 * 2**20
FORBIDDEN = frozenset(
    {
        "answer",
        "answers",
        "reference",
        "references",
        "score",
        "scores",
        "predictions",
        "response",
        "teacher_forcing_target",
    }
)
_STRING = re.compile(r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"')


def _skip_json_value(text, start):
    """Skip a field lexically: forbidden string values are never decoded."""
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        raise ValueError("truncated ledger JSON")
    if text[i] == '"':
        match = _STRING.match(text, i)
        if match is None:
            raise ValueError("invalid string in ledger JSON")
        return match.end()
    if text[i] in "[{":
        close = "]" if text[i] == "[" else "}"
        i += 1
        while i < len(text):
            if text[i].isspace() or text[i] in ",:":
                i += 1
            elif text[i] == close:
                return i + 1
            elif text[i] in "]}":
                raise ValueError("mismatched JSON delimiter")
            else:
                i = _skip_json_value(text, i)
        raise ValueError("unterminated ledger JSON container")
    match = re.match(r"(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)", text[i:])
    if match is None:
        raise ValueError("invalid ledger JSON scalar")
    return i + match.end()


def input_only_projection(raw):
    """Redact benchmark-value fields before JSON object construction.

    The parent ledger hash/validator already fixes the original complete JSON.
    This projection never decodes forbidden field values or exposes them to
    hooks, the model, comparisons or diagnostic exports.
    """
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    pieces, last, i = [], 0, 0
    while i < len(text):
        if text[i] != '"':
            i += 1
            continue
        match = _STRING.match(text, i)
        if match is None:
            raise ValueError("invalid ledger JSON string")
        end = match.end()
        colon = end
        while colon < len(text) and text[colon].isspace():
            colon += 1
        if colon < len(text) and text[colon] == ":" and json.loads(match.group()) in FORBIDDEN:
            value_end = _skip_json_value(text, colon + 1)
            pieces.extend((text[last : colon + 1], "null"))
            last = value_end
            i = value_end
        else:
            i = end
    pieces.append(text[last:])
    value = json.loads("".join(pieces))
    samples = []
    for item in value["samples"]:
        samples.append({key: item[key] for key in ("sample_id", "prompt", "media")})
    return {key: value[key] for key in ("dataset_id", "dataset_revision", "split", "task")} | {
        "samples": samples
    }


def _function_hash(path, name):
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def registered_plan(parent, *, source=SOURCE):
    lock_path = parent / "protocol_lock.json"
    lock = json.loads(lock_path.read_text())
    old = Path(lock["source_root"])
    clean_source(old, lock["source_sha"])
    if runtime_files(source) != runtime_files(old):
        raise ValueError("production runtime changed; cannot inherit native parent gates")
    controls = (
        "mosaickv/scripts/evicastkv_campaign.py",
        "mosaickv/src/mosaickv/adapters/huggingface/base.py",
        "mosaickv/src/mosaickv/adapters/huggingface/llava_next.py",
        "mosaickv/configs/experiments/evicastkv_fox/coco/full_kv.json",
    )
    if any(sha(source / p) != sha(old / p) for p in controls):
        raise ValueError("parent input/model/adapter controls changed")
    boundary = Path("mosaickv/src/mosaickv/evaluation/fox_boundary.py")
    if _function_hash(source / boundary, "hold_back_last_text_token") != _function_hash(
        old / boundary, "hold_back_last_text_token"
    ):
        raise ValueError("held-back-token input construction changed")
    arithmetic = lock.get("arithmetic", {})
    if (
        arithmetic.get("allow_fp16_reduced_precision_reduction") is not True
        or arithmetic.get("attention_layout_policy", "native") != "native"
    ):
        raise ValueError("diagnostic must retain native parent FP16 arithmetic")
    bindings = {str(lock_path): sha(lock_path)}
    bindings.update({str(source / p): sha(source / p) for p in controls})
    for stage in ("correctness", "scoring", "profiling", "measurement"):
        path = parent / "analysis" / stage / f"GATE_{stage}.json"
        gate = json.loads(path.read_text())
        if (
            gate.get("status") != "passed"
            or gate.get("protocol_sha256") != sha(lock_path)
            or gate.get("source_sha") != lock["source_sha"]
            or gate.get("blockers")
        ):
            raise ValueError(f"parent gate not passed: {stage}")
        bindings[str(path)] = sha(path)
    info = lock["cohorts"]["coco"]["profiler"]
    path = Path(info["path"])
    if sha(path) != info["sha256"]:
        raise ValueError("consumed profiler ledger changed")
    inputs = input_only_projection(path.read_bytes())
    ids = [s["sample_id"] for s in inputs["samples"]]
    if ids != info["sample_ids"] or len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("complete unique consumed profiler identities required")
    bindings[str(path)] = sha(path)
    inputs["samples"] = inputs["samples"][:2]
    return lock, inputs, bindings


def prepare(parent, root):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("boundary operator diagnostic must use a new root")
    root.resolve().relative_to(Path("/scratch/djy8hg").resolve())
    lock, inputs, bindings = registered_plan(parent)
    cfg = resolved_config(lock, "coco", "full_kv", None)
    from mosaickv.config import canonical_config, config_sha256

    if (
        cfg.model.precision.value != "fp16"
        or cfg.execution.attention_implementation != "eager"
        or not cfg.method.is_full_cache
    ):
        raise ValueError("only native FP16 eager FullKV is registered")
    write_once(root / "resolved_config.json", canonical_config(cfg))
    write_once(root / "inputs.json", inputs)
    write_once(
        root / "protocol.json",
        dict(
            schema_version=1,
            stage=STAGE,
            status="READY_NOT_RUN",
            source_root=str(SOURCE),
            source_sha=revision,
            parent=str(parent),
            bindings=bindings,
            inputs_sha256=sha(root / "inputs.json"),
            resolved_config_sha256=sha(root / "resolved_config.json"),
            config_sha256=config_sha256(cfg),
            sample_count=2,
            seed=lock["seed"],
            arithmetic=lock["arithmetic"],
            maximum_capture_bytes_per_input=MAX_CAPTURE_BYTES,
            fullkv_chunked_first_logit_absolute_tolerance=LOGIT_ATOL,
            capture_boundary="final original prompt position only; A monolithic, B held token once",
            benchmark_fields_lexically_redacted_before_JSON_decode=True,
            benchmark_labels_accessed=False,
            scoring=False,
            headline_timing=False,
            candidate_or_arithmetic_change=False,
            fresh_confirmation_authorized=False,
            maximum_aggregate_A100s=6,
            worker_gpus=1,
            worker_cpus=12,
            worker_wall_minutes=120,
        ),
    )


def operator_targets(adapter):
    """Resolve installed module objects rather than guessing one wrapper path."""
    layers = tuple(adapter._language_layers())
    names = {id(module): name for name, module in adapter.model.named_modules()}
    owners = [
        module
        for module in adapter.model.modules()
        if getattr(module, "layers", None) is not None and tuple(module.layers) == layers
    ]
    if len(owners) != 1:
        raise NotImplementedError("language decoder parent is not uniquely exposed")
    targets = []
    for index, layer in enumerate(layers):
        members = [
            ("layer", layer),
            ("input_layernorm", layer.input_layernorm),
            ("attention", layer.self_attn),
        ]
        members.extend(
            (f"attention.{name}", getattr(layer.self_attn, name))
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        members.extend(
            (("post_attention_layernorm", layer.post_attention_layernorm), ("mlp", layer.mlp))
        )
        members.extend(
            (f"mlp.{name}", getattr(layer.mlp, name))
            for name in ("gate_proj", "up_proj", "down_proj")
        )
        targets.extend(
            (f"layer.{index}.{label}", names[id(module)], module) for label, module in members
        )
    for label, module in (
        ("final_norm", owners[0].norm),
        ("readout", adapter.model.get_output_embeddings()),
    ):
        if module is None or id(module) not in names:
            raise NotImplementedError(f"missing installed {label} module")
        targets.append((label, names[id(module)], module))
    return targets


class FinalPositionCapture:
    """Read-only module hooks with an explicit CPU-only capture byte ceiling."""

    def __init__(self, targets, maximum_bytes=MAX_CAPTURE_BYTES):
        self.targets = tuple(targets)
        self.maximum_bytes = maximum_bytes
        self.total_bytes = 0
        self.records = {}
        self.orders = {}
        self.active = None
        self.handles = []

    def _record(self, label, tensor):
        if self.active is None:
            return
        import torch

        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim not in (2, 3)
            or tensor.shape[0] != 1
        ):
            raise ValueError("operator capture requires batch-one [B,N,D] or [B,D] tensor")
        if tensor.ndim == 3 and tensor.shape[1] < 1:
            raise ValueError("empty operator sequence")
        vector = tensor[:, -1:, :] if tensor.ndim == 3 else tensor
        size = vector.numel() * vector.element_size()
        if self.total_bytes + size > self.maximum_bytes:
            raise ValueError("final-position capture exceeded registered byte ceiling")
        if label in self.records[self.active]:
            raise ValueError("operator executed more than once in a captured path")
        cpu = vector.detach().to(device="cpu", copy=True).contiguous()
        self.records[self.active][label] = dict(
            vector=cpu, source_shape=list(tensor.shape), dtype=str(tensor.dtype)
        )
        self.orders[self.active].append(label)
        self.total_bytes += size

    def __enter__(self):
        if self.handles:
            raise RuntimeError("capture cannot be entered twice")
        for label, _path, module in self.targets:

            def before(_module, args, kwargs, *, name=label):
                tensor = args[0] if args else kwargs.get("hidden_states")
                self._record(name + ":input", tensor)

            def after(_module, _args, output, *, name=label):
                self._record(name + ":output", output[0] if isinstance(output, tuple) else output)

            self.handles.extend(
                (
                    module.register_forward_pre_hook(before, with_kwargs=True),
                    module.register_forward_hook(after),
                )
            )
        return self

    @contextmanager
    def phase(self, name):
        if self.active is not None or name in self.records:
            raise ValueError("capture phases must be unique and nonnested")
        self.records[name], self.orders[name], self.active = {}, [], name
        try:
            yield
        finally:
            self.active = None

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = None


def run_paths(adapter, prepared, capture):
    """One independent monolithic prefill; one prefix prefill plus held token."""
    prefix, held = hold_back_last_text_token(prepared)
    original = prepared.logical_sequence_length
    if prefix.logical_sequence_length != original - 1:
        raise AssertionError("prefix construction did not remove exactly one original token")
    with capture:
        with capture.phase("A"):
            full = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
        if (
            full.state.logical_sequence_length != original
            or full.state.next_decode_position != original
            or full.state.active_cache_length != original
        ):
            raise AssertionError("monolithic FullKV state changed original positions/capacity")
        logits_a = full.logits.detach().float().cpu().clone()
        del full
        gc.collect()
        prefix_output = adapter.prefill(prefix, capture_queries=False, capture_attentions=False)
        state = prefix_output.state
        if (
            state.logical_sequence_length != original - 1
            or state.next_decode_position != original - 1
            or state.active_cache_length != original - 1
        ):
            raise AssertionError("held token would be processed at the wrong logical position")
        with capture.phase("B"):
            split = adapter.decode_one_token(
                held, state, capture_queries=False, capture_attentions=False
            )
        state = split.state
        if (
            state.logical_sequence_length != original
            or state.next_decode_position != original
            or state.active_cache_length != original
        ):
            raise AssertionError("held token was double-fed or FullKV state capacity changed")
        logits_b = split.logits.detach().float().cpu().clone()
    expected = {
        label + ":" + side
        for label, _path, _module in capture.targets
        for side in ("input", "output")
    }
    if any(set(capture.records[path]) != expected for path in ("A", "B")):
        raise ValueError("a registered operator hook did not execute on both paths")
    return dict(
        captures=capture.records,
        orders=capture.orders,
        logits_A=logits_a,
        logits_B=logits_b,
        held_token_forward_count=1,
        original_logical_length=original,
        prefix_logical_length=original - 1,
        captured_bytes=capture.total_bytes,
        source_KV_modified=False,
        compressed_cache_used=False,
    )


def compare_vectors(reference, candidate):
    import torch

    if reference.device.type != "cpu" or candidate.device.type != "cpu":
        raise ValueError("analysis requires CPU captures only")
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("operator comparison shape/dtype mismatch")
    left, right = reference.double(), candidate.double()
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("nonfinite operator capture")
    unequal = reference.view(torch.uint8).reshape(-1) != candidate.view(torch.uint8).reshape(-1)
    delta = right - left
    first = torch.nonzero(unequal, as_tuple=False)
    return dict(
        shape=list(reference.shape),
        dtype=str(reference.dtype),
        bitwise_equal=not bool(unequal.any()),
        first_different_storage_byte=int(first[0].item()) if first.numel() else None,
        maximum_absolute_difference=float(delta.abs().max()),
        normalized_L2=float(
            torch.linalg.vector_norm(delta) / max(1.0, float(torch.linalg.vector_norm(left)))
        ),
        normalized_L2_definition="FP64 ||candidate-reference||2 / max(1, ||reference||2)",
    )


def compare_capture(record):
    if record["orders"]["A"] != record["orders"]["B"] or not record["orders"]["A"]:
        raise ValueError("A/B operator execution order or coverage differs")
    comparisons = []
    for name in record["orders"]["A"]:
        a, b = (record["captures"][path][name] for path in ("A", "B"))
        result = compare_vectors(a["vector"], b["vector"])
        comparisons.append(
            dict(
                operator=name,
                source_shape_A=a["source_shape"],
                source_shape_B=b["source_shape"],
                **result,
            )
        )
    logits = compare_vectors(record["logits_A"], record["logits_B"])
    return dict(
        operators=comparisons,
        first_nonbitwise_operator=next(
            (c["operator"] for c in comparisons if not c["bitwise_equal"]), None
        ),
        first_operator_exceeding_absolute_1e_minus4=next(
            (c["operator"] for c in comparisons if c["maximum_absolute_difference"] > LOGIT_ATOL),
            None,
        ),
        first_logit_comparison=logits,
        fullkv_chunked_logit_control_passed=logits["maximum_absolute_difference"] <= LOGIT_ATOL,
        fullkv_chunked_logit_absolute_tolerance=LOGIT_ATOL,
        original_control_unchanged=True,
        caveat=(
            "Earliest observed final-position divergence is localization, not proof of "
            "the causal source; attention internals and previous-position vectors "
            "are not captured. Intermediate 1e-4 crossings are descriptive only."
        ),
    )


def _save_tensor_once(path, value):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".capture-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.chmod(0o444)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _load_protocol(root):
    protocol = json.loads((root / "protocol.json").read_text())
    clean_source(Path(protocol["source_root"]), protocol["source_sha"])
    for filename, field in (
        ("inputs.json", "inputs_sha256"),
        ("resolved_config.json", "resolved_config_sha256"),
    ):
        if sha(root / filename) != protocol[field]:
            raise ValueError(f"bound diagnostic artifact changed: {filename}")
    for path, expected in protocol["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("parent/source binding changed")
    return protocol


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

    protocol = _load_protocol(root)
    if (root / "result.json").exists():
        raise FileExistsError("diagnostic worker result is immutable")
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("one allocated Slurm GPU required; no login-node execution")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("cuBLAS deterministic configuration must be set before Python starts")
    flags = dict(
        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        allow_fp16_accumulation=torch.backends.cuda.matmul.allow_fp16_accumulation,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    if flags != dict(
        allow_fp16_reduced_precision_reduction=True, allow_fp16_accumulation=False, allow_tf32=False
    ):
        raise RuntimeError("parent native arithmetic flags differ; no precision fix authorized")
    torch.set_num_threads(12)
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    cfg = load_config(root / "resolved_config.json")
    if config_sha256(cfg) != protocol["config_sha256"]:
        raise ValueError("resolved FullKV configuration changed")
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    if any(p.device.type != "cuda" or p.dtype != torch.float16 for p in adapter.model.parameters()):
        raise RuntimeError("native single-GPU FP16 model cannot offload or mix dtypes")
    gpu = asdict(capture_gpu_environment(torch))
    visible = [d for d in gpu["devices"] if d["visible_to_process"]]
    if (
        len(visible) != 1
        or "A100" not in visible[0]["name"]
        or gpu["query_error"]
        or gpu["background_processes"]
    ):
        raise RuntimeError("single-A100 isolation cannot be established")
    targets = operator_targets(adapter)
    write_once(
        root / "environment.json",
        dict(
            environment=environment(),
            gpu=gpu,
            **scheduler_identity({}),
            source_sha=protocol["source_sha"],
            model=asdict(cfg.model),
            dataset=asdict(cfg.dataset),
            config_sha256=protocol["config_sha256"],
            processor_sha256=digest(processor_signature(adapter.processor)),
            actual_arithmetic_flags=flags,
            seed=protocol["seed"],
            measurement_type="operator_diagnostic_not_timing_or_scoring",
            backend="huggingface",
            attention="eager",
            vllm="not_used",
            sglang="not_used",
            operators=[
                dict(
                    label=label,
                    installed_module_path=path,
                    module_class=f"{type(module).__module__}.{type(module).__qualname__}",
                )
                for label, path, module in targets
            ],
        ),
    )
    inputs = json.loads((root / "inputs.json").read_text())["samples"]
    if len(inputs) != 2:
        raise ValueError("exactly two consumed inputs registered")
    records = []
    for index, sample in enumerate(inputs):
        start = time.perf_counter()
        try:
            media = []
            for item in sample["media"]:
                path = Path(item["path"]).resolve(strict=True)
                path.relative_to(Path("/scratch/djy8hg").resolve())
                if item["kind"] != "image" or sha(path) != item["sha256"]:
                    raise ValueError("input media identity changed or unsupported")
                with Image.open(path) as image:
                    media.append(MediaItem(MediaKind.IMAGE, image.convert("RGB").copy()))
            messages = build_multimodal_messages(sample["prompt"], tuple(media))
            prepared = adapter.prepare_inputs(messages)
            capture = FinalPositionCapture(targets, protocol["maximum_capture_bytes_per_input"])
            result = run_paths(adapter, prepared, capture)
            torch.cuda.synchronize(adapter.device)
            report = compare_capture(result)
            path = root / "captures" / f"{index:02d}.pt"
            _save_tensor_once(path, result)
            records.append(
                dict(
                    index=index,
                    sample_id=sample["sample_id"],
                    status="completed",
                    capture_path=str(path),
                    capture_sha256=sha(path),
                    input_sha256=digest(sample),
                    instrumented_wall_seconds=time.perf_counter() - start,
                    captured_bytes=result["captured_bytes"],
                    report=report,
                )
            )
            del prepared, result, capture
        except Exception as error:
            records.append(
                dict(
                    index=index,
                    sample_id=sample["sample_id"],
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            )
        gc.collect()
        torch.cuda.empty_cache()
    write_once(
        root / "result.json",
        dict(
            source_sha=protocol["source_sha"],
            protocol_sha256=sha(root / "protocol.json"),
            environment_sha256=sha(root / "environment.json"),
            measurement_type="operator_diagnostic_not_timing_or_scoring",
            records=records,
        ),
    )


def analyze(root):
    import torch

    protocol = _load_protocol(root)
    path = root / "result.json"
    if not path.exists():
        write_once(
            root / "GATE_boundary_operators.json",
            dict(
                status="blocked_diagnostic_execution",
                reason="worker result missing",
                B_quality_claim_allowed=False,
            ),
        )
        return
    result = json.loads(path.read_text())
    if (
        result["protocol_sha256"] != sha(root / "protocol.json")
        or result["source_sha"] != protocol["source_sha"]
        or result["environment_sha256"] != sha(root / "environment.json")
    ):
        raise ValueError("worker provenance changed")
    expected = json.loads((root / "inputs.json").read_text())["samples"]
    records = result["records"]
    if (
        len(records) != 2
        or [r["sample_id"] for r in records] != [s["sample_id"] for s in expected]
        or [r["index"] for r in records] != [0, 1]
    ):
        raise ValueError("complete ordered two-input cohort required; no intersection")
    reports = []
    for record in records:
        if record["status"] != "completed":
            continue
        capture_path = Path(record["capture_path"])
        capture_path.resolve(strict=True).relative_to(root.resolve())
        if sha(capture_path) != record["capture_sha256"]:
            raise ValueError("saved CPU capture changed")
        captures = torch.load(capture_path, map_location="cpu", weights_only=True)
        report = compare_capture(captures)
        if report != record["report"]:
            raise ValueError("CPU capture reanalysis differs from worker comparison")
        reports.append(dict(index=record["index"], **report))
    complete = len(reports) == 2
    write_once(root / "operator_comparisons.json", reports)
    write_once(
        root / "GATE_boundary_operators.json",
        dict(
            status="passed_diagnostic_capture" if complete else "blocked_diagnostic_execution",
            source_sha=protocol["source_sha"],
            protocol_sha256=sha(root / "protocol.json"),
            result_sha256=sha(path),
            operator_comparisons_sha256=sha(root / "operator_comparisons.json"),
            attempted=2,
            completed=len(reports),
            failed=2 - len(reports),
            fullkv_chunked_logit_control_all_passed=complete
            and all(r["fullkv_chunked_logit_control_passed"] for r in reports),
            original_absolute_logit_tolerance=LOGIT_ATOL,
            thresholds_changed=False,
            A_primary_gate_rewritten=False,
            B_quality_claim_allowed=False,
            fresh_confirmation_authorized=False,
            superiority_established=False,
            diagnostic_capture_completeness_is_not_numerical_control_pass=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        prepare(args.parent, args.root)
    elif args.mode == "worker":
        worker(args.root)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
