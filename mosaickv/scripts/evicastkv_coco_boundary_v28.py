#!/usr/bin/env python3
"""128-prediction corrected COCO boundary control, gated by whole-KV evidence.

Same two consumed inputs, original FP16 model/weights/epsilon/arithmetic and
1e-4 tolerance. Only the held-token boundary uses the passed V9 shape
transport; all 127 continuations use restored ordinary native methods.
This is not benchmark scoring, source/GQA support, or a performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

from mosaickv.evaluation.coco_numerical_status import (
    CONSUMED_IDS,
    MODEL,
    SHAPE_ROOT,
    audit,
)
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
BRIDGE_PRODUCER = "21fbfe2aaa98a1e7ab73f315c65616053bc4e599"
BRIDGE_ROOT = Path("/scratch/djy8hg/workdir/fox-review-v31/run-20261111-1")
BRIDGE_PLAN_SHA = "74f70c760530e5e514415b779fd8ccdaf7ccffaf4e8367db53ca369c091e5330"
BRIDGE_GATE_NAME = "GATE_bounded_reader_v28.json"
POLICY = dict(
    id="V28_consumed_COCO_corrected_boundary_128_native_continuation_v1",
    consumed_inputs_only=True,
    model=MODEL,
    cases=list(CONSUMED_IDS),
    predictions=128,
    held_token_predictions=1,
    ordinary_continuation_steps=127,
    tolerance=1e-4,
    seed=20271014,
    boundary_transport="passed_V9_full_row_native_shapes_and_boundary_only_RMSNorm",
    native_precision_weights_epsilon_preserved=True,
    historical_native_failure_preserved=True,
    parent_benchmark_maximum_new_tokens=64,
    diagnostic_maximum_predictions=128,
    reference_prefix_forced_on_both_paths=True,
    maximum_a100s=6,
    six_a100s_explicitly_authorized=True,
    workers=2,
    gpus_per_worker=1,
    cpus_per_worker=12,
    worker_minutes=15,
    fresh_confirmation_authorized=False,
    model_source_reconstruction_enabled=False,
    six_mechanisms_enabled=False,
    scoring=False,
    headline_timing=False,
    whole_request_peak_claim=False,
    paper_campaign_authorized=False,
    whole_KV_prerequisite_producers=["same_clean_execution", BRIDGE_PRODUCER],
    derived_prerequisite_origin="derived_validation_not_new_GPU_execution",
    controlled_execution_base_sha=BRIDGE_PRODUCER,
    base_source_change_policy="additions_only_all_original_files_byte_identical",
)


def read(path):
    return json.loads(Path(path).read_text())


def execution_base_proof(revision):
    """Verify every producer-tracked file, not merely selected module names."""
    output = subprocess.check_output(
        ["git", "diff", "--name-status", "-z", "--no-renames", BRIDGE_PRODUCER, revision],
        cwd=SOURCE,
        text=True,
    )
    fields = output.split("\0")
    if fields[-1] != "":
        raise ValueError("incomplete Git source-comparison record")
    fields.pop()
    if len(fields) % 2:
        raise ValueError("malformed Git source-comparison record")
    additions = []
    for status, name in zip(fields[::2], fields[1::2], strict=True):
        if status != "A" or not name:
            raise ValueError("COCO base controls changed: only additive CPU/tooling files allowed")
        additions.append(name)
    transport = SOURCE / "mosaickv/src/mosaickv/evaluation/fox_shape_transport.py"
    if sha(transport) != read(SHAPE_ROOT / "transport.json")["implementation_sha256"]:
        raise ValueError("original passed V9 transport hash changed")
    return dict(
        actual_GPU_execution_base_sha=BRIDGE_PRODUCER,
        new_control_source_sha=revision,
        all_original_producer_files_byte_identical=True,
        original_V9_transport_sha256=sha(transport),
        source_change_policy=POLICY["base_source_change_policy"],
        added_paths=additions,
    )


def require_whole_gate(gate, revision):
    if gate.get("failures") != []:
        raise ValueError("complete successful immutable whole-KV gate required")
    for name in ("complete", "source_fidelity_128_pass", "whole_request_KV_byte_match_pass"):
        if gate.get(name) is not True:
            raise ValueError(f"whole-KV prerequisite did not pass: {name}")
    derived_keys = (
        "actual_execution_source_sha",
        "validation_source_sha",
        "gate_origin",
        "independent_replay_from_actual_GPU_artifacts",
        "actual_execution_root",
        "actual_execution_root_sha256",
        "actual_plan_path",
        "actual_plan_sha256",
        "actual_gate_path",
        "actual_gate_sha256",
    )
    if gate.get("source_sha") == revision and all(name not in gate for name in derived_keys):
        return {}
    if (
        gate.get("source_sha") != BRIDGE_PRODUCER
        or gate.get("actual_execution_source_sha") != BRIDGE_PRODUCER
        or gate.get("validation_source_sha") != revision
        or gate.get("gate_origin") != POLICY["derived_prerequisite_origin"]
        or gate.get("independent_replay_from_actual_GPU_artifacts") is not True
    ):
        raise ValueError("immutable whole-KV execution/validation source provenance mismatch")
    path_fields = ("actual_execution_root", "actual_plan_path", "actual_gate_path")
    if any(not isinstance(gate.get(name), str) or not gate[name] for name in path_fields):
        raise ValueError("complete explicit actual-execution path bindings required")
    root = Path(gate["actual_execution_root"]).resolve(strict=True)
    if root != BRIDGE_ROOT.resolve(strict=True) or gate.get(
        "actual_execution_root_sha256"
    ) != digest(str(root)):
        raise ValueError("actual-execution root binding changed")
    bindings = {}
    for path_field, hash_field, relative in (
        ("actual_plan_path", "actual_plan_sha256", "plan.json"),
        ("actual_gate_path", "actual_gate_sha256", f"analysis/{BRIDGE_GATE_NAME}"),
    ):
        path = Path(gate[path_field]).resolve(strict=True)
        if path != (root / relative).resolve(strict=True) or sha(path) != gate.get(hash_field):
            raise ValueError("actual-execution plan/gate artifact binding changed")
        bindings[str(path)] = gate[hash_field]
    if gate["actual_plan_sha256"] != BRIDGE_PLAN_SHA:
        raise ValueError("actual-execution plan is not the pinned immutable producer plan")
    original_plan = read(root / "plan.json")
    original_gate = read(root / "analysis" / BRIDGE_GATE_NAME)
    if (
        original_plan.get("source_sha") != BRIDGE_PRODUCER
        or original_gate.get("source_sha") != BRIDGE_PRODUCER
        or original_gate.get("plan_sha256") != BRIDGE_PLAN_SHA
        or original_gate.get("failures") != []
        or any(
            original_gate.get(name) is not True
            for name in ("complete", "source_fidelity_128_pass", "whole_request_KV_byte_match_pass")
        )
    ):
        raise ValueError("original pinned GPU producer did not pass its registered gates")
    return bindings


def validate(root):
    root = Path(root)
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy_sha256"] != digest(POLICY) or digest(plan["policy"]) != digest(POLICY):
        raise ValueError("immutable V28 numerical policy changed")
    if digest(execution_base_proof(plan["source_sha"])) != digest(plan["execution_base_proof"]):
        raise ValueError("immutable base-control source proof changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"immutable V28 prerequisite changed: {name}")
    for name, field in (
        ("inputs.json", "inputs_sha256"),
        ("resolved_config.json", "config_sha256"),
        ("recovered_numerical_evidence.json", "recovery_sha256"),
    ):
        if sha(root / name) != plan[field]:
            raise ValueError(f"prepared V28 evidence changed: {name}")
    require_whole_gate(read(plan["whole_gate"]), plan["source_sha"])
    return plan


def prepare(root, whole_gate):
    revision = clean_source(SOURCE)
    base_proof = execution_base_proof(revision)
    whole_gate = Path(whole_gate).resolve(strict=True)
    whole_gate.relative_to(Path("/scratch/djy8hg").resolve())
    whole = read(whole_gate)
    prerequisite_bindings = require_whole_gate(whole, revision)
    recovery = audit()
    implementation = SOURCE / "mosaickv/src/mosaickv/evaluation/fox_shape_transport.py"
    transport = read(SHAPE_ROOT / "transport.json")
    if sha(implementation) != transport["implementation_sha256"]:
        raise ValueError("passed V9 numerical transport implementation changed")
    inputs, config = read(SHAPE_ROOT / "inputs.json"), read(SHAPE_ROOT / "resolved_config.json")
    if tuple(row["sample_id"] for row in inputs["samples"]) != CONSUMED_IDS:
        raise ValueError("only the two registered consumed identities are allowed")
    if config["model"] != MODEL or config["generation"]["max_new_tokens"] != 64:
        raise ValueError("pinned original model and disclosed parent generation policy required")
    config["generation"]["max_new_tokens"] = POLICY["diagnostic_maximum_predictions"]
    bindings = dict(recovery["bindings"], **prerequisite_bindings)
    if whole.get("gate_origin") == POLICY["derived_prerequisite_origin"]:
        provenance_path = whole_gate.parent / "provenance.json"
        provenance = read(provenance_path)
        if (
            provenance.get("execution_source_sha") != BRIDGE_PRODUCER
            or provenance.get("analyzer_source_sha") != revision
            or provenance.get("preserves_original_artifacts") is not True
            or provenance.get("GPU_inference_repeated") is not False
            or not isinstance(provenance.get("bindings"), dict)
            or not provenance["bindings"]
        ):
            raise ValueError("complete typed derived-validation provenance required")
        for name, expected in provenance["bindings"].items():
            path = Path(name).resolve(strict=True)
            path.relative_to(Path("/scratch/djy8hg").resolve())
            if sha(path) != expected:
                raise ValueError("actual GPU input/output changed after independent replay")
            bindings[str(path)] = expected
        bindings[str(provenance_path)] = sha(provenance_path)
    bindings[str(whole_gate)] = sha(whole_gate)
    bindings[str(implementation)] = sha(implementation)
    for sample in inputs["samples"]:
        if set(sample) != {"sample_id", "prompt", "media"}:
            raise ValueError("input ledger may not expose benchmark labels or references")
        for media in sample["media"]:
            path = Path(media["path"]).resolve(strict=True)
            path.relative_to(Path("/scratch/djy8hg").resolve())
            if media["kind"] != "image" or sha(path) != media["sha256"]:
                raise ValueError("pinned consumed media changed")
            bindings[str(path)] = media["sha256"]
    root = Path(root).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if not root.is_relative_to(allowed) or root == allowed or root.exists():
        raise ValueError("new immutable scratch output root required")
    root.mkdir(parents=True, exist_ok=False)
    (root / "jobs").mkdir()
    (root / "slurm").mkdir()
    write_once(root / "inputs.json", inputs)
    write_once(root / "resolved_config.json", config)
    write_once(root / "recovered_numerical_evidence.json", recovery)
    write_once(
        root / "plan.json",
        dict(
            source_sha=revision,
            execution_base_proof=base_proof,
            policy=POLICY,
            policy_sha256=digest(POLICY),
            bindings=bindings,
            whole_gate=str(whole_gate),
            expected_workers=2,
            inputs_sha256=sha(root / "inputs.json"),
            config_sha256=sha(root / "resolved_config.json"),
            recovery_sha256=sha(root / "recovered_numerical_evidence.json"),
        ),
    )


def _tensor_sha(tensor):
    import torch

    host = tensor.detach().contiguous().view(torch.uint8).cpu()
    value = hashlib.sha256()
    value.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    value.update(host.numpy().tobytes())
    return value.hexdigest()


def cache_parity(adapter, left_state, right_state):
    import torch

    left, right = (
        adapter._legacy_layers(state.past_key_values)[0] for state in (left_state, right_state)
    )
    if not left or len(left) != len(right):
        raise ValueError("same nonempty native cache layer count required")
    layers = []
    for index, (pair_a, pair_b) in enumerate(zip(left, right, strict=True)):
        components = []
        for name, a, b in zip(("K", "V"), pair_a, pair_b, strict=True):
            if a.shape != b.shape or a.dtype != b.dtype or a.ndim != 4:
                raise ValueError("native cache geometry or precision changed")
            finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            equal = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            components.append(
                dict(
                    component=name,
                    shape=list(a.shape),
                    dtype=str(a.dtype),
                    finite=finite,
                    reference_sha256=_tensor_sha(a),
                    candidate_sha256=_tensor_sha(b),
                    bitwise_equal=equal,
                    maximum_absolute_difference=0.0
                    if equal
                    else float((a.float() - b.float()).abs().max()),
                )
            )
        layers.append(dict(layer=index, components=components))
    return dict(
        layers=layers,
        bitwise_equal=all(
            c["finite"] and c["bitwise_equal"] and c["reference_sha256"] == c["candidate_sha256"]
            for layer in layers
            for c in layer["components"]
        ),
        maximum_absolute_difference=max(
            c["maximum_absolute_difference"] for layer in layers for c in layer["components"]
        ),
    )


def _forward_identities(adapter):
    import torch
    from transformers.models.mistral import modeling_mistral

    modules = [
        m
        for m in adapter.model.modules()
        if isinstance(m, torch.nn.Linear) or type(m) is modeling_mistral.MistralRMSNorm
    ]
    return (
        *(getattr(m.forward, "__func__", m.forward) for m in modules),
        modeling_mistral.eager_attention_forward,
    )


def run_paths(adapter, prepared, *, predictions=128):
    import torch

    from mosaickv.evaluation.fox_boundary import hold_back_last_text_token
    from mosaickv.evaluation.fox_shape_transport import (
        padded_prefix,
        remove_dummy_cache_slot,
        shape_matched_transport,
    )

    if type(predictions) is not int or not 2 <= predictions <= 128:
        raise ValueError("bounded 2..128 prediction diagnostic required")
    n = prepared.logical_sequence_length
    prefix, held = hold_back_last_text_token(prepared)
    if prefix.logical_sequence_length != n - 1:
        raise ValueError("exactly one original final text token must be held back")
    original_methods = _forward_identities(adapter)
    with torch.inference_mode(), ExitStack() as stack:
        operations = stack.enter_context(
            shape_matched_transport(adapter, n, native_norm_boundary=True)
        )
        reference = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
        padded = padded_prefix(prefix)
        prefix_output = adapter.prefill(padded, capture_queries=False, capture_attentions=False)
        state_b = remove_dummy_cache_slot(adapter, prefix_output.state, prefix)
        del prefix_output, padded
        if state_b.active_cache_length != n - 1 or state_b.next_decode_position != n - 1:
            raise ValueError("dummy cache slot or position remains active")
        with operations.boundary_normalization():
            corrected = adapter.decode_one_token(
                held, state_b, capture_queries=False, capture_attentions=False
            )
        if any(
            value != n
            for value in (
                corrected.state.active_cache_length,
                corrected.state.logical_sequence_length,
                corrected.state.next_decode_position,
            )
        ):
            raise ValueError("held-token state has duplicated or shifted positions")
        initial = cache_parity(adapter, reference.state, corrected.state)
        reference_logits = [reference.logits.detach().reshape(-1).float().cpu()]
        candidate_logits = [corrected.logits.detach().reshape(-1).float().cpu()]
        state_a, state_b = reference.state, corrected.state
        token = reference.next_token_id
        stack.close()
        restored = _forward_identities(adapter) == original_methods
        if not restored:
            raise RuntimeError("boundary wrappers leaked into ordinary continuation")
        counts = dict(operations)
        for _ in range(predictions - 1):
            a = adapter.decode_one_token(
                token, state_a, capture_queries=False, capture_attentions=False
            )
            b = adapter.decode_one_token(
                token, state_b, capture_queries=False, capture_attentions=False
            )
            state_a, state_b, token = a.state, b.state, a.next_token_id
            reference_logits.append(a.logits.detach().reshape(-1).float().cpu())
            candidate_logits.append(b.logits.detach().reshape(-1).float().cpu())
        if dict(operations) != counts or _forward_identities(adapter) != original_methods:
            raise RuntimeError("corrected boundary operators ran during continuation")
        final = cache_parity(adapter, state_a, state_b)
        expected_position = n + predictions - 1
        if any(
            value != expected_position
            for state in (state_a, state_b)
            for value in (
                state.active_cache_length,
                state.logical_sequence_length,
                state.next_decode_position,
            )
        ):
            raise ValueError("ordinary continuation changed logical/cache positions")
    return dict(
        reference_logits=torch.stack(reference_logits),
        candidate_logits=torch.stack(candidate_logits),
        initial_cache=initial,
        final_cache=final,
        operation_counts=counts,
        ordinary_methods_restored_before_continuation=restored,
        held_token_forward_count=1,
        ordinary_continuation_steps=predictions - 1,
        original_prompt_length=n,
        final_logical_position=expected_position,
        reference_prefix_forced_on_both_paths=True,
    )


def summarize(record, *, predictions=128, layers=32):
    import torch

    a, b = record["reference_logits"], record["candidate_logits"]
    if (
        a.ndim != 2
        or a.shape != b.shape
        or a.shape[0] != predictions
        or a.dtype != torch.float32
        or b.dtype != torch.float32
        or not torch.isfinite(a).all()
        or not torch.isfinite(b).all()
    ):
        raise ValueError("complete finite FP32 prediction trajectories required")
    counts = record["operation_counts"]
    expected = dict(
        observed_prefill_layers=layers,
        attention_padded_calls=layers,
        linear_padded_calls=7 * layers,
        normalization_expected_calls=2 * layers + 1,
        normalization_padded_calls=2 * layers + 1,
        normalization_input_calls=layers,
        normalization_post_attention_calls=layers,
        normalization_final_calls=1,
        normalization_phase_entries=1,
        normalization_phase_completions=1,
    )
    if any(
        type(counts.get(key)) is not int or counts[key] != value for key, value in expected.items()
    ):
        raise ValueError("native boundary scopes/call counts differ from registered V9 policy")
    if (
        record["ordinary_methods_restored_before_continuation"] is not True
        or record["held_token_forward_count"] != 1
        or record["ordinary_continuation_steps"] != predictions - 1
        or record["reference_prefix_forced_on_both_paths"] is not True
        or record["final_logical_position"] != record["original_prompt_length"] + predictions - 1
    ):
        raise ValueError("held-token/ordinary-continuation execution scope changed")
    for phase in ("initial_cache", "final_cache"):
        cache = record[phase]
        if len(cache["layers"]) != layers or [layer["layer"] for layer in cache["layers"]] != list(
            range(layers)
        ):
            raise ValueError("initial/final native cache hash coverage incomplete")
        all_equal = True
        maximum = 0.0
        expected_length = record["original_prompt_length"] + (
            predictions - 1 if phase == "final_cache" else 0
        )
        for layer in cache["layers"]:
            if [c["component"] for c in layer["components"]] != ["K", "V"]:
                raise ValueError("both native cache components must be covered")
            for component in layer["components"]:
                hashes = (component["reference_sha256"], component["candidate_sha256"])
                if (
                    component["shape"] != [1, 8, expected_length, 128]
                    or component["dtype"] != "torch.float16"
                    or type(component["bitwise_equal"]) is not bool
                    or type(component["finite"]) is not bool
                    or any(
                        not isinstance(value, str)
                        or len(value) != 64
                        or any(character not in "0123456789abcdef" for character in value)
                        for value in hashes
                    )
                ):
                    raise ValueError("registered FP16 GQA cache geometry/hash coverage changed")
                difference = component["maximum_absolute_difference"]
                if type(difference) not in (int, float) or not 0 <= difference < float("inf"):
                    raise ValueError("nonfinite cache difference")
                same = component["reference_sha256"] == component["candidate_sha256"]
                if same != component["bitwise_equal"] or (same and difference != 0):
                    raise ValueError("cache hashes disagree with bitwise comparison")
                all_equal = all_equal and component["finite"] is True and same
                maximum = max(maximum, difference)
        if (
            cache["bitwise_equal"] is not all_equal
            or cache["maximum_absolute_difference"] != maximum
        ):
            raise ValueError("cache hash summary inconsistent")
    differences = (a - b).abs().amax(-1)
    agreement = int((a.argmax(-1) == b.argmax(-1)).sum())
    maximum = float(differences.max())
    return dict(
        predictions=predictions,
        prediction_agreement=agreement,
        maximum_logit_difference=maximum,
        per_prediction_maximum_logit_difference=differences.tolist(),
        bitwise_logits=bool(torch.equal(a, b)),
        initial_cache_bitwise_equal=record["initial_cache"]["bitwise_equal"],
        final_cache_bitwise_equal=record["final_cache"]["bitwise_equal"],
        ordinary_continuation_steps=predictions - 1,
        pass_128=agreement == predictions
        and maximum <= POLICY["tolerance"]
        and record["initial_cache"]["bitwise_equal"]
        and record["final_cache"]["bitwise_equal"],
        unchanged_absolute_tolerance=POLICY["tolerance"],
    )


def worker(root, index):
    from dataclasses import asdict

    import torch
    from evicastkv_campaign import environment
    from PIL import Image

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = validate(root)
    if index not in (0, 1):
        raise ValueError("only two registered consumed cases")
    if (
        sha(root / "inputs.json") != plan["inputs_sha256"]
        or sha(root / "resolved_config.json") != plan["config_sha256"]
    ):
        raise ValueError("prepared input/config changed")
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    errors, summary = [], None
    started = time.monotonic()
    try:
        if (
            not os.environ.get("SLURM_JOB_ID")
            or torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
        ):
            raise ValueError("one allocated A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUBLAS policy must precede Python")
        flags = dict(
            allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            allow_fp16_accumulation=torch.backends.cuda.matmul.allow_fp16_accumulation,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        )
        if flags != dict(
            allow_fp16_reduced_precision_reduction=True,
            allow_fp16_accumulation=False,
            allow_tf32=False,
        ):
            raise ValueError("original native arithmetic flags changed")
        torch.set_num_threads(12)
        torch.manual_seed(POLICY["seed"])
        torch.use_deterministic_algorithms(True)
        config = read(root / "resolved_config.json")
        cfg = RunConfig.from_mapping(config)
        if config["model"] != MODEL:
            raise ValueError("pinned FP16/GQA Mistral checkpoint required")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        if any(
            p.device.type != "cuda" or p.dtype != torch.float16 for p in adapter.model.parameters()
        ):
            raise ValueError("native single-GPU FP16 model required")
        gpu = asdict(capture_gpu_environment(torch))
        visible = [d for d in gpu["devices"] if d["visible_to_process"]]
        if len(visible) != 1 or gpu["query_error"] or gpu["background_processes"]:
            raise ValueError("A100 isolation cannot be established")
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                plan_sha256=sha(root / "plan.json"),
                config_sha256=plan["config_sha256"],
                model=MODEL,
                dataset=config["dataset"],
                backend="huggingface",
                attention="eager",
                seed=POLICY["seed"],
                environment=environment(),
                GPU=gpu,
                arithmetic_flags=flags,
                **scheduler_identity({}),
                benchmark_labels_read=False,
                measurement_type="consumed_COCO_boundary_correctness_not_quality_or_latency",
                vllm="not_used",
                sglang="not_used",
                source_reconstruction_enabled=False,
            ),
        )
        sample = read(root / "inputs.json")["samples"][index]
        media = []
        for item in sample["media"]:
            path = Path(item["path"]).resolve(strict=True)
            if sha(path) != item["sha256"]:
                raise ValueError("consumed media changed")
            with Image.open(path) as image:
                media.append(MediaItem(MediaKind.IMAGE, image.convert("RGB").copy()))
        prepared = adapter.prepare_inputs(build_multimodal_messages(sample["prompt"], tuple(media)))
        record = run_paths(adapter, prepared, predictions=POLICY["predictions"])
        summary = summarize(record)
        parent = read(SHAPE_ROOT / "GATE_shape_transport.json")["reports"][index]["continuation"]
        expected = parent["reference_token_ids"]
        if record["reference_logits"][:16].argmax(-1).tolist() != expected:
            raise ValueError("first sixteen native reference tokens differ from passed V9")
        with (out / "trajectories.pt").open("xb") as stream:
            torch.save(record, stream)
        write_once(out / "summary.json", summary)
    except BaseException as exc:
        errors.append(
            dict(
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                worker=index,
            )
        )
    finally:
        outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        write_once(
            out / "result.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256=sha(root / "plan.json"),
                index=index,
                sample_id=CONSUMED_IDS[index],
                failures=errors,
                outputs=outputs,
                summary=summary,
                wall_seconds=time.monotonic() - started,
                **scheduler_identity({}),
            ),
        )


def analyze(root):
    import torch

    root = Path(root)
    plan = validate(root)
    errors, reports = [], []
    for index in (0, 1):
        try:
            out = root / f"workers/{index}"
            result = read(out / "result.json")
            if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != sha(
                root / "plan.json"
            ):
                raise ValueError("worker producer provenance changed")
            if (
                result["failures"]
                or result["index"] != index
                or result["sample_id"] != CONSUMED_IDS[index]
            ):
                raise ValueError(f"worker incomplete: {result['failures']}")
            for name, expected in result["outputs"].items():
                if sha(out / name) != expected:
                    raise ValueError(f"worker output changed: {name}")
            record = torch.load(out / "trajectories.pt", map_location="cpu", weights_only=True)
            summary = summarize(record)
            if digest(summary) != digest(result["summary"]) or digest(summary) != digest(
                read(out / "summary.json")
            ):
                raise ValueError("independent numerical summary differs")
            reports.append(dict(index=index, sample_id=CONSUMED_IDS[index], **summary))
        except BaseException as exc:
            errors.append(dict(index=index, error=f"{type(exc).__name__}: {exc}"))
    complete = not errors and len(reports) == 2
    gate = dict(
        source_sha=plan["source_sha"],
        plan_sha256=sha(root / "plan.json"),
        complete=complete,
        source_scope="FP16_GQA_COCO_FullKV_evaluation_only_boundary_control",
        reports=reports,
        failures=errors,
        corrected_boundary_128_pass=complete and all(r["pass_128"] is True for r in reports),
        historical_native_boundary_failure_preserved=True,
        production_defaults_changed=False,
        source_reconstruction_validated=False,
        quality_or_latency_claim_allowed=False,
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "analysis/GATE_coco_boundary_v28.json", gate)
    return gate


def submit(root, revision, label, mode, *, dependency=None, gpu=False):
    root = Path(root)
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox28-{label}",
        f"--partition={'bii-gpu' if gpu else 'interactive'}",
        "--cpus-per-task=12",
        f"--mem={'64G' if gpu else '24G'}",
        f"--time=00:{15 if gpu else 5}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if gpu:
        command += ["--gres=gpu:a100:1", "--reservation=bi_fox_dgx", "--array=0-1%2"]
    if dependency is not None:
        if not str(dependency).isdigit():
            raise ValueError("numeric acknowledged afterany dependency required")
        command.append(f"--dependency=afterany:{dependency}")
    command += [
        str(SOURCE / "mosaickv/slurm/fox_coco_boundary_v28.sbatch"),
        str(SOURCE),
        str(root),
        mode,
        revision,
    ]
    write_once(root / f"jobs/{label}-request.json", dict(command=command, source_sha=revision))
    job = subprocess.check_output(command, text=True, timeout=30).strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("uncertain scheduler acknowledgement; do not duplicate")
    write_once(
        root / f"jobs/{label}-submission.json", dict(command=command, job=job, source_sha=revision)
    )
    return job


def launch(root, whole_gate):
    revision = clean_source(SOURCE)
    modules = subprocess.check_output(
        ["git", "submodule", "status", "--recursive"], cwd=SOURCE, text=True
    )
    if any(line[:1] in ("-", "+", "U") for line in modules.splitlines()):
        raise ValueError("all pinned submodules must be initialized unchanged")
    queue = subprocess.check_output(
        ["squeue", "--me", "--noheader", "--states=RUNNING,PENDING", "--format=%b"], text=True
    )
    active = sum(int(line.strip().split(":")[-1]) for line in queue.splitlines() if "gpu:" in line)
    if active > 4:
        raise ValueError("two-worker admission could exceed six aggregate GPU allocations")
    prepare(root, whole_gate)
    worker_job = submit(root, revision, "gpu", "worker", gpu=True)
    analysis_job = submit(root, revision, "analysis", "analyze", dependency=worker_job)
    print(json.dumps(dict(workers=worker_job, analysis=analysis_job, root=str(root))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--whole-gate", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode in ("prepare", "launch"):
        if args.whole_gate is None:
            parser.error("whole-gate prerequisite is required")
        globals()[args.mode](args.root, args.whole_gate)
    elif args.mode == "worker":
        if args.index not in (0, 1):
            parser.error("worker requires index0or1")
        worker(args.root, args.index)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
