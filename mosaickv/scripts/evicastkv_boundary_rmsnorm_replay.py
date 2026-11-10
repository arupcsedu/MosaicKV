#!/usr/bin/env python3
"""One native RMSNorm shape replay, on two already-consumed saved vectors.

No full VLM, precision change, output substitution, cache modification, new
cohort, or performance claim. Only one checkpoint norm weight is read locally.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import traceback
from dataclasses import asdict
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
MODEL = dict(
    id="llava-hf/llava-v1.6-mistral-7b-hf",
    revision="2424fdd47412fccc66d91719126b420e9fbd7065",
    precision="fp16",
)
WEIGHT_KEY = "language_model.model.layers.29.input_layernorm.weight"
OPERATOR = "layer.29.input_layernorm"
POLICY = dict(
    id="consumed_native_layer29_RMSNorm_shape_replay_v1",
    operator=OPERATOR,
    weight_key=WEIGHT_KEY,
    predictions_generated=0,
    new_model_or_benchmark_labels=False,
    original_forward_only=True,
    original_precision="torch.float16",
    required_hidden_width=4096,
    original_logit_gate_absolute_tolerance=1e-4,
    operator_comparison="bitwise_against_both_saved_A_and_B_and_report_max_abs",
    branches=["single_compact", "single_same_buffer_view", "full_rows"],
    repetitions=3,
    interpretation="operator_localization_not_full_model_parity_or_superiority",
    full_model_transport_automatic_release=False,
    changes_to_original_negative_gates=False,
    weights_or_epsilon_changed=False,
    original_GPU_input_layout="not_saved_recapture_original_layout_if_replay_is_inconclusive",
    scratch_buffer="zero_other_rows_last_row_copies_same_saved_vector",
    intermediate_observations="FP32_variance_and_rsqrt_measured_separately_never_substituted",
    maximum_A100s_total=6,
    worker_gpus=1,
    worker_cpus=12,
    worker_wall_minutes=20,
)


def read(path):
    return json.loads(Path(path).read_text())


def tensor_sha(tensor):
    import torch

    return hashlib.sha256(
        str((tuple(tensor.shape), str(tensor.dtype))).encode()
        + tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def scratch_path(path):
    value = Path(path).resolve(strict=True)
    value.relative_to(Path("/scratch/djy8hg").resolve())
    return value


def installed_sources():
    names = (
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.mistral.configuration_mistral",
        "transformers.models.llava_next.modeling_llava_next",
        "transformers.integrations.hub_kernels",
    )
    result = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"missing installed native implementation: {name}")
        result[name] = dict(path=spec.origin, sha256=sha(Path(spec.origin)))
    return result


def validate_model(config):
    if config.get("model") != MODEL:
        raise ValueError("only the pinned FP16 LLaVA-v1.6-Mistral model is registered")
    execution = config.get("execution", {})
    if (
        execution.get("backend") != "huggingface"
        or execution.get("attention_implementation") != "eager"
    ):
        raise ValueError("registered parent is HF eager only")
    if config.get("method") != "full_kv" or config.get("cache", {}).get("retention_ratio") != 1.0:
        raise ValueError("original FullKV retention-one parent required")


def checkpoint_inventory(config, *, cache=None):
    """Read only index/config and the selected tensor; never from_pretrained."""
    import torch
    from safetensors import safe_open
    from transformers import MistralConfig
    from transformers.models.llava_next.modeling_llava_next import LlavaNextForConditionalGeneration

    validate_model(config)
    cache = scratch_path(
        cache or os.environ.get("HF_HUB_CACHE", "/scratch/djy8hg/cache/mosaickv/huggingface/hub")
    )
    snapshot = (
        cache / ("models--" + MODEL["id"].replace("/", "--")) / "snapshots" / MODEL["revision"]
    )
    index_path, config_path = snapshot / "model.safetensors.index.json", snapshot / "config.json"
    for path in (index_path, config_path):
        scratch_path(path).relative_to(cache)
    index, hf_config = read(index_path), read(config_path)
    if (
        hf_config.get("model_type") != "llava_next"
        or hf_config.get("text_config", {}).get("model_type") != "mistral"
    ):
        raise ValueError("cached checkpoint is not the registered Mistral architecture")
    native = MistralConfig(**hf_config["text_config"])
    if (
        native.hidden_size != 4096
        or native.num_hidden_layers != 32
        or native.num_key_value_heads != 8
        or not math.isfinite(native.rms_norm_eps)
        or native.rms_norm_eps <= 0
    ):
        raise ValueError("unexpected Mistral width/layer/GQA/epsilon geometry")
    mapping = LlavaNextForConditionalGeneration._checkpoint_conversion_mapping
    if mapping.get("^language_model.model") != "model.language_model":
        raise ValueError("installed checkpoint-to-captured-module mapping changed")
    filename = index.get("weight_map", {}).get(WEIGHT_KEY)
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".safetensors")
    ):
        raise ValueError("selected norm weight missing or unsafe shard name")
    shard = scratch_path(snapshot / filename)
    shard.relative_to(cache)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        # safe_open is not a mapping; its explicit key API is authoritative.
        keys = handle.keys()
        if WEIGHT_KEY not in keys:
            raise ValueError("index names a shard without the selected norm weight")
        weight = handle.get_tensor(WEIGHT_KEY)
    if weight.shape != (4096,) or weight.dtype != torch.float16 or not torch.isfinite(weight).all():
        raise ValueError("only exact finite 4096-wide original FP16 norm weight is registered")
    return dict(
        model=MODEL,
        cache=str(cache),
        checkpoint_index=str(index_path),
        checkpoint_index_sha256=sha(index_path),
        checkpoint_config=str(config_path),
        checkpoint_config_sha256=sha(config_path),
        shard=str(shard),
        weight_key=WEIGHT_KEY,
        weight_dtype=str(weight.dtype),
        weight_shape=list(weight.shape),
        weight_tensor_sha256=tensor_sha(weight),
        epsilon=native.rms_norm_eps,
        captured_module_path="model.language_model.layers.29.input_layernorm",
        checkpoint_conversion_mapping=mapping,
        tensor_count_read=1,
        full_model_loaded=False,
        only_selected_tensor_materialized=True,
        safetensors_shard_mapped_without_materializing_other_weights=True,
    ), weight


def selected_case(capture):
    import torch

    result = {}
    for side in ("A", "B"):
        for kind in ("input", "output"):
            entry = capture["captures"][side][f"{OPERATOR}:{kind}"]
            tensor, source_shape = entry["vector"], entry["source_shape"]
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.float16
                or tuple(tensor.shape) != (1, 1, 4096)
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("finite CPU FP16 final norm vector required")
            if (
                len(source_shape) != 3
                or source_shape[0] != 1
                or source_shape[-1] != 4096
                or not 1 <= source_shape[1] <= 4096
            ):
                raise ValueError("registered norm source shape is invalid")
            result[f"{side}_{kind}"] = tensor.clone()
            result[f"{side}_{kind}_source_shape"] = source_shape
            result[f"{side}_{kind}_saved_CPU_stride"] = list(tensor.stride())
    if not torch.equal(result["A_input"].view(torch.int16), result["B_input"].view(torch.int16)):
        raise ValueError("native operator replay requires bitwise identical saved A/B inputs")
    if result["A_input_source_shape"][1] <= 1 or result["B_input_source_shape"][1] != 1:
        raise ValueError("full-prompt-versus-one-token operator boundary required")
    result["original_GPU_layout_known"] = False
    return result


def parent_evidence(parent):
    import evicastkv_boundary_operators as boundary
    import torch

    parent = scratch_path(parent)
    protocol = boundary._load_protocol(parent)
    result, gate, environment = (
        read(parent / name)
        for name in ("result.json", "GATE_shape_transport.json", "environment.json")
    )
    if (
        result["source_sha"] != protocol["source_sha"]
        or result["protocol_sha256"] != sha(parent / "protocol.json")
        or result["environment_sha256"] != sha(parent / "environment.json")
    ):
        raise ValueError("parent worker identity drift")
    if (
        gate["status"] != "blocked_new_shape_transport"
        or not gate["complete"]
        or gate["unchanged_tolerance"] != 1e-4
        or gate["historical_native_gate_still_negative"] is not True
        or gate["source_sha"] != protocol["source_sha"]
    ):
        raise ValueError("preserved complete negative shape-transport gate required")
    config = read(parent / "resolved_config.json")
    validate_model(config)
    if environment["model"] != MODEL or environment["source_sha"] != protocol["source_sha"]:
        raise ValueError("parent numerical model/source mismatch")
    expected = read(parent / "inputs.json")["samples"]
    records = result["records"]
    if (
        len(records) != 2
        or [r["index"] for r in records] != [0, 1]
        or [r["sample_id"] for r in records] != [s["sample_id"] for s in expected]
    ):
        raise ValueError("exactly both consumed parent inputs required")
    bound = {
        str(parent / name): sha(parent / name)
        for name in (
            "protocol.json",
            "result.json",
            "GATE_shape_transport.json",
            "environment.json",
            "resolved_config.json",
            "inputs.json",
            "transport.json",
        )
    }
    cases = []
    for record in records:
        path = scratch_path(record["capture_path"])
        path.relative_to(parent)
        if (
            record["status"] != "completed"
            or sha(path) != record["capture_sha256"]
            or record["input_sha256"] != digest(expected[record["index"]])
        ):
            raise ValueError("parent saved operator capture changed")
        capture = torch.load(path, map_location="cpu", weights_only=True)
        selected = selected_case(capture)
        selected.update(
            index=record["index"],
            sample_id=record["sample_id"],
            parent_capture_path=str(path),
            parent_capture_sha256=sha(path),
        )
        cases.append(selected)
        bound[str(path)] = sha(path)
    return config, cases, bound


def prepare(root, parent):
    import evicastkv_boundary_operators as boundary

    root = Path(root)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch replay root required")
    source_sha = clean_source(SOURCE)
    config, cases, bound = parent_evidence(parent)
    checkpoint, _weight = checkpoint_inventory(config)
    files = installed_sources()
    root.mkdir(parents=True)
    boundary._save_tensor_once(root / "selected_inputs.pt", dict(cases=cases))
    protocol = dict(
        source_sha=source_sha,
        policy=POLICY,
        policy_sha256=digest(POLICY),
        parent=str(Path(parent).resolve()),
        bindings=bound,
        checkpoint=checkpoint,
        installed_sources=files,
        config=config,
        source_implementation_sha256=sha(Path(__file__)),
        selected_inputs_sha256=sha(root / "selected_inputs.pt"),
        expected_cases=2,
        no_new_GPU_model_predictions=True,
        no_full_model_auto_release=True,
    )
    write_once(root / "protocol.json", protocol)
    return dict(workers=1, cases=2, protocol_sha256=sha(root / "protocol.json"))


def validate(root):
    protocol = read(root / "protocol.json")
    clean_source(SOURCE, protocol["source_sha"])
    if (
        protocol["policy"] != POLICY
        or protocol["policy_sha256"] != digest(POLICY)
        or protocol["source_implementation_sha256"] != sha(Path(__file__))
    ):
        raise ValueError("native norm diagnostic policy/implementation changed")
    for path, expected in protocol["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("native norm parent binding changed")
    for item in protocol["installed_sources"].values():
        if sha(Path(item["path"])) != item["sha256"]:
            raise ValueError("installed norm implementation changed")
    if sha(root / "selected_inputs.pt") != protocol["selected_inputs_sha256"]:
        raise ValueError("selected consumed input vectors changed")
    checkpoint, weight = checkpoint_inventory(
        protocol["config"], cache=protocol["checkpoint"]["cache"]
    )
    if checkpoint != protocol["checkpoint"]:
        raise ValueError("selected original norm weight/config/index changed")
    return protocol, weight


def tensor_layout(tensor):
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        storage_offset=tensor.storage_offset(),
        data_pointer_mod_256=tensor.data_ptr() % 256,
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        storage_bytes=tensor.untyped_storage().nbytes(),
        contiguous=tensor.is_contiguous(),
    )


def replay(norm, vector, rows):
    """Call the untouched module on the same input at different leading shapes."""
    import torch

    if (
        vector.ndim != 3
        or tuple(vector.shape[:2]) != (1, 1)
        or vector.dtype != torch.float16
        or not torch.isfinite(vector).all()
        or type(rows) is not int
        or not 2 <= rows <= 4096
    ):
        raise ValueError("finite one-row native FP16 input and full row count required")
    if (
        norm.weight.dtype != vector.dtype
        or norm.weight.device != vector.device
        or norm.weight.shape != (vector.shape[-1],)
    ):
        raise ValueError("native norm weight shape/device/precision mismatch")
    before_weight, before_input = tensor_sha(norm.weight), tensor_sha(vector)
    full = torch.zeros((1, rows, vector.shape[-1]), dtype=vector.dtype, device=vector.device)
    full[:, -1:, :].copy_(vector)
    variants = dict(
        single_compact=vector.clone().contiguous(),
        single_same_buffer_view=full[:, -1:, :],
        full_rows=full,
    )
    results = {}
    with torch.inference_mode():
        for name, value in variants.items():
            outputs = [
                norm(value)[:, -1:, :].detach().cpu().clone() for _ in range(POLICY["repetitions"])
            ]
            if any(
                not torch.equal(outputs[0].view(torch.int16), item.view(torch.int16))
                for item in outputs[1:]
            ):
                raise ValueError("original native norm replay was not repeatable")
            # These are separately observed original FP32 operators, not a
            # replacement forward or alternative precision policy.
            squared = value.to(torch.float32).pow(2)
            variance = squared.mean(-1, keepdim=True)
            inverse = torch.rsqrt(variance + norm.variance_epsilon)
            results[name] = dict(
                output=outputs[0],
                input_layout=tensor_layout(value),
                output_dtype=str(outputs[0].dtype),
                variance_last=variance[:, -1:].detach().cpu(),
                rsqrt_last=inverse[:, -1:].detach().cpu(),
                repeat_bitwise=True,
            )
    if tensor_sha(norm.weight) != before_weight or tensor_sha(vector) != before_input:
        raise ValueError("native replay mutated weights or saved input")
    return dict(
        branches=results,
        original_forward_unchanged=True,
        weight_sha256=before_weight,
        input_sha256=before_input,
        weight_unchanged=True,
        input_unchanged=True,
        epsilon=norm.variance_epsilon,
        full_input_backing_bytes_only=full.untyped_storage().nbytes(),
        complete_operator_workspace_peak_measured=False,
        memory_or_timing_claim=False,
    )


def compare(left, right):
    import torch

    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or not torch.isfinite(left).all()
        or not torch.isfinite(right).all()
    ):
        raise ValueError("matching finite native output shape/precision required")
    return dict(
        bitwise_equal=torch.equal(left.view(torch.int16), right.view(torch.int16)),
        unequal_elements=int((left.view(torch.int16) != right.view(torch.int16)).sum()),
        maximum_absolute_difference=float((left.float() - right.float()).abs().max()),
    )


def worker(root):
    import torch
    from evicastkv_campaign import environment
    from transformers.integrations.hub_kernels import is_hub_kernels_available
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    protocol, weight = validate(root)
    if (root / "result.json").exists():
        raise FileExistsError("native norm result is write-once")
    if (
        not os.environ.get("SLURM_JOB_ID")
        or torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name()
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("one Slurm A100 and deterministic prelaunch CUDA configuration required")
    if is_hub_kernels_available():
        raise ValueError(
            "only the recorded native RMSNorm, not a hub kernel replacement, is registered"
        )
    torch.set_num_threads(12)
    torch.manual_seed(protocol["config"]["execution"]["seed"])
    torch.use_deterministic_algorithms(True)
    flags = dict(
        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        allow_fp16_accumulation=torch.backends.cuda.matmul.allow_fp16_accumulation,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    if flags != read(Path(protocol["parent"]) / "environment.json")["actual_arithmetic_flags"]:
        raise ValueError("parent native arithmetic flags changed; no precision fix is permitted")
    gpu = asdict(capture_gpu_environment(torch))
    if gpu["query_error"] or gpu["background_processes"]:
        raise ValueError("GPU isolation cannot be established")
    write_once(
        root / "environment.json",
        dict(
            source_sha=protocol["source_sha"],
            environment=environment(),
            gpu=gpu,
            **scheduler_identity({}),
            cuda_runtime=torch.version.cuda,
            model=MODEL,
            dataset=protocol["config"]["dataset"],
            backend="native_HF_RMSNorm_operator_only",
            attention="not_used",
            vllm="not_used",
            sglang="not_used",
            seed=protocol["config"]["execution"]["seed"],
            measurement_type="consumed_operator_replay_not_full_model_or_performance",
            config_sha256=digest(dict(policy=POLICY, checkpoint=protocol["checkpoint"])),
            full_VLM_loaded=False,
            actual_arithmetic_flags=flags,
        ),
    )
    norm = (
        MistralRMSNorm(4096, eps=protocol["checkpoint"]["epsilon"])
        .eval()
        .to(device="cuda", dtype=torch.float16)
    )
    norm.requires_grad_(False)
    norm.load_state_dict(dict(weight=weight), strict=True)
    if tensor_sha(norm.weight) != protocol["checkpoint"]["weight_tensor_sha256"]:
        raise ValueError("GPU norm weight differs from exact cached tensor")
    cases = torch.load(root / "selected_inputs.pt", map_location="cpu", weights_only=True)["cases"]
    records = []
    import evicastkv_boundary_operators as boundary

    for case in cases:
        try:
            values = replay(
                norm, case["A_input"].to(device="cuda"), case["A_input_source_shape"][1]
            )
            torch.cuda.synchronize()
            path = root / "captures" / f"{case['index']:02d}.pt"
            boundary._save_tensor_once(path, values)
            records.append(
                dict(
                    index=case["index"],
                    sample_id=case["sample_id"],
                    status="completed",
                    capture_path=str(path),
                    capture_sha256=sha(path),
                )
            )
        except Exception as error:
            records.append(
                dict(
                    index=case["index"],
                    sample_id=case["sample_id"],
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            )
    write_once(
        root / "result.json",
        dict(
            source_sha=protocol["source_sha"],
            protocol_sha256=sha(root / "protocol.json"),
            environment_sha256=sha(root / "environment.json"),
            records=records,
        ),
    )


def analyze(root):
    import torch

    root = Path(root)
    protocol, _weight = validate(root)
    result = read(root / "result.json")
    if (
        result["source_sha"] != protocol["source_sha"]
        or result["protocol_sha256"] != sha(root / "protocol.json")
        or result["environment_sha256"] != sha(root / "environment.json")
    ):
        raise ValueError("native norm worker provenance changed")
    environment = read(root / "environment.json")
    if (
        environment["source_sha"] != protocol["source_sha"]
        or environment["model"] != MODEL
        or environment["full_VLM_loaded"] is not False
    ):
        raise ValueError("native norm environment/model scope changed")
    cases = torch.load(root / "selected_inputs.pt", map_location="cpu", weights_only=True)["cases"]
    if (
        len(result["records"]) != 2
        or [row["index"] for row in result["records"]] != [0, 1]
        or [row["sample_id"] for row in result["records"]] != [case["sample_id"] for case in cases]
    ):
        raise ValueError("complete two-case norm replay required")
    reports, failures = [], []
    for row, case in zip(result["records"], cases, strict=True):
        if row["status"] != "completed":
            failures.append(row)
            continue
        path = scratch_path(row["capture_path"])
        path.relative_to(root.resolve())
        if sha(path) != row["capture_sha256"]:
            raise ValueError("saved native norm outputs changed")
        record = torch.load(path, map_location="cpu", weights_only=True)
        if (
            set(record["branches"]) != set(POLICY["branches"])
            or record["weight_sha256"] != protocol["checkpoint"]["weight_tensor_sha256"]
            or not record["input_unchanged"]
            or not record["weight_unchanged"]
            or record["input_sha256"] != tensor_sha(case["A_input"])
            or record["original_forward_unchanged"] is not True
            or record["epsilon"] != protocol["checkpoint"]["epsilon"]
            or any(branch["repeat_bitwise"] is not True for branch in record["branches"].values())
        ):
            raise ValueError("native norm replay changed operator scope or parameters")
        checks = {
            name: {
                f"saved_{side}": compare(branch["output"], case[f"{side}_output"])
                for side in ("A", "B")
            }
            for name, branch in record["branches"].items()
        }
        reproduced = (
            checks["single_compact"]["saved_B"]["bitwise_equal"]
            and checks["full_rows"]["saved_A"]["bitwise_equal"]
        )
        reports.append(
            dict(
                index=case["index"],
                sample_id=case["sample_id"],
                comparisons=checks,
                saved_A_B=compare(case["A_output"], case["B_output"]),
                shape_pair_reproduced=reproduced,
                branch_layouts={
                    name: branch["input_layout"] for name, branch in record["branches"].items()
                },
                branch_variance_and_rsqrt={
                    name: dict(
                        variance=branch["variance_last"].tolist(),
                        rsqrt=branch["rsqrt_last"].tolist(),
                    )
                    for name, branch in record["branches"].items()
                },
            )
        )
    complete = len(reports) == 2 and not failures
    gate = dict(
        source_sha=protocol["source_sha"],
        protocol_sha256=sha(root / "protocol.json"),
        status="completed_native_norm_operator_replay"
        if complete
        else "blocked_incomplete_operator_replay",
        complete=complete,
        reports=reports,
        failures=failures,
        both_saved_shape_pairs_reproduced=complete
        and all(row["shape_pair_reproduced"] for row in reports),
        original_first_logit_tolerance_unchanged=1e-4,
        negative_native_and_shape_gates_preserved=True,
        full_model_parity_established=False,
        full_model_auto_release=False,
        superiority_or_performance_claim=False,
        original_GPU_input_layout_not_known=True,
    )
    write_once(root / "GATE_native_rmsnorm_replay.json", gate)
    write_once(
        root / "analysis_manifest.json",
        dict(
            source_sha=protocol["source_sha"],
            result_sha256=sha(root / "result.json"),
            gate_sha256=sha(root / "GATE_native_rmsnorm_replay.json"),
            policy_sha256=digest(POLICY),
            source_implementation_sha256=sha(Path(__file__)),
        ),
    )
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        print(json.dumps(prepare(args.root, args.parent)))
    elif args.mode == "worker":
        worker(args.root)
    else:
        print(json.dumps(analyze(args.root)))


if __name__ == "__main__":
    main()
