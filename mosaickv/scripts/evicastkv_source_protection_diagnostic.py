#!/usr/bin/env python3
"""Same-consumed MM-Vet protection diagnosis; never changes output or policy.

Captured operator differences localize observations, not causal downstream
attribution. Reference tokens are evaluation-only conditioning, never selector
inputs. Packet inspection is extra audit work, excluded from latency claims.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
ARMS = (
    "fullkv_native_reference",
    "source3_mandatory_only",
    "source3_random_optional",
    "source3_attention_optional",
)
PREDICTIONS = (20, 21, 22, 23)
MAX_CAPTURE_BYTES = 32 * 2**20
POLICY = dict(
    predictions=PREDICTIONS,
    capture_dtype="original_BF16_CPU",
    maximum_capture_bytes=MAX_CAPTURE_BYTES,
    output_predictions=128,
    continuation_updates=127,
    guard=dict(tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0),
    native_parent_absolute_tolerance=0.001,
    packet_source_bits=3,
    protection_fractions=[0.0, 0.03, 0.05],
    same_parent_ceiling=True,
    no_reference_access_for_selection=True,
    no_vocabulary_patch=True,
    no_policy_change=True,
    no_latency_claim=True,
    no_fresh_cohort=True,
    tie_probe_ids=[10580, 12440],
    tie_probe_prediction_zero_based=22,
)


def read(path):
    return json.loads(Path(path).read_text())


def require_isolated_slurm_gpu(torch_module, *, snapshot=None, environ=None):
    """Fail before loading weights when scheduler/device isolation is unclear."""
    from mosaickv.measurements.telemetry import capture_gpu_environment

    environment = os.environ if environ is None else environ
    job = environment.get("SLURM_JOB_ID", "")
    if not isinstance(job, str) or not job.isdigit() or int(job) < 1:
        raise ValueError("one concrete Slurm job allocation is required")
    if (
        torch_module.cuda.device_count() != 1
        or "A100" not in torch_module.cuda.get_device_name()
        or environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("one Slurm A100 and deterministic prelaunch workspace required")
    gpu = (
        dataclasses.asdict(capture_gpu_environment(torch_module)) if snapshot is None else snapshot
    )
    visible = [device for device in gpu.get("devices", ()) if device.get("visible_to_process")]
    if (
        gpu.get("query_error")
        or gpu.get("background_processes")
        or (
            len(visible) != 1
            or "A100" not in visible[0].get("name", "")
            or gpu.get("visible_process_concurrency") not in (0, 1)
        )
    ):
        raise ValueError("GPU telemetry failed or the assigned A100 has background activity")
    return gpu


def tensor_hash(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("actual dense tensor required for packet hash")
    host = tensor.detach().contiguous().cpu()
    if host.is_floating_point() and not torch.isfinite(host).all():
        raise ValueError("nonfinite packet tensor")
    value = hashlib.sha256()
    value.update(str((str(host.dtype), tuple(host.shape))).encode())
    value.update(host.view(torch.uint8).numpy().tobytes())
    return dict(
        sha256=value.hexdigest(),
        dtype=str(host.dtype),
        shape=list(host.shape),
        CPU_hash_staging_bytes=host.numel() * host.element_size(),
    )


class Captures:
    def __init__(self, maximum=MAX_CAPTURE_BYTES):
        self.maximum, self.bytes, self.values = maximum, 0, {}

    def save(self, key, value, *, vector=False):
        import torch

        if not isinstance(value, torch.Tensor):
            raise ValueError("capture must be an actual tensor")
        if vector:
            if value.ndim != 3 or value.shape[0] != 1 or value.dtype != torch.bfloat16:
                raise ValueError("one-batch original BF16 hidden vector required")
            value = value[0, -1]
        if key in self.values or value.numel() * value.element_size() + self.bytes > self.maximum:
            raise ValueError("duplicate capture or preregistered CPU capture-byte limit exceeded")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("nonfinite activation")
        host = value.detach().contiguous().cpu().clone()
        self.values[key] = host
        self.bytes += host.numel() * host.element_size()


def first_tensor(value):
    import torch

    result = value[0] if isinstance(value, (tuple, list)) else value
    if not isinstance(result, torch.Tensor):
        raise ValueError("model operator did not expose a tensor")
    return result


@contextmanager
def capture_operators(adapter, arm, captures):
    """Observe only four fixed decode prefixes; hooks never modify outputs."""
    layers = tuple(adapter._language_layers())
    parents = [
        module
        for module in adapter.model.modules()
        if getattr(module, "layers", None) is not None and tuple(module.layers) == layers
    ]
    if len(parents) != 1 or not hasattr(parents[0], "norm"):
        raise ValueError("unique language decoder/final normalization required")
    state = dict(prediction=0, updates=0)
    original_prefill, original_decode = adapter.prefill, adapter.decode_one_token

    def prefill(*args, **kwargs):
        state["prediction"] = 0
        return original_prefill(*args, **kwargs)

    def decode(*args, **kwargs):
        state["updates"] += 1
        state["prediction"] = state["updates"]
        return original_decode(*args, **kwargs)

    def save(operator, value):
        if state["prediction"] in PREDICTIONS:
            captures.save(f"{arm}|{state['prediction']}|{operator}", value, vector=True)

    with ExitStack() as stack:
        stack.enter_context(patch.object(adapter, "prefill", prefill))
        stack.enter_context(patch.object(adapter, "decode_one_token", decode))
        for index, layer in enumerate(layers):

            def before(_module, args, kwargs, *, name=f"layer{index}.input"):
                save(name, kwargs.get("hidden_states", args[0] if args else None))

            def attention(_module, _args, output, *, name=f"layer{index}.attention_output"):
                save(name, first_tensor(output))

            def residual(_module, args, *, name=f"layer{index}.post_attention_residual"):
                save(name, first_tensor(args))

            def after(_module, _args, output, *, name=f"layer{index}.output"):
                save(name, first_tensor(output))

            for handle in (
                layer.register_forward_pre_hook(before, with_kwargs=True),
                layer.self_attn.register_forward_hook(attention),
                layer.post_attention_layernorm.register_forward_pre_hook(residual),
                layer.register_forward_hook(after),
            ):
                stack.callback(handle.remove)
        handle = parents[0].norm.register_forward_hook(
            lambda module, args, output: save("final_norm", first_tensor(output))
        )
        stack.callback(handle.remove)
        yield state
        if state["updates"] != 127:
            raise ValueError("operator capture did not observe all 127 original decode updates")


def inspect_packets(adapter, prefill, formats, arm, captures):
    """Inspect chosen packets only, while original prefill K/V is still live."""
    import torch

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    originals, _ = adapter._legacy_layers(prefill.state.past_key_values)
    layers = tuple(adapter._language_layers())
    if len(formats) != len(originals) or len(formats) != len(layers):
        raise ValueError("chosen packets do not cover every source layer")
    result = []
    for index, (packet, original, layer) in enumerate(zip(formats, originals, layers, strict=True)):
        if getattr(packet, "kind", None) != "evicastkv_full_prompt_shared_source_precision_variant":
            raise ValueError("diagnostic requires the original shared-source packet family")
        hashes = {
            name: tensor_hash(getattr(packet.source, name))
            for name in ("codes", "minimum", "scale")
        }
        hashes.update(
            {name: tensor_hash(getattr(packet, name)) for name in ("cos", "sin", "union", "slots")}
        )
        rows = packet.row_positions
        if rows.dtype != torch.int32 or rows.ndim != 1 or rows.unique().numel() != rows.numel():
            raise ValueError("unique actual int32 protection positions required")
        h, n = original[0].shape[1:3]
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= h * n):
            raise ValueError("protected row position exceeds original source")
        captures.save(f"{arm}|packet_rows|layer{index}", rows)
        before = int(torch.cuda.memory_allocated()) if rows.is_cuda else None
        restored = packet.materialize(layer.self_attn)
        indices = rows.long()
        selected = tuple(value[0, indices // n, indices % n].contiguous() for value in original)
        actual = tuple(value[0, indices // n, indices % n].contiguous() for value in restored)
        equal = [
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(selected, actual, strict=True)
        ]
        workspace = backing_bytes((*restored, indices, *selected, *actual))
        if not all(equal):
            raise RuntimeError("chosen packet changed protected original K/V bits")
        result.append(
            dict(
                layer=index,
                source_bits=packet.source.bits,
                source_axis=packet.source.axis,
                quant_group=packet.source.group,
                tensor_hashes=hashes,
                protected_rows=int(rows.numel()),
                protected_positions_sha256=tensor_hash(rows)["sha256"],
                protected_original_K_bitwise_equal=equal[0],
                protected_original_V_bitwise_equal=equal[1],
                explicit_extra_audit_workspace_bytes=workspace,
                allocator_bytes_before_audit=before,
                allocator_bytes_after_audit=int(torch.cuda.memory_allocated())
                if rows.is_cuda
                else None,
                allocator_peak_including_instrumentation=int(torch.cuda.max_memory_allocated())
                if rows.is_cuda
                else None,
                audit_scratch_KV_only_peak_certified=False,
                transport_patch_scalars=[int(p.indices.numel()) for p in packet.transport_patches],
                materialized_original_source_not_retained=True,
            )
        )
        del restored, indices, selected, actual
    return result


@contextmanager
def inspect_chosen_packets(arm, captures, records):
    import evicastkv_review_source_prefix as source

    original = source._formats_with_ceiling

    def build(*args, **kwargs):
        formats, proof = original(*args, **kwargs)
        if records:
            raise ValueError("expected one chosen packet transaction, not all ladder candidates")
        records.extend(inspect_packets(args[0], args[2], formats, arm, captures))
        return formats, proof

    with patch.object(source, "_formats_with_ceiling", build):
        yield


def prepare(root, parent):
    from evicastkv_review_source_prefix import MODEL, label_blind_payload

    root, parent = Path(root), Path(parent)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new immutable scratch diagnostic root required")
    revision = clean_source(SOURCE)
    plan = read(parent / "plan.json")
    old_result = read(parent / "workers/1/result.json")
    if old_result["status"] != "completed" or old_result["failures"]:
        raise ValueError("completed consumed MM-Vet worker required")
    job = plan["jobs"][1]
    payload = label_blind_payload(job["inputs"][0])
    if payload["task"] != "mmvet_prefixkv":
        raise ValueError("only already consumed MM-Vet input is registered")
    bindings = {
        str(parent / "plan.json"): sha(parent / "plan.json"),
        str(parent / "workers/1/result.json"): sha(parent / "workers/1/result.json"),
    }
    for filename, expected in old_result["outputs"].items():
        path = parent / "workers/1" / filename
        if sha(path) != expected:
            raise ValueError("parent raw evidence changed")
        bindings[str(path)] = expected
    for path in (
        Path(job["inputs"][0]),
        *(Path(m["path"]) for m in payload["media"]),
        Path(__file__),
        SOURCE / "mosaickv/scripts/evicastkv_review_source_prefix.py",
        SOURCE / "mosaickv/scripts/evicastkv_review_source_controls.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/shared_source_format.py",
        SOURCE / "mosaickv/env/common/requirements.lock",
    ):
        bindings[str(path)] = sha(path)
    config = json.loads(json.dumps(job.get("config", plan["config"])))
    config["execution"]["seed"] = (
        13  # Same actual producer seed, now explicit in resolved provenance.
    )
    write_once(root / "resolved_config.json", config)
    bindings[str(root / "resolved_config.json")] = sha(root / "resolved_config.json")
    write_once(
        root / "protocol.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            parent=str(parent),
            producer_sha=plan["source_sha"],
            model=MODEL,
            job=job,
            config=config,
            bindings=bindings,
            arms=ARMS,
            policy=POLICY,
            workers=1,
            maximum_A100s_aggregate=6,
            cpus_per_worker=12,
            wall_minutes=120,
            expected_rows=4,
            fresh_confirmation_authorized=False,
            no_new_mask_policy=True,
            no_quality_claim=True,
            no_production_change=True,
        ),
    )
    return dict(workers=1, expected_rows=4, protocol_sha256=sha(root / "protocol.json"))


def validate(root):
    plan = read(Path(root) / "protocol.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy"] != json.loads(json.dumps(POLICY)) or tuple(plan["arms"]) != ARMS:
        raise ValueError("bounded protection diagnostic policy changed")
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"immutable diagnostic binding changed: {path}")
    return plan


def worker(root, index=0):
    import torch
    from evicastkv_campaign import environment
    from evicastkv_review_source_prefix import (
        MODEL,
        comparisons,
        forced_prefix,
        label_blind_payload,
        native_reference,
        request,
    )

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.hf_cli import _model_kwargs

    root = Path(root)
    plan = validate(root)
    if index != 0:
        raise ValueError("one consumed worker only")
    output = root / "worker"
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"])
    captured, packets, saved, rows = Captures(), {}, {}, []
    try:
        gpu = require_isolated_slurm_gpu(torch)
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(13)
        config = RunConfig.from_mapping(plan["config"])
        if (
            config.model.id,
            config.model.revision,
        ) != MODEL or config.model.precision.value != "bf16":
            raise ValueError("same pinned BF16 LLaVA checkpoint required")
        payload = label_blind_payload(plan["job"]["inputs"][0])
        parent_logits = torch.load(
            Path(plan["parent"]) / "workers/1/logits.pt", weights_only=True, map_location="cpu"
        )
        reference = parent_logits[ARMS[0]]
        prefix = forced_prefix(reference)
        registered_prefix = read(Path(plan["parent"]) / "workers/1/reference_prefix.json")
        if (
            list(prefix) != registered_prefix["tokens"]
            or digest(prefix) != registered_prefix["sha256"]
        ):
            raise ValueError("saved conditioning prefix disagrees with saved FullKV logits")
        write_once(
            output / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                config_sha256=config_sha256(config),
                protocol_sha256=sha(root / "protocol.json"),
                policy_sha256=digest(POLICY),
                model=dataclasses.asdict(config.model),
                dataset=dataclasses.asdict(config.dataset),
                processor_revision=MODEL[1],
                environment=environment(),
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                gpu=gpu,
                cuda_runtime=torch.version.cuda,
                vllm="not_used",
                sglang="not_used",
                backend="huggingface",
                attention="eager",
                seed=13,
                reference_prefix_sha256=digest(prefix),
                sample_id=payload["sample_id"],
                reference_artifact_sha256=sha(Path(plan["parent"]) / "workers/1/logits.pt"),
                measurement_type="consumed_protection_packet_activation_diagnostic",
                result_category="unified_comparison",
                result_origin="locally_measured",
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                benchmark_labels_read=False,
            ),
        )
        adapter = load_hf_adapter(
            MODEL[0],
            revision=MODEL[1],
            model_kwargs=_model_kwargs(config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        with torch.inference_mode(), (output / "raw.jsonl").open("x") as stream:
            for arm in ARMS:
                row = dict(
                    arm=arm,
                    status="failed",
                    sample_id=payload["sample_id"],
                    manifest_path=str(output / "manifest.json"),
                    config_sha256=config_sha256(config),
                    resolved_execution_sha256=digest(
                        dict(config_sha256=config_sha256(config), policy=POLICY, arm=arm)
                    ),
                )
                try:
                    with capture_operators(adapter, arm, captured):
                        if arm == ARMS[0]:
                            logits, record = native_reference(adapter, payload)
                        else:
                            packet_records = []
                            with inspect_chosen_packets(arm, captured, packet_records):
                                logits, record = request(
                                    adapter,
                                    payload,
                                    config,
                                    arm,
                                    prefix=prefix,
                                    cap=plan["job"]["initial_ceiling_bytes"],
                                )
                            packets[arm] = packet_records
                    numerical = comparisons(reference, logits)
                    if arm == ARMS[0] and (
                        numerical["prefix_128"]["token_agreement"] != 1.0
                        or numerical["prefix_128"]["maximum_logit_difference"] > 0.001
                        or float((reference[:16].double() - logits[:16].double()).abs().max())
                        > 0.001
                    ):
                        raise ValueError("repeated native FullKV differs from saved reference")
                    saved[arm] = logits
                    row.update(record, diagnostics=numerical, status="completed")
                except Exception as exception:
                    row["error"] = f"{type(exception).__name__}: {exception}"
                    result["failures"].append(dict(arm=arm, error=row["error"]))
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                if arm == ARMS[0] and row["status"] != "completed":
                    break
    except Exception as exception:
        result["failures"].append(dict(error=f"{type(exception).__name__}: {exception}"))
    for filename, value in (("captures.pt", captured.values), ("logits.pt", saved)):
        with (output / filename).open("xb") as destination:
            torch.save(value, destination)
    write_once(output / "packets.json", packets)
    result.update(
        rows=len(rows),
        captured_CPU_bytes=captured.bytes,
        maximum_capture_bytes=MAX_CAPTURE_BYTES,
        status="completed" if len(rows) == 4 and not result["failures"] else "failed",
    )
    result["outputs"] = {path.name: sha(path) for path in output.iterdir() if path.is_file()}
    write_once(output / "result.json", result)
    return result


def analyze(root):
    import torch
    from evicastkv_review_source_prefix import comparisons, verify_recomputed_diagnostics

    root = Path(root)
    plan = validate(root)
    result = read(root / "worker/result.json")
    if result["source_sha"] != plan["source_sha"]:
        raise ValueError("worker source differs from registered source")
    for name, expected in result["outputs"].items():
        if sha(root / "worker" / name) != expected:
            raise ValueError("diagnostic worker artifact changed")
    rows = [json.loads(line) for line in (root / "worker/raw.jsonl").read_text().splitlines()]
    failures = list(result["failures"])
    if tuple(row["arm"] for row in rows) != ARMS or result["status"] != "completed":
        failures.append(dict(error="all four diagnostic arms must complete"))
    logits = torch.load(root / "worker/logits.pt", weights_only=True, map_location="cpu")
    captures = torch.load(root / "worker/captures.pt", weights_only=True, map_location="cpu")
    packets = read(root / "worker/packets.json")
    reference = torch.load(
        Path(plan["parent"]) / "workers/1/logits.pt", weights_only=True, map_location="cpu"
    )[ARMS[0]]
    verifications = []
    for row in rows:
        if row["status"] == "completed":
            verifications.append(
                dict(
                    arm=row["arm"],
                    **verify_recomputed_diagnostics(
                        comparisons(reference, logits[row["arm"]]), row["diagnostics"]
                    ),
                )
            )
    actual_capture_bytes = sum(t.numel() * t.element_size() for t in captures.values())
    if (
        actual_capture_bytes != result["captured_CPU_bytes"]
        or actual_capture_bytes > MAX_CAPTURE_BYTES
    ):
        raise ValueError("saved capture byte limit/accounting mismatch")
    operators = [
        f"layer{layer}.{kind}"
        for layer in range(32)
        for kind in ("input", "attention_output", "post_attention_residual", "output")
    ] + ["final_norm"]
    expected = {
        f"{arm}|{prediction}|{operator}"
        for arm in ARMS
        for prediction in PREDICTIONS
        for operator in operators
    }
    expected.update(f"{arm}|packet_rows|layer{layer}" for arm in ARMS[1:] for layer in range(32))
    if set(captures) != expected:
        failures.append(
            dict(
                error="incomplete or unexpected fixed operator/packet capture grid",
                missing=sorted(expected - set(captures)),
                unexpected=sorted(set(captures) - expected),
            )
        )
    layer_records, invariants = [], []
    if all(arm in packets for arm in ARMS[1:]):
        for index in range(len(packets[ARMS[1]])):
            entries = [packets[arm][index] for arm in ARMS[1:]]
            invariant = {
                name: len({entry["tensor_hashes"][name]["sha256"] for entry in entries}) == 1
                for name in ("codes", "minimum", "scale", "cos", "sin", "union", "slots")
            }
            invariants.append(
                dict(
                    layer=index,
                    equal_base_source_and_phase=all(invariant.values()),
                    tensor_invariants=invariant,
                    protected_KV_equal=all(
                        entry["protected_original_K_bitwise_equal"]
                        and entry["protected_original_V_bitwise_equal"]
                        for entry in entries
                    ),
                )
            )
    contrasts = [(ARMS[0], arm) for arm in ARMS[1:]] + [(ARMS[1], arm) for arm in ARMS[2:]]
    for reference_arm, arm in contrasts:
        for prediction in PREDICTIONS:
            for operator in operators:
                left_key, right_key = (
                    f"{reference_arm}|{prediction}|{operator}",
                    f"{arm}|{prediction}|{operator}",
                )
                if right_key not in captures or left_key not in captures:
                    continue
                left, right = captures[left_key].double(), captures[right_key].double()
                layer_records.append(
                    dict(
                        reference_arm=reference_arm,
                        arm=arm,
                        prediction_zero_based=prediction,
                        operator=operator,
                        bitwise_equal=torch.equal(
                            captures[left_key].view(torch.int16),
                            captures[right_key].view(torch.int16),
                        ),
                        maximum_absolute_difference=float((left - right).abs().max()),
                        L2_difference=float((left - right).norm()),
                        causal_attribution=False,
                    )
                )
    first_differences = []
    for reference_arm, arm in contrasts:
        for prediction in PREDICTIONS:
            differing = [
                row
                for row in layer_records
                if row["reference_arm"] == reference_arm
                and row["arm"] == arm
                and row["prediction_zero_based"] == prediction
                and not row["bitwise_equal"]
            ]
            first_differences.append(
                dict(
                    reference_arm=reference_arm,
                    arm=arm,
                    prediction_zero_based=prediction,
                    first_observed_difference=differing[0] if differing else None,
                    causal_attribution=False,
                )
            )
    tie = []
    for arm, tensor in logits.items():
        values = tensor[22].double()
        tie.append(
            dict(
                arm=arm,
                prediction_zero_based=22,
                prediction_one_based=23,
                token_10580_logit=float(values[10580]),
                token_12440_logit=float(values[12440]),
                signed_10580_minus_12440=float(values[10580] - values[12440]),
                greedy_token=int(values.argmax()),
                not_benchmark_quality=True,
            )
        )
    write_once(root / "analysis/operators.json", layer_records)
    write_once(root / "analysis/first_observed_differences.json", first_differences)
    write_once(root / "analysis/source_invariants.json", invariants)
    write_once(root / "analysis/tied_readout.json", tie)
    gate = dict(
        status="passed_capture" if not failures and len(rows) == 4 else "failed_capture",
        source_sha=plan["source_sha"],
        rows=len(rows),
        failures=failures,
        captured_CPU_bytes=actual_capture_bytes,
        packet_layers=len(invariants),
        all_base_source_hashes_equal=bool(invariants)
        and all(v["equal_base_source_and_phase"] for v in invariants),
        all_protected_rows_bitwise_original=bool(invariants)
        and all(v["protected_KV_equal"] for v in invariants),
        numerical_verifications=verifications,
        unchanged_32_token_guard=POLICY["guard"],
        additional_packet_audit_workspace_reported=True,
        timings_not_comparative=True,
        operator_differences_are_observational_not_causal=True,
        no_online_correction=True,
        fresh_confirmation_authorized=False,
        superiority_established=False,
    )
    write_once(root / "analysis/GATE_source_protection.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    result = (
        prepare(args.root, args.parent)
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
