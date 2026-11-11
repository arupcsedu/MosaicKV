"""CPU-only recovery of already executed COCO numerical controls.

128 *checks* across 32 inputs are not a 128-token continuation. The later
shape-matched boundary control ran 16 predictions on each of two consumed
inputs. Neither control validates BF16 source reconstruction on FP16/GQA,
repairs the historical native chunking failure, or authorizes confirmation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from mosaickv.evaluation.fox_campaign import digest, sha

NATIVE_GATE = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-v4-native-coco-20261109/analysis/correctness/GATE_correctness.json"
)
SHAPE_ROOT = Path("/scratch/djy8hg/audits/fox-review-v9-shape-normalization-20261110/norm")
NATIVE_GATE_SHA = "ca593e0522d3a9ee996fac0ce60a58b5802d35c4ff4404cdbb5799643b3439d9"
SHAPE_GATE_SHA = "4b453dd1dcf76515b600664075b15ec72f0a3d4fa2db2b176667646a7b16498c"
SHAPE_SOURCE = "554a04910b9838a4cee8bf71b6a8a21fc6544f26"
MODEL = dict(
    id="llava-hf/llava-v1.6-mistral-7b-hf",
    revision="2424fdd47412fccc66d91719126b420e9fbd7065",
    precision="fp16",
)
CONSUMED_IDS = (
    "coco_caption:COCO_val2014_000000260440.jpg",
    "coco_caption:COCO_val2014_000000512467.jpg",
)


def _read(path):
    return json.loads(Path(path).read_text())


def validate_native(record):
    """Validate executed count semantics, not a renamed token-level claim."""
    if (
        record.get("status") != "passed"
        or record.get("stage") != "correctness"
        or record.get("blockers") != []
    ):
        raise ValueError("complete passed native COCO correctness gate required")
    counts = record.get("tests_actually_executed", {})
    expected = dict(
        actual=128,
        completed=128,
        expected=128,
        pairing_count=64,
        workers_completed=4,
        workers_expected=4,
    )
    if any(
        type(counts.get(key)) is not int or counts[key] != value for key, value in expected.items()
    ):
        raise ValueError("native correctness gate check/worker counts changed")
    return dict(
        status="passed_native_packed_subset_controls",
        executed_checks=128,
        consumed_inputs=32,
        checks_per_input=4,
        scope="registered_native_subset_transport_and_retention_one_controls",
        continuation_128_predictions_established=False,
    )


def validate_shape(record, transport, inputs, config):
    """Fail closed on altered arithmetic, counts, scope or boundary evidence."""
    if (
        record.get("source_sha") != SHAPE_SOURCE
        or record.get("status") != "passed_new_shape_transport"
        or record.get("complete") is not True
        or record.get("failures") != []
        or record.get("unchanged_tolerance") != 1e-4
    ):
        raise ValueError("complete passed original-tolerance shape control required")
    for key, value in dict(
        historical_native_gate_still_negative=True,
        native_norm_boundary=True,
        ordinary_continuation_after_boundary=True,
        shared_transport_only=True,
        production_default_changed=False,
        quality_or_latency_claim_allowed=False,
        fresh_confirmation_authorized=False,
    ).items():
        if record.get(key) is not value:
            raise ValueError(f"numerical scope drift: {key}")
    if (
        transport.get("continuation_predictions") != 16
        or transport.get("tolerance") != 1e-4
        or transport.get("native_norm_boundary") is not True
        or transport.get("native_historical_gate_superseded") is not False
        or transport.get("no_output_or_logit_patch") is not True
        or transport.get("online_default") is not False
    ):
        raise ValueError("registered shape policy or historical-negative preservation changed")
    samples = inputs.get("samples", [])
    if tuple(sample.get("sample_id") for sample in samples) != CONSUMED_IDS:
        raise ValueError("only the two registered consumed COCO inputs are admissible")
    if config.get("model") != MODEL or config.get("method") != "full_kv":
        raise ValueError("pinned FP16/GQA Mistral FullKV config required")
    if (
        config.get("execution", {}).get("backend") != "huggingface"
        or config.get("execution", {}).get("attention_implementation") != "eager"
        or config.get("cache", {}).get("retention_ratio") != 1.0
    ):
        raise ValueError("native eager FullKV retention-one config required")
    reports = record.get("reports", [])
    if len(reports) != 2 or [report.get("index") for report in reports] != [0, 1]:
        raise ValueError("two complete ordered boundary cases required")
    for report in reports:
        continuation = report.get("continuation", {})
        left, right = (
            continuation.get(key) for key in ("reference_token_ids", "candidate_token_ids")
        )
        if (
            report.get("passed") is not True
            or continuation.get("predictions") != 16
            or not isinstance(left, list)
            or not isinstance(right, list)
            or len(left) != 16
            or len(right) != 16
            or any(type(token) is not int or token < 0 for token in left + right)
            or left != right
            or continuation.get("token_agreement") != 1.0
            or continuation.get("reference_prefix_forced_on_both_paths") is not True
            or continuation.get("ordinary_continuation_after_boundary") is not True
        ):
            raise ValueError("registered 16-prediction boundary continuation failed")
        differences = continuation.get("per_prediction_maximum_logit_difference", [])
        if (
            len(differences) != 16
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1e-4
                for v in differences
            )
            or continuation.get("maximum_logit_difference") != max(differences)
            or continuation.get("unchanged_absolute_tolerance") != 1e-4
        ):
            raise ValueError("original numerical threshold or logit evidence failed")
        for phase in ("initial_cache", "final_cache"):
            cache = continuation.get(phase, {})
            if (
                cache.get("bitwise_equal") is not True
                or cache.get("maximum_absolute_difference") != 0.0
            ):
                raise ValueError("initial/final cache bitwise parity failed")
        counts = continuation.get("operation_counts", {})
        expected = dict(
            observed_prefill_layers=32,
            attention_padded_calls=32,
            linear_padded_calls=224,
            normalization_expected_calls=65,
            normalization_padded_calls=65,
            normalization_input_calls=32,
            normalization_post_attention_calls=32,
            normalization_final_calls=1,
            normalization_phase_entries=1,
            normalization_phase_completions=1,
        )
        if any(
            type(counts.get(key)) is not int or counts[key] != value
            for key, value in expected.items()
        ):
            raise ValueError("shape/norm wrappers leaked or did not execute exactly once")
        first = report.get("first_logit", {})
        if (
            first.get("bitwise_equal") is not True
            or first.get("maximum_absolute_difference") != 0.0
        ):
            raise ValueError("saved original first-logit control failed")
    return dict(
        status="passed_evaluation_only_shape_matched_boundary_control",
        consumed_inputs=2,
        predictions_per_input=16,
        total_predictions=32,
        maximum_logit_difference=max(
            r["continuation"]["maximum_logit_difference"] for r in reports
        ),
        initial_and_final_cache_bitwise_equal=True,
        native_historical_gate_still_negative=True,
        ordinary_continuation_after_held_token=True,
        continuation_128_predictions_established=False,
        online_default_changed=False,
    )


def audit(native_gate=NATIVE_GATE, shape_root=SHAPE_ROOT):
    """Hash-bind actual historical artifacts; no GPU/model loading or new labels."""
    import torch

    native_gate, shape_root = Path(native_gate), Path(shape_root)
    if (
        sha(native_gate) != NATIVE_GATE_SHA
        or sha(shape_root / "GATE_shape_transport.json") != SHAPE_GATE_SHA
    ):
        raise ValueError("registered numerical evidence content changed")
    native = validate_native(_read(native_gate))
    gate, transport, inputs, config = (
        _read(shape_root / name)
        for name in (
            "GATE_shape_transport.json",
            "transport.json",
            "inputs.json",
            "resolved_config.json",
        )
    )
    shape = validate_shape(gate, transport, inputs, config)
    result, manifest, protocol = (
        _read(shape_root / name)
        for name in ("result.json", "analysis_manifest.json", "protocol.json")
    )
    for path, expected in (
        (shape_root / "result.json", manifest["result_sha256"]),
        (shape_root / "transport.json", manifest["transport_sha256"]),
        (shape_root / "GATE_shape_transport.json", manifest["gate_sha256"]),
        (shape_root / "protocol.json", transport["base_protocol_sha256"]),
        (shape_root / "inputs.json", protocol["inputs_sha256"]),
        (shape_root / "resolved_config.json", protocol["resolved_config_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError(f"sealed numerical artifact changed: {path}")
    if manifest["verdict_sha256"] != digest(gate) or result["source_sha"] != SHAPE_SOURCE:
        raise ValueError("worker/verdict provenance drift")
    bindings = {str(native_gate): sha(native_gate)}
    bindings.update({str(path): sha(path) for path in shape_root.iterdir() if path.is_file()})
    records = result.get("records", [])
    if len(records) != 2:
        raise ValueError("two hash-bound worker records required")
    for index, item in enumerate(records):
        if (
            item["status"] != "completed"
            or item["index"] != index
            or item["sample_id"] != CONSUMED_IDS[index]
            or item["input_sha256"] != digest(inputs["samples"][index])
        ):
            raise ValueError("worker input/status provenance changed")
        path = Path(item["capture_path"]).resolve(strict=True)
        path.relative_to(shape_root.resolve(strict=True))
        if sha(path) != item["capture_sha256"]:
            raise ValueError("saved raw numerical capture changed")
        capture = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.equal(capture["logits_A"], capture["logits_B"]):
            raise ValueError("raw saved first logits differ")
        if digest(capture["shape_transport_continuation"]) != digest(
            gate["reports"][index]["continuation"]
        ):
            raise ValueError("sealed continuation differs from raw capture")
        bindings[str(path)] = sha(path)
    return dict(
        status="passed_CPU_evidence_recovery_not_new_GPU_execution",
        native_subset_controls=native,
        shape_boundary_controls=shape,
        bindings=bindings,
        model=MODEL,
        original_native_boundary_failure_preserved=True,
        BF16_full_MHA_source_evidence_transfers_to_FP16_GQA=False,
        source_reconstruction_on_COCO_checkpoint_validated=False,
        whole_request_peak_transfers_between_models=False,
        new_GPU_inference=False,
        benchmark_labels_read=False,
        fresh_confirmation_authorized=False,
        superiority_established=False,
        remaining=[
            "No 128-prediction corrected-boundary continuation has been executed.",
            "No complete 32-input corrected-boundary scoring lane has been executed.",
            "FP16/GQA source reconstruction needs its own model-specific gates if claimed.",
            "Historical native chunking failures remain negative; "
            "corrected transport is evaluation-only.",
        ],
    )
