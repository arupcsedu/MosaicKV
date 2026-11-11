#!/usr/bin/env python3
"""Consumed-input layer-local source retirement; no new quality/latency claim.

Reuse the passed lossless lane's bounded operator/model/worker lifecycle, native
128-token reference and independent numerical analyzer. Only request construction
changes: original attention seals one exact source packet before the next layer,
without a global SourceCapture or QueryCapture. Raw K/V copied to CPU for an
independent bitwise control are never used by the constructor or decode reader.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_lossless_source_control as parent

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
DEFAULT_LOSSLESS_GATE_ROOT = Path("/scratch/djy8hg/workdir/fox-review-v12/run-20261111-1/lossless")
PARENT_SOURCE = "76d4ced1a10c6746bad73f564e84210522355ef9"
LABEL = "evicastkv_streaming_original_BF16_source_lossless_control"
GATE = "GATE_streaming_lossless_source.json"
POLICY = dict(
    parent.POLICY,
    label=LABEL,
    parent_lossless_source_sha=PARENT_SOURCE,
    construction="per_layer_SealAfterAttention_RetiringPrefillCache",
    global_source_and_query_capture=False,
    positions="all_original_prompt_positions_no_selection_change",
    prefill_control="independent_native_KV_and_last_position_logits_bitwise",
    CPU_original_KV_control="outside_stream_prefill_never_a_decode_source_or_residual",
    measured_prefill_includes_all_layer_encoders_and_tensor_audits=True,
    no_whole_request_peak_claim=True,
)
_CUDA_OPERATOR = parent.cuda_operator_control


def parent_gate(root):
    """Authenticate the completed V12 prerequisite without executing it again."""
    root = Path(root)
    gate = parent.read(root / "GATE_lossless_source_control.json")
    protocol = parent.read(root / "protocol.json")
    result = parent.read(root / "worker/result.json")
    required = (
        "all_original_prompt_KV_bitwise",
        "all_unchanged_32_token_guards_pass",
        "all_128_source_tokens_agree",
        "both_initial_caches_smaller_than_raw",
    )
    outputs = result.get("outputs", {})
    if (
        gate.get("status") != "completed_lossless_source_diagnostic"
        or gate.get("completed") != 2
        or gate.get("failures") != []
        or any(gate.get(key) is not True for key in required)
        or gate.get("source_sha") != PARENT_SOURCE
        or protocol.get("source_sha") != PARENT_SOURCE
        or result.get("source_sha") != PARENT_SOURCE
        or result.get("status") != "completed"
        or result.get("completed_samples") != 2
        or result.get("failures") != []
        or not {"raw.jsonl", "logits.pt", "manifest.json", "GATE_cuda_operator.json"}
        <= set(outputs)
    ):
        raise ValueError("Passed immutable V12 lossless bitwise/32/128/storage gate required")
    bindings = {
        str(root / name): sha(root / name)
        for name in ("GATE_lossless_source_control.json", "protocol.json", "worker/result.json")
    }
    for name, expected in outputs.items():
        if Path(name).name != name or sha(root / "worker" / name) != expected:
            raise ValueError("V12 lossless prerequisite output changed")
        bindings[str(root / "worker" / name)] = expected
    return protocol, bindings


def prepare(root, *, cpu_gate_root=None, lossless_gate_root=DEFAULT_LOSSLESS_GATE_ROOT):
    root = Path(root)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("New immutable scratch streaming-lossless root required")
    previous, bindings = parent_gate(lossless_gate_root)
    if cpu_gate_root is None:
        raise ValueError("Current immutable complete CPU gate required")
    # This metadata-only registration reuses the already audited consumed-input,
    # media, configuration and CPU-gate bindings. It is never executed as a lane.
    parent.prepare(root / "consumed-input-registration", cpu_gate_root=cpu_gate_root)
    registration = parent.read(root / "consumed-input-registration/protocol.json")
    if [job["input"] for job in registration["jobs"]] != [job["input"] for job in previous["jobs"]]:
        raise ValueError("Only the same two prerequisite consumed inputs are permitted")
    bindings.update(registration["bindings"])
    bindings[str(root / "consumed-input-registration/protocol.json")] = sha(
        root / "consumed-input-registration/protocol.json"
    )
    for path in (
        Path(__file__),
        SOURCE / "mosaickv/src/mosaickv/evaluation/streaming_lossless_source.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/streaming_prefill.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/storage_runs.py",
    ):
        bindings[str(path)] = sha(path)
    plan = dict(registration, policy=POLICY, bindings=bindings)
    plan.update(
        source_root=str(SOURCE),
        source_sha=clean_source(SOURCE),
        prerequisite_lossless_root=str(Path(lossless_gate_root).resolve()),
        registration_is_metadata_only=True,
        measurement_type="consumed_streaming_original_BF16_source_fidelity_control",
    )
    write_once(root / "protocol.json", plan)
    return dict(workers=1, expected_rows=2, protocol_sha256=sha(root / "protocol.json"))


def validate(root):
    plan = parent.read(Path(root) / "protocol.json")
    clean_source(SOURCE, plan["source_sha"])
    if digest(plan["policy"]) != digest(POLICY) or any(
        sha(Path(path)) != expected for path, expected in plan["bindings"].items()
    ):
        raise ValueError("Immutable streaming lossless source/control bindings changed")
    parent_gate(plan["prerequisite_lossless_root"])
    return plan


def native_prefill_control(adapter, payload):
    """Unmodified native prefill, strictly outside streamed request measurement."""
    import evicastkv_request_validation as runtime

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    prepared = adapter.prepare_inputs(runtime.load_messages(payload))
    result = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    pairs, _ = adapter._legacy_layers(result.state.past_key_values)
    if not pairs or len(pairs) != len(adapter._language_layers()):
        raise ValueError("Complete native prefill K/V control required")
    original_refs = []
    for pair in pairs:
        for tensor in pair:
            seen = set()
            while tensor is not None and id(tensor) not in seen:
                seen.add(id(tensor))
                original_refs.append(weakref.ref(tensor))
                tensor = tensor._base
    del pair, tensor
    raw_bytes = backing_bytes(tuple(t for pair in pairs for t in pair))
    cpu_pairs = tuple(tuple(t.detach().cpu().clone() for t in pair) for pair in pairs)
    cpu_logits = result.logits.detach().cpu().clone()
    length = result.state.logical_sequence_length
    result.state.past_key_values = None
    prepared.model_inputs.clear()
    del result, prepared, pairs
    gc.collect()
    if any(ref() is not None for ref in original_refs):
        raise ValueError("Native control original GPU K/V did not retire before streamed request")
    return dict(
        pairs=cpu_pairs,
        logits=cpu_logits,
        raw_bytes=raw_bytes,
        logical_length=length,
        retired=True,
        CPU_verification_KV_bytes=backing_bytes(tuple(t for pair in cpu_pairs for t in pair)),
    )


def verify_prefill(adapter, packets, prefill, control):
    """Independent every-word native check; no control tensor enters the reader."""
    if (
        len(packets) != len(control["pairs"])
        or prefill.state.logical_sequence_length != control["logical_length"]
        or not parent.bitwise(prefill.logits.detach().cpu(), control["logits"])
    ):
        raise ValueError("Streamed native prefill geometry/logits changed a bit")
    records = []
    for index, (packet, original) in enumerate(zip(packets, control["pairs"], strict=True)):
        restored = packet.materialize(adapter._language_layers()[index].self_attn)
        if not all(
            parent.bitwise(actual.detach().cpu(), expected)
            for actual, expected in zip(restored, original, strict=True)
        ):
            raise ValueError("Streamed packet differs from independent original K/V")
        records.append(dict(layer=index, all_independent_original_KV_words_bitwise=True))
        del restored
    return records


def request(adapter, payload, config, *, prefix, deadline=None, save_prefill=None):
    import evicastkv_request_validation as runtime
    import torch

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.storage_runs import run
    from mosaickv.evaluation.streaming_lossless_source import StreamingLosslessSource
    from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention

    del config  # frozen configuration is manifested; neither scoring nor selection occurs
    if len(prefix) != 128 or any(type(token) is not int or token < 0 for token in prefix):
        raise ValueError("Complete FullKV decode-only conditioning required")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Bounded streaming control deadline reached before native prefill")
    control = native_prefill_control(adapter, payload)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started, timings = time.monotonic(), {}

    def timed(name, action):
        torch.cuda.synchronize()
        begin = time.monotonic()
        value = action()
        torch.cuda.synchronize()
        timings[name] = time.monotonic() - begin
        return value

    prepared = timed(
        "media_and_processor", lambda: adapter.prepare_inputs(runtime.load_messages(payload))
    )
    if prepared.logical_sequence_length != control["logical_length"]:
        raise ValueError("Streamed and native controls have different expanded prompt lengths")
    cache = RetiringPrefillCache(len(adapter._language_layers()), audit=True)
    encoder = StreamingLosslessSource(adapter, prepared, cache)
    fresh = dataclasses.replace(
        prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache)
    )
    with SealAfterAttention(adapter, cache, encoder):
        prefill = timed(
            "prefill_including_layer_local_construction_and_audit",
            lambda fresh=fresh: adapter.prefill(
                fresh, capture_queries=False, capture_attentions=False
            ),
        )
    report, packets = encoder.finish(), cache.formats()
    prefill_logits = prefill.logits.detach().cpu().clone()
    from evicastkv_review_source_controls import _packet_hash

    packet_hash = _packet_hash(packets)
    independent = timed(
        "post_prefill_independent_verification_not_a_decode_source",
        lambda prefill=prefill, control=control: verify_prefill(adapter, packets, prefill, control),
    )
    if save_prefill is not None:
        save_prefill(dict(native=control["logits"], streaming=prefill_logits))
    if (
        report["prefill_storage"]["original_KV_tensor_objects_alive"]
        or report["prefill_storage"]["original_native_layers_remaining"]
        or report["original_raw_KV_bytes"] != control["raw_bytes"]
        or not 0 < report["persistent_bytes"] < control["raw_bytes"]
    ):
        raise ValueError("Streamed source retirement/raw storage control failed")
    refs = tuple(cache.original_refs)
    retained = dataclasses.replace(
        prefill,
        state=dataclasses.replace(prefill.state, past_key_values=None),
        attention_weights=(),
    )
    raw, cpu_control_bytes = control["raw_bytes"], control["CPU_verification_KV_bytes"]
    del control, independent, prefill_logits
    begin = time.monotonic()
    prepared.model_inputs.clear()
    fresh.model_inputs.clear()
    del fresh, prepared, prefill, cache, encoder
    gc.collect()
    if any(ref() is not None for ref in refs):
        raise ValueError("Retired streamed original K/V remains live before decoding")
    timings["retire_prefill_handle_and_CPU_verification_control"] = time.monotonic() - begin
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Bounded streaming control deadline reached before reader decode")
    logits, record = timed(
        "decode_including_reader_construction_and_retirement",
        lambda: run(
            adapter,
            retained,
            packets,
            tokens=128,
            forced=prefix,
            audit=True,
            source_history=True,
            source_history_constructor="direct",
            shared_workspace=True,
            workspace_graph=False,
            initial_physical_byte_cap=raw,
        ),
    )
    if _packet_hash(packets) != packet_hash:
        raise ValueError("Reader mutated immutable streamed source packet")
    record.update(
        method=LABEL,
        phase_wall_seconds=timings,
        streamed_source=report,
        independent_prefill_control=dict(
            every_layer_original_KV_bitwise=True,
            last_position_logits_bitwise=True,
            raw_GPU_control_retired_before_streamed_prefill=True,
            CPU_original_KV_verification_bytes_outside_decode_representation=cpu_control_bytes,
            CPU_control_retired_before_decode=True,
        ),
        raw_source_kv_bytes=raw,
        representation_storage_bytes=backing_bytes(tuple(t for p in packets for t in p.tensors())),
        representation_sha256=packet_hash,
        inactive_original_source_retired=True,
        global_source_or_query_capture=False,
        original_prompt_KV_bitwise=True,
        fixed_prefix_decode_only=True,
        source_fallback=False,
        actual_bytes_not_shared_budget=True,
        end_to_end_diagnostic_seconds=time.monotonic() - started,
        whole_request_allocator_peak_bytes=int(torch.cuda.max_memory_allocated()),
        whole_request_allocator_reserved_peak_bytes=int(torch.cuda.max_memory_reserved()),
        whole_request_KV_peak_certified=False,
        no_latency_or_quality_claim=True,
    )
    return logits, record


def cuda_operator_control():
    """Actual tiny CUDA packet plus actual HF streamed-prefill controls before weights."""
    import torch
    from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

    from mosaickv.evaluation.streaming_lossless_source import StreamingLosslessSource
    from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention

    operator = _CUDA_OPERATOR()
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=512,
        intermediate_size=768,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).bfloat16().eval().cuda()
    adapter = SimpleNamespace(_language_layers=lambda: model.model.layers)
    tokens = torch.arange(8, device="cuda")[None]
    original = model(tokens, use_cache=True, past_key_values=DynamicCache())
    cache = RetiringPrefillCache(2, audit=True)
    encoder = StreamingLosslessSource(adapter, SimpleNamespace(logical_sequence_length=8), cache)
    with SealAfterAttention(adapter, cache, encoder):
        streamed = model(tokens, use_cache=True, past_key_values=cache)
    report = encoder.finish()
    if not parent.bitwise(original.logits, streamed.logits):
        raise ValueError("Actual tiny CUDA streamed prefill logits changed a bit")
    for index, packet in enumerate(cache.formats()):
        pair = packet.materialize(model.model.layers[index].self_attn)
        reference = original.past_key_values.layers[index]
        if not all(
            parent.bitwise(a, b)
            for a, b in zip(pair, (reference.keys, reference.values), strict=True)
        ):
            raise ValueError("Actual tiny CUDA streamed original K/V changed a bit")
    if report["prefill_storage"]["original_KV_tensor_objects_alive"]:
        raise ValueError("Actual tiny CUDA streamed original cache did not retire")
    torch.cuda.synchronize()
    return dict(operator, actual_tiny_CUDA_HF_streamed_prefill=True, streaming=report)


def verify_row(row, job):
    storage, stream = row["storage"], row["streamed_source"]
    control = row["independent_prefill_control"]
    if (
        row["index"] != job["index"]
        or row["sample_id"] != job["sample_id"]
        or row["task"] != job["task"]
        or row["method"] != LABEL
        or row["used_fullkv_fallback"] is not False
        or row["source_fallback"] is not False
        or row["inactive_original_source_retired"] is not True
        or row["global_source_or_query_capture"] is not False
        or row["original_prompt_KV_bitwise"] is not True
        or row["output_tokens"] != 128
        or any(
            type(value) is not int or value < 1
            for value in (
                row["raw_source_kv_bytes"],
                storage["initial_active_kv_storage_bytes"],
                storage["final_active_kv_storage_bytes"],
                storage["peak_active_kv_storage_bytes"],
                stream["persistent_bytes"],
            )
        )
        or storage["history_updates_per_layer"] != [127] * 32
        or not 0 < storage["initial_active_kv_storage_bytes"] < row["raw_source_kv_bytes"]
        or not 0
        < storage["final_active_kv_storage_bytes"]
        <= storage["peak_active_kv_storage_bytes"]
        or storage["initial_active_kv_storage_bytes"] > storage["peak_active_kv_storage_bytes"]
        or stream["original_raw_KV_bytes"] != row["raw_source_kv_bytes"]
        or stream["persistent_bytes"] != row["representation_storage_bytes"]
        or stream["global_source_or_query_capture"] is not False
        or stream["original_KV_all_word_bitwise_validation"] is not True
        or stream["prefill_storage"]["original_KV_tensor_objects_alive"] != 0
        or stream["prefill_storage"]["original_native_layers_remaining"] != 0
        or stream["prefill_storage"]["logical_lengths"] != [row["original_logical_length"]] * 32
        or stream["whole_request_gpu_peak_certified"] is not False
        or any(
            control.get(name) is not True
            for name in (
                "every_layer_original_KV_bitwise",
                "last_position_logits_bitwise",
                "raw_GPU_control_retired_before_streamed_prefill",
                "CPU_control_retired_before_decode",
            )
        )
    ):
        raise ValueError("Complete streamed lossless retirement/prefill/byte controls required")
    records = stream["layer_records"]
    if sorted(record["layer"] for record in records) != list(range(32)) or any(
        not 0 < record["persistent_bytes"] < record["raw_kv_bytes"]
        or record["reference"] != "original_prefill_KV_not_lossy_parent_decoder"
        or record["all_prompt_positions_retained"] is not True
        for record in records
    ):
        raise ValueError("Every original prompt layer must retain exact feasible source storage")
    if (
        sum(record["persistent_bytes"] for record in records) != stream["persistent_bytes"]
        or sum(record["raw_kv_bytes"] for record in records) != row["raw_source_kv_bytes"]
    ):
        raise ValueError("Summed complete layer storage differs from physical source accounting")
    timings = row["phase_wall_seconds"]
    expected = {
        "media_and_processor",
        "prefill_including_layer_local_construction_and_audit",
        "post_prefill_independent_verification_not_a_decode_source",
        "retire_prefill_handle_and_CPU_verification_control",
        "decode_including_reader_construction_and_retirement",
    }
    if set(timings) != expected or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in timings.values()
    ):
        raise ValueError("Complete finite streaming diagnostic phase timings required")
    return True


def worker(root, index=0):
    import torch

    plan = validate(root)
    by_input = {(job["task"], job["sample_id"]): job["index"] for job in plan["jobs"]}

    def run_request(adapter, payload, config, *, prefix, deadline=None):
        sample = by_input[(payload["task"], payload["sample_id"])]
        return request(
            adapter,
            payload,
            config,
            prefix=prefix,
            deadline=deadline,
            save_prefill=lambda value: torch.save(
                value, Path(root) / "worker" / f"prefill_{sample}.pt"
            ),
        )

    with (
        patch.object(parent, "LABEL", LABEL),
        patch.object(parent, "POLICY", POLICY),
        patch.object(parent, "validate", validate),
        patch.object(parent, "verify_row", verify_row),
        patch.object(parent, "request", run_request),
        patch.object(parent, "cuda_operator_control", cuda_operator_control),
    ):
        return parent.worker(root, index=index)


def analyze(root):
    import torch

    with (
        patch.object(parent, "LABEL", LABEL),
        patch.object(parent, "POLICY", POLICY),
        patch.object(parent, "validate", validate),
        patch.object(parent, "verify_row", verify_row),
    ):
        numerical = parent.analyze(root)
    passed = numerical["status"] == "completed_lossless_source_diagnostic" and all(
        numerical.get(key) is True
        for key in (
            "all_original_prompt_KV_bitwise",
            "all_unchanged_32_token_guards_pass",
            "all_128_source_tokens_agree",
            "both_initial_caches_smaller_than_raw",
        )
    )
    rows = [json.loads(line) for line in (Path(root) / "worker/raw.jsonl").read_text().splitlines()]
    failures = list(numerical["failures"])
    operator = parent.read(Path(root) / "worker/GATE_cuda_operator.json")
    if operator.get("actual_tiny_CUDA_HF_streamed_prefill") is not True:
        passed = False
        failures.append(dict(error="Actual tiny CUDA HF streaming precheck did not pass"))
    for row in rows:
        if row["status"] != "completed":
            continue
        saved = torch.load(
            Path(root) / "worker" / f"prefill_{row['index']}.pt",
            weights_only=True,
            map_location="cpu",
        )
        if (
            set(saved) != {"native", "streaming"}
            or any(
                value.dtype != torch.bfloat16
                or value.ndim != 2
                or value.shape[0] != 1
                or not torch.isfinite(value).all()
                for value in saved.values()
            )
            or not parent.bitwise(saved["native"], saved["streaming"])
        ):
            passed = False
            failures.append(dict(index=row["index"], error="Saved prefill logit bits differ"))
        if any(
            row["diagnostics"][name]["registered_guard_pass"] is not True
            for name in (
                "registered_guard_32",
                "cache_dependent_guard_32",
                "prefix_64",
                "prefix_128",
            )
        ):
            passed = False
            failures.append(dict(index=row["index"], error="Unchanged full-horizon guard failed"))
    gate = dict(
        status="completed_streaming_lossless_diagnostic"
        if passed
        else "blocked_streaming_lossless",
        passed=passed,
        source_sha=numerical["source_sha"],
        method=LABEL,
        rows=numerical["rows"],
        completed=numerical["completed"],
        failures=failures,
        independent_native_prefill_KV_and_logits_bitwise=passed,
        all_originals_retired_before_decode=passed,
        all_32_and_128_guards_pass=passed,
        inherited_numerical_gate_sha256=sha(Path(root) / "GATE_lossless_source_control.json"),
        independent_quality_N=0,
        fresh_confirmation_authorized=False,
        whole_request_KV_peak_certified=False,
        latency_superiority_established=False,
        superiority_established=False,
        validated_storage_variant_and_six_component_paths_unchanged=True,
    )
    write_once(Path(root) / GATE, gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--lossless-gate-root", type=Path, default=DEFAULT_LOSSLESS_GATE_ROOT)
    args = parser.parse_args()
    result = (
        prepare(
            args.root, cpu_gate_root=args.cpu_gate_root, lossless_gate_root=args.lossless_gate_root
        )
        if args.mode == "prepare"
        else globals()[args.mode](args.root)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
