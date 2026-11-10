#!/usr/bin/env python3
"""Two consumed FullKV cases: shape-matched boundary-B correctness experiment.

Historical native failures are preserved. This new execution transport changes
shapes, not weights, stored precision, tolerance or compressor decisions.
It is intentionally excluded from quality/timing and online-support claims.
"""

from __future__ import annotations

import gc
import json
import math
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch

import evicastkv_boundary_operators as boundary
from evicastkv_campaign import SOURCE

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.fox_shape_transport import (
    padded_prefix,
    remove_dummy_cache_slot,
    shape_matched_transport,
)


def prepare(root, parent, previous, *, native_norm_boundary=False, norm_evidence=None):
    if type(native_norm_boundary) is not bool:
        raise ValueError("explicit boundary-native-normalization variant required")
    norm_binding = {}
    if native_norm_boundary:
        if norm_evidence is None:
            raise ValueError("saved native RMSNorm replay evidence is required")
        norm_evidence = Path(norm_evidence)
        finding = json.loads(norm_evidence.read_text())
        if (
            finding.get("complete") is not True
            or finding.get("both_saved_shape_pairs_reproduced") is not True
            or finding.get("full_model_parity_established") is not False
            or finding.get("original_first_logit_tolerance_unchanged") != boundary.LOGIT_ATOL
        ):
            raise ValueError("complete unrelaxed native-norm replay must precede this variant")
        norm_binding = dict(path=str(norm_evidence), sha256=sha(norm_evidence))
    elif norm_evidence is not None:
        raise ValueError("norm evidence cannot silently activate the historical transport")
    if root.exists():
        raise FileExistsError("new shape-control root required")
    previous_gate = previous / "boundary/GATE_boundary_linear.json"
    gate = json.loads(previous_gate.read_text())
    if (
        gate["status"] != "passed_operator_diagnostic"
        or not gate["native_operator_replay_all_matched"]
    ):
        raise ValueError("prior native projection behavior must be localized first")
    boundary.prepare(parent, root)
    write_once(
        root / "transport.json",
        dict(
            source_sha=clean_source(SOURCE),
            base_protocol_sha256=sha(root / "protocol.json"),
            prior_gate=str(previous_gate),
            prior_gate_sha256=sha(previous_gate),
            transport="dummy-prefix-native-shapes-with-boundary-only-native-RMSNorm-v2"
            if native_norm_boundary
            else "dummy-prefix-plus-native-shape-matched-linear-and-attention",
            native_norm_boundary=native_norm_boundary,
            native_norm_evidence=norm_binding,
            ordinary_continuation_after_boundary=native_norm_boundary,
            no_output_or_logit_patch=True,
            implementation_sha256=sha(
                SOURCE / "mosaickv/src/mosaickv/evaluation/fox_shape_transport.py"
            ),
            continuation_predictions=16,
            tolerance=boundary.LOGIT_ATOL,
            same_dtype_and_weights=True,
            placeholder_is_not_held_token=True,
            native_historical_gate_superseded=False,
            headline_timing_eligible=False,
            online_default=False,
            fresh_confirmation_authorized=False,
        ),
    )
    return dict(workers=1, spec_sha256=sha(root / "transport.json"))


def cache_difference(adapter, a, b):
    import torch

    aa, bb = (adapter._legacy_layers(s.past_key_values)[0] for s in (a, b))
    if len(aa) != len(bb):
        raise ValueError("layer count changed")
    maximum, equal = 0.0, True
    for pa, pb in zip(aa, bb, strict=True):
        for left, right in zip(pa, pb, strict=True):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError("cache geometry/precision changed")
            equal = equal and torch.equal(left, right)
            maximum = max(maximum, float((left.float() - right.float()).abs().max()))
    return dict(bitwise_equal=equal, maximum_absolute_difference=maximum)


def run_paths(adapter, prepared, capture, *, native_norm_boundary=False):
    if type(native_norm_boundary) is not bool:
        raise ValueError("explicit boundary-native-normalization variant required")
    prefix, held = boundary.hold_back_last_text_token(prepared)
    n = prepared.logical_sequence_length
    if prefix.logical_sequence_length != n - 1:
        raise ValueError("held-token boundary changed")
    with capture, ExitStack() as shape_stack:
        # Historical default retains its original wrapper lifetime. The v2
        # variant restores ALL forwards before the fifteen continuation steps.
        kwargs = dict(native_norm_boundary=True) if native_norm_boundary else {}
        operations = shape_stack.enter_context(shape_matched_transport(adapter, n, **kwargs))
        with capture.phase("A"):
            a = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
        a_first = a.logits.detach().float().cpu().clone()
        dummy = padded_prefix(prefix)
        p = adapter.prefill(dummy, capture_queries=False, capture_attentions=False)
        b_state = remove_dummy_cache_slot(adapter, p.state, prefix)
        del p, dummy
        gc.collect()
        if b_state.active_cache_length != n - 1 or b_state.next_decode_position != n - 1:
            raise ValueError("dummy row remains physically or logically active")
        phase = operations.boundary_normalization() if native_norm_boundary else nullcontext()
        with capture.phase("B"), phase:
            b = adapter.decode_one_token(
                held, b_state, capture_queries=False, capture_attentions=False
            )
        b_first = b.logits.detach().float().cpu().clone()
        if any(
            value != n
            for value in (
                b.state.logical_sequence_length,
                b.state.active_cache_length,
                b.state.next_decode_position,
            )
        ):
            raise ValueError("held token duplicated or logical position changed")
        initial_cache = cache_difference(adapter, a.state, b.state)
        if native_norm_boundary:
            shape_stack.close()
        a_ids, b_ids = [int(a.next_token_id.item())], [int(b.next_token_id.item())]
        differences = [float((a.logits.float() - b.logits.float()).abs().max())]
        aa, bb, token = a.state, b.state, a.next_token_id
        for _ in range(15):
            # Keep the reference prefix identical even if a negative control
            # diverges; benchmark answers are never read or fed to selection.
            ra = adapter.decode_one_token(
                token, aa, capture_queries=False, capture_attentions=False
            )
            rb = adapter.decode_one_token(
                token, bb, capture_queries=False, capture_attentions=False
            )
            aa, bb, token = ra.state, rb.state, ra.next_token_id
            a_ids.append(int(ra.next_token_id.item()))
            b_ids.append(int(rb.next_token_id.item()))
            differences.append(float((ra.logits.float() - rb.logits.float()).abs().max()))
        final_cache = cache_difference(adapter, aa, bb)
        continuation = dict(
            predictions=16,
            reference_token_ids=a_ids,
            candidate_token_ids=b_ids,
            token_agreement=sum(x == y for x, y in zip(a_ids, b_ids, strict=True)) / 16,
            maximum_logit_difference=max(differences),
            per_prediction_maximum_logit_difference=differences,
            unchanged_absolute_tolerance=boundary.LOGIT_ATOL,
            reference_prefix_forced_on_both_paths=True,
            initial_cache=initial_cache,
            final_cache=final_cache,
            operation_counts=dict(operations),
            native_norm_boundary=native_norm_boundary,
            ordinary_continuation_after_boundary=native_norm_boundary,
        )
    return dict(
        captures=capture.records,
        orders=capture.orders,
        logits_A=a_first,
        logits_B=b_first,
        held_token_forward_count=1,
        original_logical_length=n,
        prefix_logical_length=n - 1,
        captured_bytes=capture.total_bytes,
        source_KV_modified=False,
        compressed_cache_used=False,
        shape_transport_continuation=continuation,
        new_backend_variant_not_native_history_repair=True,
    )


def validate_transport(root):
    transport = json.loads((root / "transport.json").read_text())
    clean_source(SOURCE, transport["source_sha"])
    if (
        sha(root / "protocol.json") != transport["base_protocol_sha256"]
        or sha(Path(transport["prior_gate"])) != transport["prior_gate_sha256"]
        or sha(SOURCE / "mosaickv/src/mosaickv/evaluation/fox_shape_transport.py")
        != transport["implementation_sha256"]
    ):
        raise ValueError("shape validation binding changed")
    if (
        transport["continuation_predictions"] != 16
        or transport["tolerance"] != boundary.LOGIT_ATOL
        or transport["native_historical_gate_superseded"] is not False
        or transport["online_default"] is not False
    ):
        raise ValueError("registered numerical scope or tolerance changed")
    if transport.get("native_norm_boundary", False):
        evidence = transport["native_norm_evidence"]
        if (
            sha(Path(evidence["path"])) != evidence["sha256"]
            or transport.get("ordinary_continuation_after_boundary") is not True
            or transport.get("no_output_or_logit_patch") is not True
        ):
            raise ValueError("boundary-native-normalization evidence or scope changed")
    return transport


def norm_boundary_scope(continuation):
    """Check execution scope separately from the scientific numerical verdict."""
    counts = continuation["operation_counts"]
    layers = counts.get("observed_prefill_layers", 0)
    if (
        continuation.get("native_norm_boundary") is not True
        or continuation.get("ordinary_continuation_after_boundary") is not True
        or type(layers) is not int
        or layers < 1
        or counts.get("normalization_phase_entries") != 1
        or counts.get("normalization_phase_completions") != 1
        or counts.get("normalization_expected_calls") != 2 * layers + 1
        or counts.get("normalization_padded_calls") != 2 * layers + 1
        or counts.get("normalization_input_calls") != layers
        or counts.get("normalization_post_attention_calls") != layers
        or counts.get("normalization_final_calls") != 1
        or counts.get("attention_padded_calls") != layers
        or counts.get("linear_padded_calls") != 7 * layers
    ):
        raise ValueError("native norm boundary incomplete or wrappers continued into decoding")
    preserved = True
    for name in ("initial_cache", "final_cache"):
        record = continuation[name]
        if (
            type(record["bitwise_equal"]) is not bool
            or not math.isfinite(record["maximum_absolute_difference"])
            or record["maximum_absolute_difference"] < 0
            or (record["bitwise_equal"] and record["maximum_absolute_difference"] != 0)
        ):
            raise ValueError("invalid cache parity record")
        preserved = preserved and record["bitwise_equal"]
    return preserved


def worker(root):
    transport = validate_transport(root)

    def paths(adapter, prepared, capture):
        return run_paths(
            adapter,
            prepared,
            capture,
            native_norm_boundary=transport.get("native_norm_boundary", False),
        )

    with patch.object(boundary, "run_paths", paths):
        boundary.worker(root)


def analyze(root):
    import torch

    transport = validate_transport(root)
    protocol = boundary._load_protocol(root)
    result = json.loads((root / "result.json").read_text())
    if (
        result["source_sha"] != protocol["source_sha"]
        or result["protocol_sha256"] != sha(root / "protocol.json")
        or result["environment_sha256"] != sha(root / "environment.json")
    ):
        raise ValueError("shape worker provenance changed")
    reports, failures = [], []
    environment = json.loads((root / "environment.json").read_text())
    if (
        environment["source_sha"] != protocol["source_sha"]
        or environment["config_sha256"] != protocol["config_sha256"]
    ):
        raise ValueError("environment source/config identity changed")
    expected = json.loads((root / "inputs.json").read_text())["samples"]
    records = result["records"]
    if (
        len(records) != 2
        or [r["index"] for r in records] != [0, 1]
        or [r["sample_id"] for r in records] != [s["sample_id"] for s in expected]
    ):
        raise ValueError("complete ordered two-case shape validation required")
    for item in records:
        if item["status"] != "completed":
            failures.append(item)
            continue
        if item["input_sha256"] != digest(expected[item["index"]]):
            raise ValueError("shape worker input changed")
        path = Path(item["capture_path"])
        path.resolve(strict=True).relative_to(root.resolve())
        if sha(path) != item["capture_sha256"]:
            raise ValueError("shape capture changed")
        capture = torch.load(path, map_location="cpu", weights_only=True)
        report = boundary.compare_capture(capture)
        continuation = capture["shape_transport_continuation"]
        aa, bb = continuation["reference_token_ids"], continuation["candidate_token_ids"]
        differences = continuation["per_prediction_maximum_logit_difference"]
        if (
            continuation["predictions"] != 16
            or len(aa) != 16
            or len(bb) != 16
            or len(differences) != 16
            or any(not math.isfinite(value) or value < 0 for value in differences)
            or differences[0] != float((capture["logits_A"] - capture["logits_B"]).abs().max())
            or any(type(token) is not int or token < 0 for token in aa + bb)
            or continuation["token_agreement"]
            != sum(a == b for a, b in zip(aa, bb, strict=True)) / 16
            or continuation["maximum_logit_difference"] != max(differences)
            or continuation["unchanged_absolute_tolerance"] != boundary.LOGIT_ATOL
            or capture["held_token_forward_count"] != 1
            or capture["original_logical_length"] != capture["prefix_logical_length"] + 1
            or capture["source_KV_modified"] is not False
            or capture["compressed_cache_used"] is not False
            or continuation["reference_prefix_forced_on_both_paths"] is not True
            or capture["new_backend_variant_not_native_history_repair"] is not True
        ):
            raise ValueError("shape continuation or unchanged FullKV contract invalid")
        passed = (
            report["fullkv_chunked_logit_control_passed"]
            and continuation["token_agreement"] == 1
            and continuation["maximum_logit_difference"] <= boundary.LOGIT_ATOL
            and continuation["operation_counts"]["attention_padded_calls"] > 0
        )
        if transport.get("native_norm_boundary", False):
            cache_passed = norm_boundary_scope(continuation)
            passed = passed and cache_passed
        reports.append(
            dict(
                index=item["index"],
                first_logit=report["first_logit_comparison"],
                continuation=continuation,
                passed=passed,
            )
        )
    complete = len(reports) == 2 and not failures
    gate = dict(
        status="passed_new_shape_transport"
        if complete and all(r["passed"] for r in reports)
        else "blocked_new_shape_transport",
        source_sha=protocol["source_sha"],
        complete=complete,
        reports=reports,
        failures=failures,
        unchanged_tolerance=boundary.LOGIT_ATOL,
        historical_native_gate_still_negative=True,
        shared_transport_only=True,
        native_norm_boundary=transport.get("native_norm_boundary", False),
        ordinary_continuation_after_boundary=transport.get("native_norm_boundary", False),
        production_default_changed=False,
        quality_or_latency_claim_allowed=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "GATE_shape_transport.json", gate)
    write_once(
        root / "analysis_manifest.json",
        dict(
            source_sha=protocol["source_sha"],
            result_sha256=sha(root / "result.json"),
            transport_sha256=sha(root / "transport.json"),
            gate_sha256=sha(root / "GATE_shape_transport.json"),
            evidence_type="new_evaluation_transport_not_native_gate_waiver",
            verdict_sha256=digest(gate),
        ),
    )
    return gate
