"""Fail-closed evidence boundaries for the shared-source storage variant.

A CUDA allocator envelope is a conservative upper bound, NOT an exact KV
measurement. It covers allocation overlap in all threads/streams of a request;
subtract only immutable, continuously resident model backing, never an arbitrary
pre-request allocator baseline. No historical counter is upgraded to a peak.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import patch

import torch

from mosaickv.evaluation.projection_cache import identity

MECHANISMS = (
    "future_query_forecasting",
    "cross_modal_evidence_graph",
    "value_aware_block_utility",
    "budgeted_submodular_selection",
    "exact_prototype_residual_tiers",
    "uncertainty_guided_residual_repair",
)


def positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def allocator_envelope(
    *,
    immutable_model_bytes: int,
    baseline_allocated_bytes: int,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    resident_model_unchanged: bool,
    complete_request: bool,
) -> dict[str, Any]:
    """Prove an upper bound without attributing activations to KV storage.

    The scope is PyTorch-owned CUDA KV backing. Vendor-internal workspaces,
    CPU/offloaded KV, allocator reservations and total process HBM are separate.
    All CUDA KV tensor storage in this implementation uses PyTorch allocations.
    """
    values = (
        immutable_model_bytes,
        baseline_allocated_bytes,
        peak_allocated_bytes,
        peak_reserved_bytes,
    )
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("nonnegative integer allocator measurements required")
    if not immutable_model_bytes <= baseline_allocated_bytes <= peak_allocated_bytes:
        raise ValueError("resident model/baseline/peak allocation ordering violated")
    if peak_reserved_bytes < peak_allocated_bytes:
        raise ValueError("reserved peak cannot be below allocated peak")
    if type(complete_request) is not bool or type(resident_model_unchanged) is not bool:
        raise ValueError("explicit completeness/residency evidence required")
    certified = complete_request and resident_model_unchanged
    return dict(
        schema_version=1,
        boundary="whole_request_PyTorch_owned_CUDA_KV_backing",
        scope_start="before_media_loading_and_processor",
        scope_end="after_detokenization_packet_retirement_and_cuda_synchronize",
        complete_request=complete_request,
        resident_model_unchanged=resident_model_unchanged,
        immutable_model_bytes=immutable_model_bytes,
        baseline_allocated_bytes=baseline_allocated_bytes,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        kv_peak_upper_bound_bytes=peak_allocated_bytes - immutable_model_bytes
        if certified
        else None,
        kv_peak_upper_bound_certified=certified,
        exact_kv_peak_bytes=None,
        exact_kv_peak_certified=False,
        includes_non_KV_request_allocations_in_upper_bound=True,
        arbitrary_baseline_subtraction_forbidden=True,
        covers_async_PyTorch_allocations=True,
        excludes="CPU_KV,vendor_internal_non_tensor_workspace,total_process_HBM",
        not_usable_as_equal_KV_peak_measurement=True,
    )


class WholeRequestEnvelope:
    """One request, one reset; no cache/weight owner is retained by the audit."""

    def __init__(self, model: Any, *, cuda: Any = None):
        self.model = model
        self.cuda = torch.cuda if cuda is None else cuda
        self.started = self.finished = False
        self.certificate = None
        self.interior_reset_attempts = 0

    def model_backing(self) -> dict[tuple[str, int, int], int]:
        owners = {}
        device = str(self.cuda.current_device())
        for tensor in (*self.model.parameters(), *self.model.buffers()):
            if tensor.device.type != "cuda" or str(tensor.device.index) != device:
                raise ValueError("single-device continuously CUDA-resident model required")
            key = identity(tensor)
            owners[key] = key[2]
        if not owners:
            raise ValueError("nonempty resident model required")
        return owners

    def __enter__(self):
        if self.started:
            raise RuntimeError("one nonreusable audit per request")
        self.started = True
        self.cuda.synchronize()
        self.owners = self.model_backing()
        self.baseline = self.cuda.memory_allocated()
        self.cuda.reset_peak_memory_stats()
        # Legacy inner runners reset before decode. Suppress those resets only
        # within this isolated request or they would erase construction peaks.
        self.reset_guard = patch.object(self.cuda, "reset_peak_memory_stats", self._reset)
        self.reset_guard.__enter__()
        return self

    def _reset(self, *args, **kwargs):
        self.interior_reset_attempts += 1

    def __exit__(self, kind, value, trace):
        try:
            self.cuda.synchronize()
            unchanged = self.owners == self.model_backing()
            self.finished = True
            self.certificate = allocator_envelope(
                immutable_model_bytes=sum(self.owners.values()),
                baseline_allocated_bytes=self.baseline,
                peak_allocated_bytes=self.cuda.max_memory_allocated(),
                peak_reserved_bytes=self.cuda.max_memory_reserved(),
                resident_model_unchanged=unchanged,
                complete_request=kind is None,
            )
            self.certificate["interior_reset_attempts_suppressed"] = self.interior_reset_attempts
        finally:
            self.reset_guard.__exit__(kind, value, trace)
        if not unchanged:
            raise RuntimeError("resident model changed during request; peak envelope invalid")
        return False


def certify_peak_separation(certificate, *, candidate_final_bytes, reference_final_bytes):
    """Certify a peak inequality, without guessing either exact peak.

    candidate_peak <= upper; reference_peak >= measured reference final KV.
    Thus candidate_peak/reference_peak <= upper/reference_final. This is not
    an equal-byte compressed-baseline comparison.
    """
    upper = certificate.get("kv_peak_upper_bound_bytes")
    if (
        certificate.get("kv_peak_upper_bound_certified") is not True
        or not positive_integer(upper)
        or not positive_integer(candidate_final_bytes)
        or not positive_integer(reference_final_bytes)
        or candidate_final_bytes > upper
    ):
        raise ValueError("complete upper certificate and consistent measured KV endpoints required")
    ratio = upper / reference_final_bytes
    return dict(
        candidate_peak_lower_bound_bytes=candidate_final_bytes,
        candidate_peak_upper_bound_bytes=upper,
        reference_peak_lower_bound_bytes=reference_final_bytes,
        peak_ratio_upper_bound=ratio,
        peak_reduction_fraction_lower_bound=1 - ratio,
        strictly_smaller_whole_request_CUDA_KV_peak_proven=ratio < 1,
        exact_peak_measured=False,
        byte_matched_comparison=False,
        proof="candidate_peak<=CUDA_allocator_envelope;reference_peak>=reference_final_KV",
    )


def comparison_admission(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_ids: Sequence[str],
    methods: Sequence[str],
    cap_bytes_by_sample: Mapping[str, int],
) -> dict[str, Any]:
    """Return inspectable blockers; never drop failures or pad a cache to pass.

    This is a feasibility pre-gate. Full admission ALSO requires independently
    hash-bound largest-feasible-capacity certificates via shared_byte_contract.
    An allocator upper bound cannot establish equal measured KV peaks.
    """
    if (
        not sample_ids
        or len(set(sample_ids)) != len(sample_ids)
        or len(set(methods)) != len(methods)
        or len(methods) < 2
        or "full_kv" in methods
        or set(cap_bytes_by_sample) != set(sample_ids)
        or not all(positive_integer(v) for v in cap_bytes_by_sample.values())
    ):
        raise ValueError(
            "explicit paired compressed methods, identities and positive caps required"
        )
    expected = {(sid, m) for sid in sample_ids for m in methods}
    keyed, blockers = {}, []
    for row in rows:
        key = (row["sample_id"], row["method"])
        if key not in expected or key in keyed:
            raise ValueError("duplicate or unexpected observation")
        keyed[key] = row
    for sid, method in sorted(expected - set(keyed)):
        blockers.append(dict(sample_id=sid, method=method, reason="missing_prediction"))
    for sid in sample_ids:
        observed = {}
        controls = set()
        for method in methods:
            row = keyed.get((sid, method))
            if row is None:
                continue
            if row.get("status") != "completed":
                blockers.append(dict(sample_id=sid, method=method, reason="failed_prediction"))
                continue
            fingerprint = row.get("comparison_controls_sha256")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                blockers.append(dict(sample_id=sid, method=method, reason="missing_controls"))
            else:
                controls.add(fingerprint)
            if row.get("used_fullkv_fallback") is not False:
                blockers.append(dict(sample_id=sid, method=method, reason="fallback_or_unknown"))
            certificate = row.get("whole_request_envelope") or {}
            if certificate.get("kv_peak_upper_bound_certified") is not True:
                blockers.append(
                    dict(sample_id=sid, method=method, reason="missing_request_envelope")
                )
            values = {}
            for name in ("initial_encoded_bytes", "final_encoded_bytes", "decode_owned_peak_bytes"):
                value = row.get(name)
                if not positive_integer(value):
                    blockers.append(dict(sample_id=sid, method=method, reason="missing_" + name))
                else:
                    values[name] = value
            if values.get("initial_encoded_bytes", math.inf) > cap_bytes_by_sample[sid]:
                blockers.append(dict(sample_id=sid, method=method, reason="initial_cap_exceeded"))
            if certificate.get("exact_kv_peak_certified") is not True:
                blockers.append(
                    dict(sample_id=sid, method=method, reason="whole_request_KV_peak_not_exact")
                )
            elif not positive_integer(certificate.get("exact_kv_peak_bytes")):
                blockers.append(
                    dict(sample_id=sid, method=method, reason="missing_exact_KV_peak_bytes")
                )
            else:
                values["whole_request_KV_peak_bytes"] = certificate["exact_kv_peak_bytes"]
            observed[method] = values
        if len(controls) > 1:
            blockers.append(dict(sample_id=sid, reason="controls_differ"))
        for name in (
            "initial_encoded_bytes",
            "final_encoded_bytes",
            "decode_owned_peak_bytes",
            "whole_request_KV_peak_bytes",
        ):
            actual = {m: v[name] for m, v in observed.items() if name in v}
            if len(set(actual.values())) > 1:
                blockers.append(dict(sample_id=sid, reason="unmatched_" + name, bytes=actual))
    return dict(
        complete=len(keyed) == len(expected) and all(r.get("status") == "completed" for r in rows),
        byte_matched_feasibility_pass=not blockers,
        paper_ready=False,
        capacity_certificates_still_required=True,
        fullkv_latency_competitor=False,
        blockers=blockers,
        original_six_component_claim_supported=False,
        candidate_label="EviCastKV_shared_source_precision_storage_variant",
        mechanism_activity={name: False for name in MECHANISMS},
    )
