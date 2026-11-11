"""Published KV ownership across streaming prefill, construction and decoding.

This is a KV-only accounting boundary, NOT whole-request GPU memory. Model
weights and ordinary attention/projection activations are excluded explicitly.
Constructor validation tensors are audited too. Every source, metadata, reconstruction
pool, retained raw owner, history growth overlap and borrowed decode K/V is
included from publication/creation to retirement. Certification requires both
the replayable actual allocator ledger and the producer's lifecycle evidence.
"""

from __future__ import annotations

from collections import Counter

import torch

from mosaickv.evaluation.kv_allocator_ledger import AllocatorLedger, verify_ownership_deltas


class WholeRequestKVLedger:
    def __init__(
        self, *, device, layer_count, source, decode_updates=127, ledger_factory=AllocatorLedger
    ):
        if type(layer_count) is not int or layer_count < 1 or type(source) is not bool:
            raise ValueError("positive layer count and explicit source boolean required")
        if type(decode_updates) is not int or decode_updates < 1:
            raise ValueError("positive integer decode_updates required")
        self.layers, self.source = layer_count, source
        self.decode_updates = decode_updates
        self.ledger = ledger_factory(device=device)
        self.phases = []
        self.closed = False

    def observe(self, tensors, phase):
        if self.closed:
            raise RuntimeError("whole-request ownership ledger already closed")
        if not isinstance(phase, str) or not phase or len(phase) > 250:
            raise ValueError("bounded explicit lifecycle phase required")
        torch.cuda.synchronize(self.ledger.device)
        self.ledger.observe(tuple(tensors), phase=phase)
        self.phases.append(phase)

    def finish(self, evidence):
        if self.closed:
            raise RuntimeError("whole-request certification is write-once")
        if not isinstance(evidence, dict):
            raise ValueError("streaming producer lifecycle evidence required")
        counts = Counter(self.phases)
        required = [
            "request_workspace_created",
            "prefill_complete",
            "decode:initial_cache",
            "decode:final_cache",
        ]
        required += [
            f"prefill_layer_{i}_{action}"
            for i in range(self.layers)
            for action in ("KV_published", "packet_constructed", "native_retired")
        ]
        if self.source:
            required += [f"prefill_layer_{i}_source_captured" for i in range(self.layers)]
        decode_phases = [
            f"decode:decode_update_{step}_layer_{layer}_copied_history"
            for step in range(1, self.decode_updates + 1)
            for layer in range(self.layers)
        ]
        required += decode_phases
        for phase in required:
            if counts[phase] != 1:
                raise ValueError(f"missing/duplicate whole-request lifecycle phase: {phase}")
        if self.phases[0] != "request_workspace_created":
            raise ValueError("measurement must begin before prefill cache construction")
        previous = self.phases.index("request_workspace_created")
        for layer in range(self.layers):
            events = (["source_captured"] if self.source else []) + [
                "KV_published",
                "packet_constructed",
                "native_retired",
            ]
            for action in events:
                index = self.phases.index(f"prefill_layer_{layer}_{action}")
                if index <= previous:
                    raise ValueError("streaming layer publication/retirement order violated")
                previous = index
        complete = self.phases.index("prefill_complete")
        if complete <= previous or self.phases[-1] != "decode:final_cache":
            raise ValueError("complete prefill-to-final-decode coverage required")
        if self.phases.index("decode:initial_cache") <= complete:
            raise ValueError("decode cannot precede streaming prefill completion")
        previous = self.phases.index("decode:initial_cache")
        for phase in decode_phases:
            index = self.phases.index(phase)
            if index <= previous:
                raise ValueError(
                    "decode history coverage must follow registered update/layer order"
                )
            previous = index
        registered_decode_phases = set(decode_phases)
        if any(
            phase.startswith("decode:decode_update_")
            and phase.endswith("_copied_history")
            and phase not in registered_decode_phases
            for phase in self.phases
        ):
            raise ValueError("unregistered decode history update exceeds declared coverage")
        integer_evidence = dict(
            native_layers_retired=self.layers,
            maximum_live_native_cache_layers=1,
            outstanding_native_owner_references=0,
        )
        if any(
            type(evidence.get(key)) is not int or evidence[key] != value
            for key, value in integer_evidence.items()
        ):
            raise ValueError("strict integer native-retirement lifecycle evidence required")
        if (
            evidence.get("native_layers_retired") != self.layers
            or evidence.get("all_native_owner_weakrefs_released") is not True
            or evidence.get("uses_original_fullkv_snapshot") is not False
            or evidence.get("original_logical_positions_preserved") is not True
            or evidence.get("constructor_workspace_shared_with_decode") is not True
            or evidence.get("native_CUDA_free_completed_verified") is not True
            or evidence.get("constructor_temporaries_audited") is not True
        ):
            raise ValueError("native owner retirement or explicit lifecycle coverage incomplete")
        for key in ("benchmark_labels_read", "fallback_used"):
            if evidence.get(key) is not False:
                raise ValueError("labels or fallback invalidate whole-request certification")
        from mosaickv.evaluation.native_retirement_receipts import verify_receipts

        verify_receipts(evidence.get("native_retirement_receipts"), expected_layers=self.layers)
        if evidence.get("constructor_audit_layers") != list(range(self.layers)):
            raise ValueError("every layer constructor requires temporary-owner auditing")
        packet_reports = evidence.get("packets")
        if not isinstance(packet_reports, list) or len(packet_reports) != self.layers:
            raise ValueError("complete per-layer constructor reports required")
        for layer, report in enumerate(packet_reports):
            operations = report.get("constructor_audit_operations")
            publication = self.phases.index(f"prefill_layer_{layer}_KV_published")
            construction = self.phases.index(f"prefill_layer_{layer}_packet_constructed")
            constructor_events = [
                i
                for i, phase in enumerate(self.phases)
                if phase == f"prefill_layer_{layer}_constructor_workspace"
            ]
            if any(not publication < i < construction for i in constructor_events):
                raise ValueError("constructor ownership phase order violates layer transaction")
            if (
                type(operations) is not int
                or operations < 1
                or counts[f"prefill_layer_{layer}_constructor_workspace"] < operations
            ):
                raise ValueError("constructor operation/ownership coverage incomplete")
        summary = self.ledger.summary()
        if summary["all_required_owners_resolved"] is not True:
            raise ValueError("unresolved actual allocator owners")
        deltas = summary["ownership_deltas"]
        replay = verify_ownership_deltas(deltas, expected_summary=summary)
        if len(deltas) != len(self.phases) or any(
            delta["phase"] != phase for delta, phase in zip(deltas, self.phases, strict=True)
        ):
            raise ValueError("lifecycle phases differ from actual owner events")
        self.closed = True
        return dict(
            scope="published_KV_and_all_cache_required_storage_prefill_construction_decode",
            whole_request_KV_peak_certified=True,
            whole_request_GPU_peak_certified=False,
            accounting_device="CUDA_GPU_storage_only; CPU_metadata_reported_separately",
            initial_boundary="decode:initial_cache",
            final_boundary="decode:final_cache",
            decode_updates=self.decode_updates,
            initial_allocator_block_bytes=deltas[self.phases.index("decode:initial_cache")][
                "current_allocator_block_bytes"
            ],
            final_allocator_block_bytes=summary["final_allocator_block_bytes"],
            peak_allocator_block_bytes=summary["peak_allocator_block_bytes"],
            initial_tensor_backing_bytes=deltas[self.phases.index("decode:initial_cache")][
                "current_tensor_backing_bytes"
            ],
            final_tensor_backing_bytes=summary["final_tensor_backing_bytes"],
            peak_tensor_backing_bytes=summary["peak_tensor_backing_bytes"],
            includes="published_native_KV,source_captures,phases,retained_raw_KV,cache_metadata,constructor_and_reader_pools,constructor_ATen_temporaries,history,incoming_KV,growth_overlap",
            excludes="weights,ordinary_model_projection_and_attention_activations,opaque_vendor_workspace,allocator_reserved",
            every_ownership_observation_device_synchronized=True,
            evidence=evidence,
            allocator=summary,
            replay=replay,
            independent_quality_confirmation=False,
            paper_campaign_authorized=False,
        )


def replay_whole_request(record, deltas, *, layer_count, source, decode_updates=127):
    """Recheck lifecycle and physical bytes on CPU, not just a producer flag."""
    if not isinstance(record, dict) or not isinstance(record.get("allocator"), dict):
        raise ValueError("complete saved whole-request certificate required")
    if not isinstance(deltas, (list, tuple)) or not deltas:
        raise ValueError("complete actual ownership deltas required")

    class SavedLedger:
        def __init__(self, **kwargs):
            pass

        def summary(self):
            return dict(record["allocator"], ownership_deltas=deltas)

    validator = WholeRequestKVLedger(
        device=0,
        layer_count=layer_count,
        source=source,
        decode_updates=decode_updates,
        ledger_factory=SavedLedger,
    )
    validator.phases = [delta.get("phase") for delta in deltas]
    computed = validator.finish(record["evidence"])
    checked = (
        "scope",
        "whole_request_KV_peak_certified",
        "whole_request_GPU_peak_certified",
        "initial_boundary",
        "final_boundary",
        "decode_updates",
        "initial_allocator_block_bytes",
        "final_allocator_block_bytes",
        "peak_allocator_block_bytes",
        "initial_tensor_backing_bytes",
        "final_tensor_backing_bytes",
        "peak_tensor_backing_bytes",
    )
    if any(
        type(record.get(key)) is not type(computed[key]) or record[key] != computed[key]
        for key in checked
    ):
        raise ValueError("saved whole-request certificate differs from independent replay")
    return {key: computed[key] for key in checked}
