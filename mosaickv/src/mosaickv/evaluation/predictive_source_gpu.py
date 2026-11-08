"""Scoped complete-source replacement; common lossless storage optimization."""

import weakref
from contextlib import contextmanager
from unittest.mock import patch

import torch

from mosaickv.evaluation import all_head_source_gpu as native
from mosaickv.evaluation import layer_code_delta as runtime


class State:
    def __init__(self, common, encoding_policy):
        self.common, self.records, self.references = common, [], []
        self.encoding_policy = encoding_policy
        self.encoding_audit = None

    def report(self):
        parent = self.common.report()
        return dict(
            parent,
            variant="common_all_consumer_compact_rare_modulo8_v1"
            if self.encoding_policy == "direct_additive_compact_rare"
            else "common_all_consumer_modulo8_source_replacement_v1"
            if self.encoding_policy == "search"
            else "common_all_consumer_direct_modulo8_raw_v1",
            encoding_policy=self.encoding_policy,
            records=self.records,
            common_codec_records=parent["records"],
            additive_replacements=sum(row["additive_selected"] for row in self.records),
            additive_original_owners_retired=sum(ref() is None for ref in self.references),
            saved_source_payload_bytes=sum(row["saved_bytes"] for row in self.records)
            if all(row["saved_bytes"] is not None for row in self.records)
            else None,
            additional_persistent_replay_bytes=0,
            numerical_precision_unchanged=True,
            original_six_mechanism_design_active=False,
            whole_request_peak_certified=False,
            source_encoding_overlap_audit=self.encoding_audit,
        )


@contextmanager
def use(enabled=True, *, decoder="bitmap_prefix", encoding_policy="search", audit_encoding=False):
    """One selector per layer; same existing decoder buffer for all 64 consumers."""
    if type(audit_encoding) is not bool:
        raise ValueError("explicit source encoding audit boolean required")
    if (
        type(enabled) is not bool
        or encoding_policy not in ("search", "direct_additive", "direct_additive_compact_rare")
        or (encoding_policy == "direct_additive_compact_rare" and decoder != "packed_words")
    ):
        raise ValueError("explicit predictive source activation required")
    with native.use(True, construction="pack_selected", decoder=decoder) as common:
        state = State(common, encoding_policy)
        original_convert = runtime.convert

        def audited_convert(packets):
            from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
            from mosaickv.evaluation.projection_cache import ScratchAudit

            original = tuple(t for packet in packets for t in packet.tensors() if t.is_cuda)
            peak = 0

            def observe(extra):
                nonlocal peak
                peak = max(peak, backing_bytes((*original, *(t for t in extra if t.is_cuda))))

            audit = ScratchAudit(excluded=set(), observe=observe)
            torch.cuda.synchronize()
            observe(())
            with audit:
                result = original_convert(packets)
                observe(tuple(t for packet in result[0] for t in packet.tensors() if t.is_cuda))
            torch.cuda.synchronize()
            state.encoding_audit = dict(
                peak_visible_GPU_tensor_bytes=peak,
                observed_ATen_operations=audit.operations,
                includes_original_packets_and_coexisting_new_payloads_and_ATen_results=True,
                original_owner_lifetime_not_extended_beyond_conversion=True,
                opaque_prefix_scan_workspace_observed=False,
                exact_whole_request_peak_certified=False,
            )
            return result

        def encoder(previous, current):
            if encoding_policy in ("direct_additive", "direct_additive_compact_rare"):
                from mosaickv.evaluation.lossless_code_delta_gpu import DeviceCodes

                candidate = (
                    native.encode(
                        previous,
                        current,
                        additive=True,
                        compact_rare=encoding_policy == "direct_additive_compact_rare",
                    )
                    if previous is not None
                    else None
                )
                candidate = (
                    DeviceCodes(current.numel(), current) if candidate is None else candidate
                )
                symbol_selected = isinstance(candidate, native.DeviceSymbols)
                # Deliberately not measured: direct encoding avoids the parent
                # search. Paired physical accounting supplies the actual saving.
                legacy_bytes, parent_bytes = None, None
            else:
                candidate, symbol_selected, legacy_bytes, parent_bytes = native.encode_best(
                    previous, current, additive_candidate=True
                )
            selected = getattr(candidate, "additive", False)
            packet = native.CodecChoice(
                candidate,
                torch.tensor(
                    [6 if selected else int(symbol_selected)],
                    dtype=torch.uint8,
                    device=current.device,
                ),
            )
            if selected:
                state.references.append(weakref.ref(current))
            common.records.append(
                dict(
                    source_code_bytes=current.numel(),
                    legacy_serialized_bytes=legacy_bytes,
                    selected_serialized_bytes=packet.serialized_bytes,
                    symbol_codec_selected=symbol_selected,
                )
            )
            if symbol_selected:
                common.references.append(weakref.ref(current))
            state.records.append(
                dict(
                    source_code_bytes=current.numel(),
                    parent_bytes=parent_bytes,
                    selected_bytes=packet.serialized_bytes,
                    additive_selected=selected,
                    saved_bytes=None
                    if parent_bytes is None
                    else parent_bytes - packet.serialized_bytes,
                )
            )
            return packet

        if not enabled:
            yield state
        else:
            from contextlib import nullcontext

            with (
                patch.object(runtime, "encode", encoder),
                (
                    patch.object(runtime, "convert", audited_convert)
                    if audit_encoding
                    else nullcontext()
                ),
            ):
                yield state
