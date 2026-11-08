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

    def report(self):
        parent = self.common.report()
        return dict(
            parent,
            variant="common_all_consumer_modulo8_source_replacement_v1"
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
        )


@contextmanager
def use(enabled=True, *, decoder="bitmap_prefix", encoding_policy="search"):
    """One selector per layer; same existing decoder buffer for all 64 consumers."""
    if type(enabled) is not bool or encoding_policy not in ("search", "direct_additive"):
        raise ValueError("explicit predictive source activation required")
    with native.use(True, construction="pack_selected", decoder=decoder) as common:
        state = State(common, encoding_policy)

        def encoder(previous, current):
            if encoding_policy == "direct_additive":
                from mosaickv.evaluation.lossless_code_delta_gpu import DeviceCodes

                candidate = (
                    native.encode(previous, current, additive=True)
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
            with patch.object(runtime, "encode", encoder):
                yield state
