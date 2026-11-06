"""Default-off same-value constructor transaction on consumed geometries only.

Reuse only already verified component values. No full reference tensor is
stored across components/layers. Native conversion audits the complete outer
packet, including cached text, against its original decoder.
"""

import time
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps

import torch

_scope = ContextVar("evicastkv_verified_constructor", default=None)


def current():
    return _scope.get()


def validate(mode):
    if type(mode) is not str or mode not in ("none", "same_buffer", "handoff"):
        raise ValueError("explicit none/same_buffer/handoff constructor required")


def correct_same_buffer(packet, attention, component, patch, actual, *, audit):
    from mosaickv.evaluation.exact_storage import SparseBitPatch
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    patch.apply(actual)
    if audit:
        empty = SparseBitPatch(
            torch.empty(0, device=actual.device, dtype=torch.int32),
            torch.empty(0, device=actual.device, dtype=torch.bfloat16),
        )
        original = replace(
            packet,
            transport_patches=tuple(patch if index == component else empty for index in range(2)),
        )
        verified = torch.empty_like(actual)
        fill_component(original, attention, component, verified)
        if not torch.equal(actual.view(torch.int16), verified.view(torch.int16)):
            raise RuntimeError("same-buffer correction differs from original second projection")
    return actual


class State:
    def __init__(self, mode, modes, *, audit, geometry=None):
        validate(mode)
        self.mode, self.modes, self.audit = mode, tuple(map(tuple, modes)), audit
        self.layers, self.engine, self.converted, self.closed = [], None, False, False
        self.components = self.word_checks = self.outer_stride_checks = 0
        self.workspace_bytes = self.maximum_text_bytes = 0
        self.geometry, self.geometry_replays = geometry, 0

    def begin(self, index, packet, text):
        if self.closed or index != len(self.layers) or index >= len(self.modes):
            raise RuntimeError("one ordered private constructor layer required")
        if packet.shape not in ((1, 32, 600, 128), (1, 32, 604, 128)):
            raise NotImplementedError("only the two consumed full-MHA source geometries supported")
        text = tuple(text)
        if (
            not text
            or text != tuple(sorted(set(text)))
            or any(not 0 <= p < packet.shape[2] for p in text)
        ):
            raise ValueError("ordered original text positions required")
        flat = None
        if self.mode == "handoff" and self.modes[index] != ("source", "source"):
            flat = torch.tensor(
                [h * packet.shape[2] + p for h in range(32) for p in text],
                device=packet.cos.device,
                dtype=torch.int32,
            )
        self.layers.append(dict(shape=packet.shape, text=text, flat=flat, bodies=[], repairs=[]))

    def capture(self, packet, attention, component, old, verified):
        from mosaickv.evaluation.exact_storage import SparseBitPatch
        from mosaickv.evaluation.full_prompt_source import restore_projected
        from mosaickv.evaluation.joint_incoming_projection import engine_factory
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

        self.components += 1
        if self.mode != "handoff":
            return
        row = self.layers[-1]
        if component != len(row["repairs"]) or row["shape"] != packet.shape:
            raise RuntimeError("exactly K then V for each private constructor layer required")
        empty = SparseBitPatch(
            torch.empty(0, device=verified.device, dtype=torch.int32),
            torch.empty(0, device=verified.device, dtype=torch.bfloat16),
        )
        template = replace(
            packet,
            protected_patches_disjoint=True,
            transport_patches=tuple(old if index == component else empty for index in range(2)),
        )
        # The original fit uses head-group one, then the frozen reader changes
        # to its registered geometry. Do not assume their floating-point words
        # coincide. Reuse the SAME component owner for that exact parent replay.
        from mosaickv.evaluation.sequential_kv_lease import fill_component
        from mosaickv.evaluation.source_row_geometry import geometry

        if self.geometry is not None:
            template = geometry(template, *self.geometry)
        fill_component(template, attention, component, verified)
        self.geometry_replays += 1
        if self.engine is None:
            self.engine = engine_factory(template)
            self.workspace_bytes = self.engine.workspace_bytes
        actual = torch.empty_like(verified)
        self.engine.project(template, attention, component, actual)
        restore_projected(template, component, actual)
        new = SparseBitPatch.build(verified, actual)
        new.apply(actual)
        if not torch.equal(verified.view(torch.int16), actual.view(torch.int16)):
            raise RuntimeError("handoff native correction failed original component words")
        ids, values = torch.cat((old.indices, new.indices)), torch.cat((old.values, new.values))
        sorted_ids = ids.sort().values
        if (
            not bool(((ids >= 0) & (ids < verified.numel())).all())
            or bool((sorted_ids[1:] == sorted_ids[:-1]).any())
            or bool(torch.isin(ids // 128, packet.row_positions).any())
        ):
            raise RuntimeError("handoff repairs conflict with protected evidence or each other")
        row["repairs"].append(SparseBitPatch(ids, values))
        if backing_bytes(tuple(t for p in row["repairs"] for t in p.tensors)) > 32768:
            raise RuntimeError("unchanged 32-KiB native correction cap exceeded")
        self.word_checks += 1
        if row["flat"] is not None:
            compact = (
                verified.reshape(1, 1, -1, 128).index_select(2, row["flat"].long()).contiguous()
            )
            if (
                compact.untyped_storage().data_ptr() == verified.untyped_storage().data_ptr()
                or compact.untyped_storage().nbytes() != compact.numel() * 2
            ):
                raise RuntimeError("text handoff must not retain a full reference owner")
            row["bodies"].append(compact)
            self.maximum_text_bytes = max(
                self.maximum_text_bytes, backing_bytes(tuple(row["bodies"]))
            )

    def text_packet(self, packet, positions, text, modes, report):
        from mosaickv.evaluation.cached_text_reader import CachedTextFormat
        from mosaickv.evaluation.exact_storage import encode
        from mosaickv.evaluation.lossless_bf16 import PackedBF16
        from mosaickv.evaluation.selective_text_construction import strip_exact

        row = self.layers[-1]
        if (
            tuple(text) != row["text"]
            or modes != self.modes[len(self.layers) - 1]
            or len(row["bodies"]) != 2
            or tuple(map(tuple, positions)) != (tuple(range(packet.shape[2])),) * packet.shape[1]
        ):
            raise RuntimeError("complete verified same-position cached text required")
        base = strip_exact(packet, frozenset(text))
        if packet.row_positions_cpu is None:
            base = replace(base, row_positions_cpu=None)
        bodies = tuple(
            encode(value) if mode == "encoded" else PackedBF16(tuple(value.shape), True, (value,))
            for value, mode in zip(row["bodies"], modes, strict=True)
        )
        row["bodies"].clear()
        if report is not None:
            report.update(
                borrowed_only_parent_exact=False,
                original_parent_materialized=False,
                parent_materialization_extent="reused_verified_component_text_rows",
            )
        return CachedTextFormat(base, row["flat"], bodies)

    def convert(self, packets, attentions, *, policy, audit):
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
        from mosaickv.evaluation.lt_source_format import LTSourceFormat, map_sources, use
        from mosaickv.evaluation.sequential_kv_lease import fill_component

        if (
            self.converted
            or policy != "direct"
            or audit is not self.audit
            or len(packets) != len(self.layers)
            or len(attentions) != len(packets)
            or self.components != 2 * len(packets)
        ):
            raise RuntimeError("complete ordered verified constructor conversion required")
        output, records = [], []
        for index, (packet, attention, row) in enumerate(
            zip(packets, attentions, self.layers, strict=True)
        ):
            if len(row["repairs"]) != 2 or row["bodies"]:
                raise RuntimeError("incomplete native/text handoff")

            def wrap(base, repairs=tuple(row["repairs"])):
                return LTSourceFormat(base, repairs)

            candidate = map_sources(packet, wrap)
            if audit:
                with use(self.engine):
                    for tail in (0, 1, 127):
                        for component in (0, 1):
                            shape = (*row["shape"][:2], row["shape"][2] + tail, 128)
                            reference = torch.full(
                                shape,
                                -71.0,
                                device=next(attention.parameters()).device,
                                dtype=torch.bfloat16,
                            )
                            actual = torch.full_like(reference, -71.0)
                            fill_component(packet, attention, component, reference)
                            fill_component(candidate, attention, component, actual)
                            if not torch.equal(
                                reference.view(torch.int16), actual.view(torch.int16)
                            ):
                                raise RuntimeError(
                                    "handoff differs from original final K/V or unused slots"
                                )
                            self.outer_stride_checks += 1
            records.append(
                dict(
                    layer=index,
                    added_patch_bytes=backing_bytes(
                        tuple(t for p in row["repairs"] for t in p.tensors)
                    ),
                )
            )
            output.append(candidate)
        total = sum(r["added_patch_bytes"] for r in records)
        if total > 2**20:
            raise RuntimeError("unchanged request correction cap exceeded")
        self.converted = True
        self.close()
        return tuple(output), dict(
            converted_layers=records,
            eligible_layers=len(records),
            added_patch_bytes=total,
            parent_KV_bitwise=True,
            original_headers_codes_weights_and_positions_unchanged=True,
            no_labels_or_answers_used=True,
            constructor_policy="direct",
            full_parent_KV_word_validation=True,
            exhaustive_stride_audit=audit,
            verified_component_handoff=True,
        )

    def close(self):
        if self.closed:
            return
        if self.engine is not None:
            self.engine.close()
            self.engine.source = self.engine.output = self.engine.incoming = None
            self.engine.extra_tensors = ()
            self.engine = None
        self.layers.clear()
        self.closed = True

    def record(self):
        return dict(
            mode=self.mode,
            components=self.components,
            native_word_checks=self.word_checks,
            final_outer_stride_checks=self.outer_stride_checks,
            native_workspace_bytes=self.workspace_bytes,
            maximum_compact_text_bytes=self.maximum_text_bytes,
            exact_parent_geometry_component_replays=self.geometry_replays,
            full_reference_tensors_retained_across_components=0,
            converted=self.converted,
            retired=self.closed,
            whole_request_GPU_peak_certified=False,
        )


def managed(function):
    @wraps(function)
    def call(*args, **kwargs):
        started = time.monotonic()
        mode = kwargs.get("verified_constructor", "none")
        validate(mode)
        if current() is not None:
            raise ValueError("private constructor scopes may not nest")
        active = mode != "none" and args[3] == "mosaickv_exact"
        if active and (
            kwargs.get("componentwise_fitting") is not True
            or kwargs.get("direct_text_policy") is None
            or kwargs.get("static_transport_patches", False)
            or kwargs.get("selective_text_rows", False)
            or kwargs.get("defer_exact_encoding", False)
        ):
            raise NotImplementedError("unchanged componentwise direct-text construction required")
        if (
            active
            and mode == "handoff"
            and (
                kwargs.get("overlap_native_fit", False)
                or kwargs.get("lt_source_reader") is not True
                or kwargs.get("lt_fit_policy") != "direct"
                or kwargs.get("source_tile_rows") != 192
                or kwargs.get("frozen_joint_plan") is not True
                or tuple(kwargs.get("construction_geometry") or ()) != (16, 4)
            )
        ):
            raise NotImplementedError(
                "handoff requires private frozen 192-row direct native fitting"
            )
        state = (
            State(
                mode,
                kwargs["direct_text_policy"]["modes"],
                audit=kwargs.get("audit_decode_storage", False),
                geometry=kwargs.get("construction_geometry"),
            )
            if active
            else None
        )
        token = _scope.set(state)
        try:
            values, packets, details = function(*args, **kwargs)
            if state is not None and state.mode == "handoff" and not state.converted:
                raise RuntimeError("native handoff was never consumed")
            if state is not None:
                state.close()
            details["verified_constructor"] = dict(
                requested=mode, eligible=active, report=None if state is None else state.record()
            )
            original_wall = details.get("total_wall_seconds")
            if original_wall is not None:
                outer_wall = time.monotonic() - started
                details["original_core_wall_seconds"] = original_wall
                details["verified_constructor_scope_overhead_seconds"] = outer_wall - original_wall
                details["total_wall_seconds"] = outer_wall
            return values, packets, details
        finally:
            if state is not None:
                state.close()
            _scope.reset(token)

    return call
