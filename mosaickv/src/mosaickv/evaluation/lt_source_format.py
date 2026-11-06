"""Default-off exact decoder substitution; never a new selection or repair policy."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.shared_source_format import SharedSourceFormat

_engine = ContextVar("evicastkv_private_lt_engine", default=None)
LAYER_PATCH_CAP = 32768


@contextmanager
def use(engine):
    if engine is None or _engine.get() is not None:
        raise RuntimeError("private nonnested native-source engine required")
    token = _engine.set(engine)
    try:
        yield
    finally:
        _engine.reset(token)


@dataclass(frozen=True)
class LTSourceFormat:
    base: object
    repairs: tuple
    kind: str = "lossless_bounded_FP32_cublasLt_execution_variant"

    def __post_init__(self):
        if (
            not isinstance(self.base, SharedSourceFormat)
            or self.base.projection_backend != "fused"
            or self.base.protected_patches_disjoint is not True
            or type(self.repairs) is not tuple
            or len(self.repairs) != 2
        ):
            raise ValueError(
                "original fused source, disjoint patches and two exact repairs required"
            )
        for patch in self.repairs:
            if (
                not isinstance(patch, SparseBitPatch)
                or patch.indices.dtype != torch.int32
                or patch.values.dtype != torch.bfloat16
                or patch.indices.ndim != 1
                or patch.values.ndim != 1
                or patch.indices.numel() != patch.values.numel()
                or patch.indices.device != self.base.cos.device
                or patch.values.device != self.base.cos.device
                or not patch.indices.is_contiguous()
                or not patch.values.is_contiguous()
            ):
                raise ValueError("same-device contiguous int32/BF16 numerical repairs required")
        if backing_bytes(tuple(t for p in self.repairs for t in p.tensors)) > LAYER_PATCH_CAP:
            raise ValueError("native-source numerical repair storage exceeds per-layer cap")

    @property
    def shape(self):
        return self.base.shape

    @property
    def exact(self):
        return self.base.exact

    @property
    def row_positions(self):
        return self.base.row_positions

    @property
    def transport_patches(self):
        return self.repairs

    @property
    def protected_patches_disjoint(self):
        return True

    def private(self):
        return (*self.base.private(), *(t for p in self.repairs for t in p.tensors))

    def shared(self):
        return self.base.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill_component(self, attention, component, destination):
        if _engine.get() is None:
            raise RuntimeError("bound private native-source workspace required")
        _engine.get().project(self.base, attention, component, destination)
        from mosaickv.evaluation.source_hotpath import restore_enabled

        if destination.device.type == "cuda" and restore_enabled():
            from mosaickv.evaluation.fused_source_restore import restore

            restore(self, component, destination)
        else:
            from mosaickv.evaluation.full_prompt_source import restore_projected

            restore_projected(self, component, destination)

    def fill(self, attention, targets, *, direct):
        if direct is not True or len(targets) != 2:
            raise ValueError("explicit direct two-component native-source lease required")
        for component, destination in enumerate(targets):
            self.fill_component(attention, component, destination)


def source_format(packet, component=None):
    from mosaickv.evaluation.head_local_precision import HeadLocalFormat
    from mosaickv.evaluation.hybrid_reader import HybridFormat
    from mosaickv.evaluation.layer_code_delta import BoundDeltaFormat, DeltaFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    while True:
        if isinstance(packet, LTSourceFormat):
            return packet
        if isinstance(packet, BoundDeltaFormat):
            packet = packet.specification
        elif isinstance(packet, DeltaFormat):
            packet = packet.template
        elif isinstance(packet, HybridFormat):
            if packet.base is None or (
                component is not None and packet.modes[component] != "source"
            ):
                return None
            packet = packet.base
        elif isinstance(packet, (HeadLocalFormat, TextSourceFormat)):
            packet = packet.base
        else:
            return None


def map_sources(packet, function):
    from mosaickv.evaluation.head_local_precision import HeadLocalFormat
    from mosaickv.evaluation.hybrid_reader import HybridFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    if isinstance(packet, LTSourceFormat):
        raise ValueError("native source substitution cannot be applied twice")
    if isinstance(packet, SharedSourceFormat):
        return function(packet)
    if isinstance(packet, HybridFormat):
        return (
            packet
            if packet.base is None
            else replace(packet, base=map_sources(packet.base, function))
        )
    if isinstance(packet, (HeadLocalFormat, TextSourceFormat)):
        return replace(packet, base=map_sources(packet.base, function))
    return packet


@torch.inference_mode()
def fit(base, attention, engine, *, direct_validation=False):
    from mosaickv.evaluation.full_prompt_source import restore_projected
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    if type(direct_validation) is not bool:
        raise ValueError("explicit direct-validation boolean required")
    if not isinstance(base, SharedSourceFormat) or base.protected_patches_disjoint is not True:
        raise ValueError("original constructor-certified disjoint source required")
    patches, checks = [], []
    for component in (0, 1):
        reference = torch.empty(base.shape, device=base.cos.device, dtype=torch.bfloat16)
        actual = torch.empty_like(reference)
        fill_component(base, attention, component, reference)
        engine.project(base, attention, component, actual)
        restore_projected(base, component, actual)
        new = SparseBitPatch.build(reference, actual)
        if direct_validation:
            # The original protected rows/transport are already restored.
            # Verify the exact new correction on these existing full K/V
            # buffers, rather than repeating the expensive matrix product.
            new.apply(actual)
            checks.append(
                (reference.view(torch.int16) == actual.view(torch.int16)).all().reshape(1)
            )
        old = (
            base.transport_patches[component]
            if base.transport_patches
            else SparseBitPatch(
                torch.empty(0, device=base.cos.device, dtype=torch.int32),
                torch.empty(0, device=base.cos.device, dtype=torch.bfloat16),
            )
        )
        ids, values = torch.cat((old.indices, new.indices)), torch.cat((old.values, new.values))
        patches.append(SparseBitPatch(ids, values))
        sorted_ids = ids.sort().values
        checks.extend(
            [
                ((ids >= 0) & (ids < reference.numel())).all().reshape(1),
                (sorted_ids[1:] != sorted_ids[:-1]).all().reshape(1),
                (~torch.isin(ids // 128, base.row_positions)).all().reshape(1),
            ]
        )
        del reference, actual, new
    if not bool(torch.cat(checks).all()):
        raise RuntimeError("native numerical patches conflict with original protected evidence")
    candidate = LTSourceFormat(base, tuple(patches))
    if not direct_validation:
        verify_strides(candidate, attention, engine)
    return candidate


@torch.inference_mode()
def verify_strides(candidate, attention, engine):
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    base = candidate.base
    with use(engine):
        for tail in (0, 1, 127):
            for component in (0, 1):
                reference = torch.full(
                    (*base.shape[:2], base.shape[2] + tail, 128),
                    -71.0,
                    device=base.cos.device,
                    dtype=torch.bfloat16,
                )
                actual = torch.full_like(reference, -71.0)
                fill_component(base, attention, component, reference)
                candidate.fill_component(attention, component, actual)
                if not torch.equal(reference.view(torch.int16), actual.view(torch.int16)):
                    raise RuntimeError(
                        "frozen native-source repair failed original K/V or history bits"
                    )
                del reference, actual


@torch.inference_mode()
def convert(packets, attentions, *, engine_factory=None, policy="full", audit=False):
    if type(policy) is not str or policy not in ("full", "direct") or type(audit) is not bool:
        raise ValueError("registered full/direct constructor and explicit audit boolean required")
    if len(packets) != len(attentions):
        raise ValueError("complete matching source/attention sequence required")
    from mosaickv.evaluation.verified_constructor import current as verified_state

    verified = verified_state()
    if verified is not None and verified.mode == "handoff":
        if engine_factory is not None:
            raise ValueError("verified handoff cannot replace its private engine factory")
        return verified.convert(packets, attentions, policy=policy, audit=audit)
    from mosaickv.evaluation.prefill_native_overlap import current

    if current() is not None:
        if engine_factory is not None:
            raise ValueError("overlap cannot replace a caller-supplied engine factory")
        return current().convert(packets, attentions, policy=policy, audit=audit)
    if engine_factory is None:
        from mosaickv.evaluation.joint_incoming_projection import engine_factory

    engine, records, result = None, [], []
    try:
        for index, (packet, attention) in enumerate(zip(packets, attentions, strict=True)):

            def build(base, attention=attention, index=index):
                nonlocal engine
                if engine is None:
                    engine = engine_factory(base)
                if policy == "full":
                    value = fit(base, attention, engine)
                else:
                    value = fit(base, attention, engine, direct_validation=True)
                    if audit:
                        verify_strides(value, attention, engine)
                records.append(
                    dict(
                        layer=index,
                        added_patch_bytes=backing_bytes(
                            tuple(t for p in value.repairs for t in p.tensors)
                        ),
                    )
                )
                return value

            result.append(map_sources(packet, build))
        if sum(r["added_patch_bytes"] for r in records) > 2**20:
            raise RuntimeError("request numerical repair storage exceeds one MiB")
        return tuple(result), dict(
            converted_layers=records,
            eligible_layers=len(records),
            added_patch_bytes=sum(r["added_patch_bytes"] for r in records),
            parent_KV_bitwise=True,
            original_headers_codes_weights_and_positions_unchanged=True,
            engine=None if engine is None else engine.record(),
            no_labels_or_answers_used=True,
            constructor_policy=policy,
            full_parent_KV_word_validation=True,
            exhaustive_stride_audit=policy == "full" or audit,
        )
    finally:
        if engine is not None:
            engine.close()
