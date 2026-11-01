"""Full-prompt shared-input precision VARIANT, not the exact-selection method.

Store one quantized normalized input shared by K/V projections. Reconstruct
each original position, retaining the old per-head protected rows bitwise.
No model weights are modified, no RoPE phases are merged, no token is omitted.
"""

from contextlib import ExitStack
from dataclasses import dataclass

import torch

from mosaickv.evaluation.conditional_precision import pack_unprotected
from mosaickv.evaluation.exact_storage import decoded, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack, unpack_cpu
from mosaickv.evaluation.protected_projection_source import protected_positions


class SourceCapture:
    """One temporary original projection input per layer, same device, one prefill."""

    def __init__(self, adapter):
        self.adapter, self.stack, self.records = adapter, ExitStack(), {}

    def __enter__(self):
        try:
            for i, block in enumerate(self.adapter._language_layers()):

                def capture(_module, args, kwargs, *, index=i):
                    x = kwargs.get("hidden_states", args[0] if args else None)
                    if (
                        x is None
                        or index in self.records
                        or x.dtype != torch.bfloat16
                        or x.ndim != 3
                        or x.shape[0] != 1
                    ):
                        raise ValueError("one batch-one BF16 prefill source per layer required")
                    self.records[index] = x.detach().clone().contiguous()

                handle = block.self_attn.register_forward_pre_hook(capture, with_kwargs=True)
                self.stack.callback(handle.remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


@dataclass(frozen=True)
class SharedSourceFormat:
    source: object
    shape: tuple
    row_positions: object
    exact: tuple
    cos: object
    sin: object
    union: object
    slots: object
    transport_patches: tuple = ()
    projection_backend: str = "fused"
    source_head_group: int = 1
    input_overlay: object = None
    kind: str = "evicastkv_full_prompt_shared_source_precision_variant"
    channel_exceptions: object = None

    def private(self):
        return (
            *self.source.tensors(),
            self.row_positions,
            *(t for body in self.exact for t in body.buffers),
            *(t for patch in self.transport_patches for t in patch.tensors),
            *(() if self.input_overlay is None else self.input_overlay.tensors()),
            *(() if self.channel_exceptions is None else self.channel_exceptions.tensors()),
        )

    def shared(self):
        return self.cos, self.sin, self.union, self.slots

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    @torch.inference_mode()
    def fill(self, attention, targets, *, direct, workspace_observer=None):
        del direct
        if self.projection_backend not in ("fused", "native_bmm_256"):
            raise NotImplementedError("unregistered source projection backend")
        if self.input_overlay is not None and self.projection_backend != "native_bmm_256":
            raise NotImplementedError("exact input fusion requires audited native source backend")
        if self.channel_exceptions is not None and self.projection_backend != "fused":
            raise NotImplementedError("coherent columns currently require verified fused transport")
        n, h, d = self.shape[2], self.shape[1], self.shape[3]
        if (
            type(self.source_head_group) is not int
            or self.source_head_group not in (1, 2, 4)
            or h % self.source_head_group
        ):
            raise ValueError("divisible explicit head group required")
        if (
            attention is None
            or len(targets) != 2
            or any(
                t.ndim != 4
                or t.shape[:2] != (1, h)
                or t.shape[-1] != d
                or t.shape[-2] < n
                or t.dtype != torch.bfloat16
                or t.device != self.source.codes.device
                for t in targets
            )
        ):
            raise ValueError("actual frozen attention and complete layer lease required")
        if targets[0].device.type == "cuda":
            if self.projection_backend == "native_bmm_256":
                from mosaickv.evaluation.native_source_chunks import (
                    new_workspace,
                    project_component,
                )

                workspace = new_workspace(self)
                if workspace_observer is not None:
                    workspace_observer((workspace,))
                for component, target in enumerate(targets):
                    project_component(self, attention, component, target, workspace)
            elif self.projection_backend == "fused":
                from mosaickv.evaluation.shared_source_kernel import project

                project(self, attention, targets)
            else:
                raise NotImplementedError("unregistered source projection backend")
            from mosaickv.evaluation.head_local_precision_kernel import scatter

            for body, target in zip(self.exact, targets, strict=True):
                scatter(body, self.row_positions, target, prompt_length=n)
        else:
            x = unpack_cpu(self.source).squeeze(1)
            if self.channel_exceptions is not None:
                from mosaickv.evaluation.source_channel_exceptions import restore_cpu

                x = restore_cpu(self.channel_exceptions, x)
            if self.input_overlay is not None:
                from mosaickv.evaluation.integrated_source_input import overlay_input

                overlay_input(self.input_overlay, x, 0)
            k = attention.k_proj(x).reshape(1, n, h, d).transpose(1, 2)
            v = attention.v_proj(x).reshape(1, n, h, d).transpose(1, 2)
            rotated = torch.cat((-k[..., d // 2 :], k[..., : d // 2]), -1)
            k = k * self.cos[:, None] + rotated * self.sin[:, None]
            for value, target, body in zip((k, v), targets, self.exact, strict=True):
                target[..., :n, :].copy_(value)
                ids = self.row_positions.long()
                target[0, ids // n, ids % n] = decoded(body)[0, 0]
        for patch, target in zip(
            self.transport_patches, targets if self.transport_patches else (), strict=True
        ):
            apply_transport_patch(patch, target, prompt_length=n)

    def materialize(self, attention):
        out = tuple(
            torch.empty(self.shape, device=self.source.codes.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        self.fill(attention, out, direct=True)
        return out


@torch.inference_mode()
def construct(
    normalized, original_pair, memberships, text, phases, identity, *, exclude_global_exact=False
):
    if type(exclude_global_exact) is not bool or len(original_pair) != 2:
        raise ValueError("explicit source variant and actual K/V pair required")
    key, value = original_pair
    if (
        key.shape != value.shape
        or key.ndim != 4
        or key.shape[0] != 1
        or key.shape[-1] != 128
        or key.dtype != torch.bfloat16
        or value.dtype != key.dtype
    ):
        raise ValueError("batch-one full-MHA BF16 original cache required")
    _, h, n, d = key.shape
    if (
        len(memberships) != h
        or n >= 32768
        or normalized.shape != (1, n, h * d)
        or normalized.device != key.device
        or normalized.dtype != key.dtype
    ):
        raise ValueError("full normalized projection input, not GQA, required")
    cos, sin = phases
    union, slots = identity
    if (
        cos.shape != (1, n, d)
        or sin.shape != cos.shape
        or union.shape != (n,)
        or slots.shape != (h, n)
        or any(t.device != key.device for t in (value, cos, sin, union, slots))
        or cos.dtype != torch.bfloat16
        or sin.dtype != cos.dtype
        or union.dtype != torch.int32
        or slots.dtype != torch.int16
        or not torch.equal(union, torch.arange(n, device=key.device, dtype=torch.int32))
        or not torch.equal(slots, union.to(torch.int16)[None].expand(h, n))
        or not all(torch.isfinite(t).all() for t in (normalized, key, value, cos, sin))
        or not torch.equal(cos[..., :64], cos[..., 64:])
        or not torch.equal(sin[..., :64], sin[..., 64:])
    ):
        raise ValueError("shared original phase and full identity positions required")
    mask = protected_positions(memberships, text, length=n).to(key.device)
    rows = mask.flatten().nonzero().flatten().to(torch.int32)
    if exclude_global_exact:
        source = pack_unprotected(normalized[:, None], mask.all(0)[None], bits=3, axis=2)
    else:
        source = pack(normalized[:, None], bits=3, axis=2)
    head, position = rows.long() // n, rows.long() % n
    exact = tuple(
        encode(t[0, head, position].reshape(1, 1, len(rows), d).contiguous()) for t in original_pair
    )
    result = SharedSourceFormat(source, tuple(key.shape), rows, exact, cos, sin, union, slots)
    forbidden = {t.untyped_storage().data_ptr() for t in (*original_pair, normalized)}
    if any(t.numel() and t.untyped_storage().data_ptr() in forbidden for t in result.private()):
        raise RuntimeError("encoded source or exact rows alias an original full cache/input owner")
    return result


def apply_transport_patch(patch, target, *, prompt_length):
    if not patch.indices.numel():
        return
    if target.device.type == "cuda":
        from mosaickv.evaluation.dual_projection_kernel import apply_patch

        apply_patch(patch, target[..., :prompt_length, :])
    else:
        ids = patch.indices.long()
        d = target.shape[-1]
        target[0, ids // (prompt_length * d), ids // d % prompt_length, ids % d] = patch.values


def transport_corrections(reference, actual, protected_rows, *, chunk_scalars=None):
    """Fix only violations of the FROZEN construction envelope, not task logits."""
    from mosaickv.evaluation.exact_storage import SparseBitPatch

    patches = []
    if chunk_scalars is not None and (type(chunk_scalars) is not int or chunk_scalars < 1):
        raise ValueError("positive explicit scalar chunk required")
    d = reference[0].shape[-1]
    for expected, output in zip(reference, actual, strict=True):
        if (
            expected.shape != output.shape
            or expected.dtype != torch.bfloat16
            or output.dtype != expected.dtype
            or not torch.isfinite(output).all()
        ):
            raise ValueError("finite compatible BF16 construction pair required")
        if chunk_scalars is None:
            bad = ~torch.isclose(expected.float(), output.float(), rtol=0.01, atol=2**-6)
            indices = bad.flatten().nonzero().flatten()
        else:
            left, right = expected.flatten(), output.flatten()
            parts = []
            for start in range(0, left.numel(), chunk_scalars):
                stop = start + chunk_scalars
                bad = ~torch.isclose(
                    left[start:stop].float(), right[start:stop].float(), rtol=0.01, atol=2**-6
                )
                parts.append(bad.nonzero().flatten() + start)
            indices = torch.cat(parts)
        if torch.isin(indices // d, protected_rows.long()).any():
            raise RuntimeError("protected K/V mismatch may not be hidden by a numerical patch")
        patches.append(
            SparseBitPatch(
                indices.to(torch.int32),
                expected.contiguous().flatten().index_select(0, indices).clone(),
            )
        )
    return tuple(patches)


def assert_transport_close(output, expected, *, chunk_scalars=524288):
    """Original assert_close on every scalar, without full-component scratch.

    Chunking does not alter elementwise math, tolerances or the pass predicate;
    only a failing assertion's diagnostic index becomes chunk-local.
    """
    if type(chunk_scalars) is not int or chunk_scalars < 1 or output.shape != expected.shape:
        raise ValueError("same shape and positive explicit verification chunk required")
    left, right = output.flatten(), expected.flatten()
    for start in range(0, left.numel(), chunk_scalars):
        torch.testing.assert_close(
            left[start : start + chunk_scalars],
            right[start : start + chunk_scalars],
            rtol=0.01,
            atol=2**-6,
        )


@torch.inference_mode()
def fit_transport(packet, attention, *, componentwise=False):
    """Native projection of the SAME encoded source; no FullKV/answer targets."""
    from dataclasses import replace

    from mosaickv.evaluation.shared_source_kernel import decode_source_for_construction

    if packet.transport_patches or packet.input_overlay is not None:
        raise ValueError("numerical transport fitting is a single immutable construction")
    if type(componentwise) is not bool:
        raise ValueError("explicit componentwise construction flag required")
    if componentwise:
        return fit_transport_components(packet, attention)
    x = decode_source_for_construction(packet)
    n, h, d = packet.shape[2], packet.shape[1], packet.shape[3]
    k = attention.k_proj(x).reshape(1, n, h, d).transpose(1, 2)
    v = attention.v_proj(x).reshape(1, n, h, d).transpose(1, 2)
    # Projection results own their storage under inference_mode; the decoded
    # source is not needed by RoPE, correction fitting or any verification.
    del x
    k = k * packet.cos[:, None] + torch.cat((-k[..., 64:], k[..., :64]), -1) * packet.sin[:, None]
    reference = (k.contiguous(), v.contiguous())
    del k, v
    actual = packet.materialize(attention)
    ids = packet.row_positions.long()
    head, position = ids // n, ids % n
    for target, body in zip(reference, packet.exact, strict=True):
        target[0, head, position] = decoded(body)[0, 0]
    patches = transport_corrections(reference, actual, packet.row_positions)
    # Sparse patches own copied indexed values, not views of the old outputs.
    # Keep every original correction and tolerance check, without overlapping
    # an otherwise dead K/V pair with the independent verification pair.
    del actual
    result = replace(packet, transport_patches=patches)
    verified = result.materialize(attention)
    for expected, output in zip(reference, verified, strict=True):
        torch.testing.assert_close(output, expected, rtol=0.01, atol=2**-6)
    return result, dict(
        patch_scalars=[p.indices.numel() for p in patches],
        persistent_patch_bytes=backing_bytes(tuple(t for p in patches for t in p.tensors)),
        frozen_construction_tolerance_pass=True,
        no_task_or_fullkv_target_for_unprotected=True,
        construction_has_temporary_decoded_source_and_native_projection=True,
    )


@torch.inference_mode()
def fit_transport_components(packet, attention):
    """Same native reference/checks, one private K/V component at a time.

    The paired constructor remains the default. This opt-in VARIANT must
    reproduce its complete frozen packet in GPU tests. BF16 operations remain
    separate ATen multiplications/addition, not fused addcmul or altered RoPE.
    """
    from dataclasses import replace

    from mosaickv.evaluation.exact_storage import SparseBitPatch
    from mosaickv.evaluation.sequential_kv_lease import fill_component
    from mosaickv.evaluation.shared_source_kernel import decode_source_for_construction

    if packet.transport_patches or packet.input_overlay is not None:
        raise ValueError("numerical transport fitting is a single immutable construction")
    n, h, d = packet.shape[2], packet.shape[1], packet.shape[3]
    if d != 128:
        raise NotImplementedError("audited full-MHA head dimension 128 required")
    dummy = SparseBitPatch(
        torch.empty(0, dtype=torch.int32, device=packet.cos.device),
        torch.empty(0, dtype=torch.bfloat16, device=packet.cos.device),
    )
    ids = packet.row_positions.long()
    head, position = ids // n, ids % n
    patches = []
    for component in range(2):
        x = decode_source_for_construction(packet)
        projection = attention.k_proj if component == 0 else attention.v_proj
        native = projection(x).reshape(1, n, h, d).transpose(1, 2)
        del x
        if component == 0:
            rotated = torch.cat((-native[..., 64:], native[..., :64]), -1)
            native.mul_(packet.cos[:, None])
            rotated.mul_(packet.sin[:, None])
            native.add_(rotated)
            del rotated
        reference = native.contiguous()
        del native
        reference[0, head, position] = decoded(packet.exact[component])[0, 0]
        actual = torch.empty_like(reference)
        fill_component(packet, attention, component, actual)
        patch = transport_corrections(
            (reference,), (actual,), packet.row_positions, chunk_scalars=524288
        )[0]
        del actual
        temporary = replace(
            packet, transport_patches=tuple(patch if j == component else dummy for j in range(2))
        )
        verified = torch.empty_like(reference)
        fill_component(temporary, attention, component, verified)
        assert_transport_close(verified, reference)
        patches.append(patch)
        del reference, verified, temporary
    result = replace(packet, transport_patches=tuple(patches))
    return result, dict(
        patch_scalars=[p.indices.numel() for p in patches],
        persistent_patch_bytes=backing_bytes(tuple(t for p in patches for t in p.tensors)),
        frozen_construction_tolerance_pass=True,
        no_task_or_fullkv_target_for_unprotected=True,
        construction_has_temporary_decoded_source_and_native_projection=True,
        componentwise_reference_and_independent_verification=True,
        private_BF16_RoPE_operations_not_fused=True,
        bounded_verification_chunk_scalars=524288,
    )
