"""Isolated eight-bit shared-source control; production three-bit reader unchanged.

This tests quantization error after a K/V factorial implicated their interaction.
It is not ``mosaickv_exact``, a six-mechanism result, or a new byte-matched claim.
The original normalized projection input is encoded without token/label access.
All protected K/V, phase maps, numerical patches and reconstruction buffers count.
"""

from dataclasses import dataclass, replace

import torch

from mosaickv.evaluation.exact_storage import decoded
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack, unpack_cpu
from mosaickv.evaluation.shared_source_format import (
    apply_transport_patch,
    transport_corrections,
)

BITS = 8
LABEL = "evicastkv_eight_bit_shared_source_fidelity_control"


def validate(packet, attention, targets):
    body = packet.source
    h, n, d = packet.shape[1:]
    if (
        body.bits != BITS
        or body.axis != 2
        or body.group != 128
        or body.shape != (1, 1, n, h * d)
        or d != 128
        or body.row_map is not None
        or body.codebook is not None
        or packet.channel_exceptions is not None
        or packet.input_overlay is not None
        or packet.projection_backend != "fused"
        or packet.source_head_group != 1
        or packet.source_row_tile != 16
        or packet.source_num_warps is not None
    ):
        raise ValueError("registered eight-bit, complete, full-MHA source geometry required")
    if len(targets) != 2 or any(
        t.ndim != 4
        or t.shape[:2] != (1, h)
        or t.shape[-2] < n
        or t.shape[-1] != d
        or t.dtype != torch.bfloat16
        or t.device != body.codes.device
        or t.stride(-1) != 1
        or t.stride(-2) != d
        for t in targets
    ):
        raise ValueError("compatible complete strided K/V leases required")
    if targets[0].stride(1) != targets[1].stride(1):
        raise ValueError("equal K/V head strides required")
    for projection in (attention.k_proj, attention.v_proj):
        if (
            projection.bias is not None
            or projection.weight.shape != (h * d, h * d)
            or projection.weight.dtype != torch.bfloat16
            or projection.weight.device != body.codes.device
            or not projection.weight.is_contiguous()
        ):
            raise ValueError("unchanged contiguous bias-free BF16 projection required")


@dataclass(frozen=True)
class PrecisionSourceControl:
    base: object
    kind: str = LABEL

    @property
    def shape(self):
        return self.base.shape

    @property
    def source(self):
        return self.base.source

    def private(self):
        return self.base.private()

    def shared(self):
        return self.base.shared()

    def tensors(self):
        return self.base.tensors()

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    @torch.inference_mode()
    def fill(self, attention, targets, *, direct, workspace_observer=None):
        del direct, workspace_observer
        validate(self.base, attention, targets)
        if targets[0].device.type == "cpu":
            return self.base.fill(attention, targets, direct=True)
        if targets[0].device.type != "cuda":
            raise NotImplementedError("CPU oracle and isolated CUDA source control only")
        # The already audited arithmetic has a BITS constexpr. Its public
        # three-bit validation/dispatch is intentionally NOT weakened here.
        import triton

        from mosaickv.evaluation.head_local_precision_kernel import scatter
        from mosaickv.evaluation.shared_source_kernel import _project

        body = self.source
        h, n, d = self.shape[1:]
        _project[(triton.cdiv(n, 16), h)](
            body.codes,
            body.minimum,
            body.scale,
            body.codes,
            body.codes,
            attention.k_proj.weight,
            attention.v_proj.weight,
            self.base.cos,
            self.base.sin,
            *targets,
            n,
            targets[0].stride(1),
            body.codes,
            WIDTH=h * d,
            DIM=d,
            BITS=BITS,
            COMPONENT=-1,
            HEAD_GROUP=1,
            HAS_COLUMNS=False,
            ROW_TILE=16,
            REUSE_HEADERS=False,
            ROW4=False,
            FIRST=0,
            TAKE=0,
            V_PREFIX=False,
            SOURCE_ROWS=0,
            HAS_ROW_MAP=False,
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
        for body, target in zip(self.base.exact, targets, strict=True):
            scatter(body, self.base.row_positions, target, prompt_length=n)
        for patch, target in zip(
            self.base.transport_patches,
            targets if self.base.transport_patches else (),
            strict=True,
        ):
            apply_transport_patch(patch, target, prompt_length=n)

    def materialize(self, attention):
        targets = tuple(
            torch.empty(self.shape, device=self.source.codes.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        self.fill(attention, targets, direct=True)
        return targets


@torch.inference_mode()
def decode_source(body):
    """Native decode of registered byte codes; this temporary is fully charged."""
    if body.bits != BITS or body.axis != 2 or body.row_map is not None:
        raise ValueError("uncompacted eight-bit per-channel body required")
    if body.codes.device.type == "cpu":
        return unpack_cpu(body).squeeze(1)
    if body.codes.device.type != "cuda":
        raise NotImplementedError("CPU/CUDA only")
    n, width = body.shape[2:]
    # Eight bits have one byte per code, in the quantizer's channel/time order.
    codes = body.codes.reshape(width, n).float()
    minimum = body.minimum.reshape(width, -1).repeat_interleave(128, -1)[:, :n]
    scale = body.scale.reshape(width, -1).repeat_interleave(128, -1)[:, :n]
    # Preserve separate native FP32 multiply and add; no fused operation.
    return (codes * scale + minimum).T.contiguous().unsqueeze(0).bfloat16()


@torch.inference_mode()
def construct(parent, normalized, attention):
    """Encode higher precision, preserving the already frozen protection policy.

    Parent is unpatched three-bit storage, not a benchmark-dependent proposal.
    Its exact protected rows, phase maps and positions are reused unchanged.
    There is no input for generated tokens, answers, labels or reference logits.
    """
    if (
        parent.source.bits != 3
        or parent.transport_patches
        or parent.input_overlay is not None
        or parent.channel_exceptions is not None
        or parent.source.row_map is not None
        or normalized.shape != (1, parent.shape[2], parent.shape[1] * parent.shape[3])
    ):
        raise ValueError("unpatched original three-bit source and normalized input required")
    body = pack(normalized[:, None], bits=BITS, axis=2)
    if body.codes.untyped_storage().data_ptr() == normalized.untyped_storage().data_ptr():
        raise RuntimeError("encoded payload aliases the original normalized source")
    base = replace(parent, source=body)
    result = PrecisionSourceControl(base)
    x = decode_source(body)
    h, n, d = parent.shape[1:]
    k = attention.k_proj(x).reshape(1, n, h, d).transpose(1, 2)
    v = attention.v_proj(x).reshape(1, n, h, d).transpose(1, 2)
    del x
    rotated = torch.cat((-k[..., d // 2 :], k[..., : d // 2]), -1)
    k = k * parent.cos[:, None] + rotated * parent.sin[:, None]
    reference = (k.contiguous(), v.contiguous())
    ids = parent.row_positions.long()
    for expected, exact in zip(reference, parent.exact, strict=True):
        expected[0, ids // n, ids % n] = decoded(exact)[0, 0]
    actual = result.materialize(attention)
    patches = transport_corrections(reference, actual, parent.row_positions)
    del actual
    result = PrecisionSourceControl(replace(base, transport_patches=patches))
    verified = result.materialize(attention)
    for expected, value, exact in zip(reference, verified, parent.exact, strict=True):
        torch.testing.assert_close(value, expected, rtol=0.01, atol=2**-6)
        protected = value[0, ids // n, ids % n].contiguous().view(torch.uint8)
        original = decoded(exact)[0, 0].contiguous().view(torch.uint8)
        if not torch.equal(protected, original):
            raise RuntimeError("higher precision altered protected original K/V")
    return result, dict(
        source_bits=BITS,
        implementation_label=LABEL,
        persistent_bytes=result.component_bytes,
        numerical_patch_bytes=backing_bytes(tuple(t for p in patches for t in p.tensors)),
        protected_original_KV_bitwise=True,
        original_logical_positions_preserved=True,
        no_original_source_alias=True,
        production_three_bit_path_changed=False,
        construction_workspace_charged_to_allocator=True,
        no_benchmark_labels_or_reference_logits=True,
        no_quality_or_latency_claim=True,
    )
