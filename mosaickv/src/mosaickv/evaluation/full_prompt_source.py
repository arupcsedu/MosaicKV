"""Opt-in lossless execution replacement for frozen SharedSourceFormat packets.

The existing precision variant is NOT the complete six-mechanism algorithm.
No query/answer fitting: only differences from the original packet decoder are
stored. Native GEMM results cannot replace that decoder without bitwise repair.
"""

from dataclasses import dataclass, replace

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

POLICIES = {"kv": (0, 1), "v_only": (1,)}
MAXIMUM_REPAIR_BYTES = 3 * 2**20


def validate(policy):
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("explicit kv or v_only native-source policy required")


def source_format(packet):
    """Inspect immutable metadata; never advance a mutable layer-code replay."""
    from mosaickv.evaluation.layer_code_delta import BoundDeltaFormat, DeltaFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    if isinstance(packet, BoundDeltaFormat):
        packet = packet.specification
    if isinstance(packet, DeltaFormat):
        packet = packet.template
    if isinstance(packet, TextSourceFormat):
        packet = packet.base
    return packet if isinstance(packet, FullPromptSourceFormat) else None


def new_workspace(packet, device):
    fmt = source_format(packet)
    if fmt is None:
        raise ValueError("registered native full-prompt format required")
    _, heads, length, dim = fmt.shape
    return torch.empty((1, length, heads * dim), device=device, dtype=torch.bfloat16)


@torch.inference_mode()
def decode_into(packet, workspace):
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    if (
        not isinstance(packet, SharedSourceFormat)
        or packet.projection_backend != "fused"
        or packet.input_overlay is not None
        or workspace.shape != (1, packet.shape[-2], packet.shape[1] * packet.shape[-1])
        or workspace.dtype != torch.bfloat16
        or not workspace.is_contiguous()
        or workspace.device != packet.source.codes.device
        or packet.source.bits != 3
        or packet.source.axis != 2
        or packet.source.group != 128
    ):
        raise ValueError("original fused three-bit source and owned full BF16 workspace required")
    if workspace.device.type == "cuda":
        from mosaickv.evaluation.reconstruction_policy import reuse_headers
        from mosaickv.evaluation.shared_source_kernel import _decoded_source
        from mosaickv.evaluation.source_channel_exceptions import kernel_arguments

        columns, values, has_columns = kernel_arguments(packet)
        n, width = packet.source.shape[2:]
        _decoded_source[((n + 15) // 16, width // 64)](
            packet.source.codes,
            packet.source.minimum,
            packet.source.scale,
            columns,
            values,
            workspace,
            n,
            WIDTH=width,
            HAS_COLUMNS=has_columns,
            REUSE_HEADERS=reuse_headers(),
            enable_fp_fusion=False,
        )
    elif workspace.device.type == "cpu":
        from mosaickv.evaluation.precision_tier import unpack_cpu
        from mosaickv.evaluation.source_channel_exceptions import restore_cpu

        workspace.copy_(unpack_cpu(packet.source).squeeze(1))
        restore_cpu(packet.channel_exceptions, workspace)
    else:
        raise NotImplementedError("only CPU oracle and actual CUDA supported")


def projection_product(workspace, weights, output):
    """Unchanged default product; an isolated operator-study injection boundary."""
    torch.bmm(workspace.expand(output.shape[0], -1, -1), weights, out=output)


@torch.inference_mode()
def project_into(packet, attention, component, destination, workspace):
    from mosaickv.evaluation.exact_storage import decoded
    from mosaickv.evaluation.shared_source_format import apply_transport_patch

    _, h, n, d = packet.shape
    if type(component) is not int or component not in (0, 1):
        raise ValueError("explicit key/value component required")
    projection = (attention.k_proj, attention.v_proj)[component]
    if (
        d != 128
        or workspace.shape != (1, n, h * d)
        or workspace.dtype != torch.bfloat16
        or not workspace.is_contiguous()
        or destination.ndim != 4
        or destination.shape[:2] != (1, h)
        or destination.shape[-2] < n
        or destination.shape[-1] != d
        or destination.stride(-1) != 1
        or destination.stride(-2) != d
        or destination.dtype != torch.bfloat16
        or destination.device != workspace.device
        or workspace.device != packet.source.codes.device
        or projection.bias is not None
        or projection.weight.shape != (h * d, h * d)
        or projection.weight.dtype != torch.bfloat16
        or not projection.weight.is_contiguous()
        or projection.weight.device != workspace.device
    ):
        raise ValueError("native BF16 full-MHA weights, full source and strided lease required")
    weights = projection.weight.view(h, d, h * d).transpose(1, 2)
    output = destination[0, :, :n, :]
    projection_product(workspace, weights, output)
    if component == 0:
        if output.device.type == "cuda":
            from mosaickv.evaluation.shared_source_kernel import rotate_native

            rotate_native(destination, packet.cos, packet.sin, 0, n)
        else:
            key = output.clone()
            output.copy_(
                key * packet.cos + torch.cat((-key[..., 64:], key[..., :64]), -1) * packet.sin
            )
    if output.device.type == "cuda":
        from mosaickv.evaluation.head_local_precision_kernel import scatter

        scatter(packet.exact[component], packet.row_positions, destination, prompt_length=n)
    else:
        ids = packet.row_positions.long()
        destination[0, ids // n, ids % n] = decoded(packet.exact[component])[0, 0]
    if packet.transport_patches:
        apply_transport_patch(packet.transport_patches[component], destination, prompt_length=n)


@dataclass(frozen=True)
class FullPromptSourceFormat:
    base: object
    repairs: tuple
    components: tuple = (0, 1)
    kind: str = "lossless_full_prompt_native_source_execution_variant"

    def __post_init__(self):
        from mosaickv.evaluation.shared_source_format import SharedSourceFormat

        if (
            not isinstance(self.base, SharedSourceFormat)
            or type(self.components) is not tuple
            or any(type(c) is not int for c in self.components)
            or self.components not in tuple(POLICIES.values())
            or len(self.repairs) != 2
            or any(
                p.indices.dtype != torch.int32
                or p.values.dtype != torch.bfloat16
                or p.indices.ndim != 1
                or p.values.ndim != 1
                or p.indices.numel() != p.values.numel()
                or p.indices.device != self.base.cos.device
                or p.values.device != self.base.cos.device
                or not p.indices.is_contiguous()
                or not p.values.is_contiguous()
                for p in self.repairs
            )
        ):
            raise ValueError(
                "registered native components and same-device lossless repairs required"
            )

    @property
    def shape(self):
        return self.base.shape

    def private(self):
        return (*self.base.private(), *(t for p in self.repairs for t in p.tensors))

    def shared(self):
        return self.base.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill_component(self, attention, component, destination, workspace):
        from mosaickv.evaluation.sequential_kv_lease import fill_component
        from mosaickv.evaluation.shared_source_format import apply_transport_patch

        if component not in self.components:
            fill_component(self.base, attention, component, destination)
            return
        if workspace is None:
            raise RuntimeError("native source requires a charged, layer-owned workspace")
        if component == self.components[0]:
            decode_into(self.base, workspace)
        project_into(self.base, attention, component, destination, workspace)
        apply_transport_patch(self.repairs[component], destination, prompt_length=self.shape[-2])

    def fill(self, attention, targets, *, direct, workspace_observer=None):
        if not direct or len(targets) != 2:
            raise ValueError("explicit two-component native construction required")
        workspace = new_workspace(self, targets[0].device)
        if workspace_observer is not None:
            workspace_observer((workspace,))
        for component, target in enumerate(targets):
            self.fill_component(attention, component, target, workspace)

    def materialize(self, attention):
        targets = tuple(
            torch.empty(self.shape, device=self.base.cos.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        self.fill(attention, targets, direct=True)
        return targets


@torch.inference_mode()
def fit(base, attention, components):
    """Reference is the frozen packet decoder, NOT labels or a FullKV answer."""
    from mosaickv.evaluation.async_bitwise import match_flags
    from mosaickv.evaluation.sequential_kv_lease import fill_component
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    if (
        not isinstance(base, SharedSourceFormat)
        or type(components) is not tuple
        or any(type(c) is not int for c in components)
        or components not in tuple(POLICIES.values())
    ):
        raise ValueError("original source packet and registered component mask required")
    # Target the actual frozen K-then-V decoder, not a differently specialized
    # simultaneous K/V construction kernel. Unchanged components need no refit.
    reference = {
        c: torch.empty(base.shape, device=base.cos.device, dtype=torch.bfloat16) for c in components
    }
    for component in components:
        fill_component(base, attention, component, reference[component])

    def empty():
        return SparseBitPatch(
            torch.empty(0, device=base.cos.device, dtype=torch.int32),
            torch.empty(0, device=base.cos.device, dtype=torch.bfloat16),
        )

    tentative = FullPromptSourceFormat(base, (empty(), empty()), components)
    workspace = new_workspace(tentative, base.cos.device)
    actual = {c: torch.empty_like(reference[c]) for c in components}
    repairs = [empty(), empty()]
    for component in components:
        tentative.fill_component(attention, component, actual[component], workspace)
        repairs[component] = SparseBitPatch.build(reference[component], actual[component])
    result = replace(tentative, repairs=tuple(repairs))
    del actual, workspace
    # Refit is forbidden. Changed batch strides must use the SAME repair bits.
    checks = []
    for tail in (0, 1, 128):
        targets = {
            c: torch.full(
                (*base.shape[:2], base.shape[-2] + tail, 128),
                -71,
                device=base.cos.device,
                dtype=torch.bfloat16,
            )
            for c in components
        }
        workspace = new_workspace(result, base.cos.device)
        for component in components:
            target = targets[component]
            result.fill_component(attention, component, target, workspace)
            checks.append(match_flags(reference[component], target[..., : base.shape[-2], :]))
            checks.append((target[..., base.shape[-2] :, :] == -71).all().reshape(1))
        del target, targets, workspace
    # Identical exact predicates, three independent stride checks, one host
    # decision per layer rather than repeated per-component synchronizations.
    if not bool(torch.cat(checks).all()):
        raise RuntimeError("native source repairs changed a frozen decoder bit or history sentinel")
    return result


class FeasibilityDeclined(RuntimeError):
    def __init__(self, report):
        super().__init__("native source failed the registered lossless repair-capacity gate")
        self.report = report


@torch.inference_mode()
def convert(packets, attentions, policy):
    """Deterministic cheapest-repair layer subset, fixed 3 MiB total cap.

    No newly generated tokens or quality scores enter this constructor. The
    original code-stream conversion remains separate and unchanged afterward.
    """
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    validate(policy)
    if len(packets) != len(attentions):
        raise ValueError("complete layer/attention correspondence required")
    candidates, records = {}, []
    for index, (packet, attention) in enumerate(zip(packets, attentions, strict=True)):
        base = packet.base if isinstance(packet, TextSourceFormat) else packet
        if not isinstance(base, SharedSourceFormat):
            continue
        candidate = fit(base, attention, POLICIES[policy])
        cost = backing_bytes(tuple(t for p in candidate.repairs for t in p.tensors))
        candidates[index] = (
            replace(packet, base=candidate) if isinstance(packet, TextSourceFormat) else candidate
        )
        records.append(
            dict(
                layer=index,
                repair_bytes=cost,
                repair_scalars=[p.indices.numel() for p in candidate.repairs],
            )
        )
    used, selected = 0, []
    for record in sorted(records, key=lambda r: (r["repair_bytes"], r["layer"])):
        if used + record["repair_bytes"] <= MAXIMUM_REPAIR_BYTES:
            selected.append(record["layer"])
            used += record["repair_bytes"]
    report = dict(
        policy=policy,
        eligible_layers=len(records),
        converted_layers=sorted(selected),
        repair_bytes=used,
        maximum_repair_bytes=MAXIMUM_REPAIR_BYTES,
        records=records,
        no_labels_or_answers_accessed=True,
        all_accepted_KV_bits_identical=True,
        enabled=bool(selected),
        source_decodes_per_layer_step=1,
        workspace_bytes=max(
            (p.shape[-2] * p.shape[1] * p.shape[-1] * 2 for p in packets), default=0
        )
        if selected
        else 0,
    )
    if records and len(selected) < min(8, len(records)):
        raise FeasibilityDeclined(report)
    chosen = set(selected)
    return tuple(
        candidates[i] if i in chosen else packet for i, packet in enumerate(packets)
    ), report


def gpu_controls():
    """Small real kernels; correctness evidence, never task quality evidence."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from mosaickv.evaluation.sequential_kv_lease import fill_component
    from mosaickv.evaluation.shared_source_format import construct

    torch.manual_seed(20261103)
    h, n, d = 4, 521, 128
    device = torch.device("cuda")
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False, device=device, dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False, device=device, dtype=torch.bfloat16),
    )
    x = torch.randn(1, n, h * d, device=device).bfloat16()
    angle = torch.randn(1, n, 64, device=device).repeat(1, 1, 2)
    phases = angle.cos().bfloat16(), angle.sin().bfloat16()
    original = tuple(torch.randn(1, h, n, d, device=device).bfloat16() for _ in range(2))
    packet = construct(
        x,
        original,
        (tuple(range(n)),) * h,
        (0, n - 1),
        phases,
        (
            torch.arange(n, dtype=torch.int32, device=device),
            torch.arange(n, dtype=torch.int16, device=device)[None].expand(h, n),
        ),
    )
    packet = replace(packet, source_head_group=4)
    expected = tuple(
        torch.empty(packet.shape, device=device, dtype=torch.bfloat16) for _ in range(2)
    )
    for component in (0, 1):
        fill_component(packet, attention, component, expected[component])
    cases = []
    for policy, components in POLICIES.items():
        candidate = fit(packet, attention, components)
        workspace = new_workspace(candidate, device)
        targets = tuple(
            torch.full((1, h, n + 11, d), -71, device=device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        original_decode, calls = decode_into, []

        def counted(base, destination, *, calls=calls, original_decode=original_decode):
            calls.append(destination.data_ptr())
            original_decode(base, destination)

        with patch(__name__ + ".decode_into", counted):
            for component in (0, 1):
                candidate.fill_component(attention, component, targets[component], workspace)
        if calls != [workspace.data_ptr()]:
            raise RuntimeError("native source was not decoded exactly once")
        if any(
            not torch.equal(a.view(torch.int16), b[..., :n, :].view(torch.int16))
            or not bool((b[..., n:, :] == -71).all())
            for a, b in zip(expected, targets, strict=True)
        ):
            raise RuntimeError("native primitive changed parent bits or history sentinels")
        cases.append(
            dict(
                policy=policy,
                bitwise=True,
                one_source_decode=True,
                workspace_bytes=backing_bytes((workspace,)),
                lossless_repair_bytes=backing_bytes(
                    tuple(t for p in candidate.repairs for t in p.tensors)
                ),
                repair_scalars=[p.indices.numel() for p in candidate.repairs],
            )
        )
    torch.cuda.synchronize()
    return dict(cases=cases, independent_quality_N=0, same_frozen_decoder_not_FullKV=True)
