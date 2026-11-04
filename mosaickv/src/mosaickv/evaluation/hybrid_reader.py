"""Consumed-input, cost-aware LOSSLESS execution variant; not a paper method.

Choices never see prompts, answers, references, labels, or generated logits.
Source ownership and every replacement buffer are explicit. The frozen cost
table is a diagnostic input; learning a deployment policy is outside this API.
"""

import math
from dataclasses import dataclass

import torch

from mosaickv.evaluation.lossless_bf16 import PackedBF16, unpack_into
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

MODES = ("source", "encoded", "resident")


@dataclass(frozen=True)
class ReaderOption:
    modes: tuple[str, str]
    private_bytes: int
    construction_ns: int
    read_ns: int

    def __post_init__(self):
        if (
            type(self.modes) is not tuple
            or len(self.modes) != 2
            or any(m not in MODES for m in self.modes)
            or any(
                type(v) is not int or v < 0
                for v in (self.private_bytes, self.construction_ns, self.read_ns)
            )
        ):
            raise ValueError("two registered modes and nonnegative integer costs required")


def choose(options, *, private_cap_bytes, decode_steps, maximum_states=20000):
    """Exact multiple-choice Pareto DP. No byte rounding or heuristic pruning.

    A bounded state explosion fails explicitly, rather than silently changing
    the registered optimization. Shared storage is reserved by the caller.
    """
    if (
        not options
        or type(private_cap_bytes) is not int
        or private_cap_bytes < 0
        or type(decode_steps) is not int
        or decode_steps < 1
        or type(maximum_states) is not int
        or maximum_states < 1
    ):
        raise ValueError("nonempty routes, hard integer cap and positive horizon required")
    states = {0: (0, ())}
    for route in options:
        if (
            not route
            or any(not isinstance(o, ReaderOption) for o in route)
            or len({o.modes for o in route}) != len(route)
        ):
            raise ValueError("nonempty unique typed route options required")
        generated = {}
        for cost, (time, plan) in states.items():
            for option in sorted(route, key=lambda o: o.modes):
                new_cost = cost + option.private_bytes
                if new_cost > private_cap_bytes:
                    continue
                value = (
                    time + option.construction_ns + decode_steps * option.read_ns,
                    (*plan, option.modes),
                )
                if new_cost not in generated or value < generated[new_cost]:
                    generated[new_cost] = value
        states, best = {}, math.inf
        for cost, value in sorted(generated.items()):
            if value[0] < best:
                states[cost], best = value, value[0]
        if not states:
            raise ValueError("no physically feasible representation plan")
        if len(states) > maximum_states:
            raise ValueError("exact cost frontier exceeds registered state bound")
    cost, (time, plan) = min(states.items(), key=lambda x: (x[1][0], x[0], x[1][1]))
    return dict(
        modes=plan,
        predicted_ns=time,
        private_bytes=cost,
        private_cap_bytes=private_cap_bytes,
        exact_frontier=True,
    )


@dataclass(frozen=True)
class HybridFormat:
    shape: tuple
    modes: tuple
    base: object
    bodies: tuple
    rotary: tuple
    kind: str = "lossless_cost_aware_hybrid_reader_variant"

    def __post_init__(self):
        if (
            len(self.shape) != 4
            or self.shape[0] != 1
            or any(type(v) is not int or v < 1 for v in self.shape)
            or type(self.modes) is not tuple
            or len(self.modes) != 2
            or any(m not in MODES for m in self.modes)
            or type(self.bodies) is not tuple
            or len(self.bodies) != 2
            or (self.base is None) != ("source" not in self.modes)
        ):
            raise ValueError("complete shape, modes and exact source ownership required")
        for mode, body in zip(self.modes, self.bodies, strict=True):
            if mode == "source":
                if body is not None or self.base.shape != self.shape:
                    raise ValueError("source mode must retain only its original packet")
            elif (
                not isinstance(body, PackedBF16)
                or body.shape != self.shape
                or (mode == "resident" and not body.raw)
            ):
                raise ValueError("same-shape lossless body (raw for resident) required")

    def private(self):
        return (
            *(() if self.base is None else self.base.private()),
            *(t for b in self.bodies if b is not None for t in b.buffers),
        )

    def shared(self):
        return self.rotary

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill_component(self, attention, component, destination, *, source_workspace=None):
        from mosaickv.evaluation.sequential_kv_lease import fill_component

        if type(component) is not int or component not in (0, 1):
            raise ValueError("explicit K/V component required")
        if self.modes[component] == "source":
            fill_component(
                self.base, attention, component, destination, source_workspace=source_workspace
            )
        else:
            unpack_into(self.bodies[component], destination)

    def fill(self, attention, targets, *, direct):
        if not direct or len(targets) != 2:
            raise ValueError("explicit direct two-component fill required")
        for component, target in enumerate(targets):
            self.fill_component(attention, component, target)


@torch.inference_mode()
def replace_packet(packet, attention, modes):
    """Construct only selected replacements; do not retain uncharged originals."""
    from mosaickv.evaluation.exact_storage import encode
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    if type(modes) is not tuple or len(modes) != 2 or any(m not in MODES for m in modes):
        raise ValueError("two explicit registered modes required")
    if modes == ("source", "source"):
        return packet
    bodies = []
    for component, mode in enumerate(modes):
        if mode == "source":
            bodies.append(None)
            continue
        original = torch.empty(
            packet.shape, device=packet.private()[0].device, dtype=torch.bfloat16
        )
        fill_component(packet, attention, component, original)
        body = (
            encode(original)
            if mode == "encoded"
            else PackedBF16(tuple(original.shape), True, (original,))
        )
        # Integer comparison preserves signed zero, NaN payloads and infinities.
        actual = torch.empty_like(original)
        unpack_into(body, actual)
        if not torch.equal(original.view(torch.int16), actual.view(torch.int16)):
            raise RuntimeError("hybrid encoding changed a parent K/V bit")
        bodies.append(body)
    return HybridFormat(
        tuple(packet.shape),
        modes,
        packet if "source" in modes else None,
        tuple(bodies),
        tuple(packet.shared()),
    )


def convert(packets, attentions, policy):
    """Apply an immutable per-input diagnostic plan before code serialization."""
    if (
        set(policy) != {"modes", "maximum_added_bytes", "base_private_bytes"}
        or type(policy["maximum_added_bytes"]) is not int
        or policy["maximum_added_bytes"] < 0
        or type(policy["base_private_bytes"]) is not int
        or policy["base_private_bytes"] < 0
        or len(packets) != len(attentions)
        or len(packets) != len(policy["modes"])
    ):
        raise ValueError("complete strictly scoped hybrid policy required")
    old = backing_bytes(tuple(t for p in packets for t in p.private()))
    if old != policy["base_private_bytes"]:
        raise ValueError("hybrid policy bound to different input storage")
    results = tuple(
        replace_packet(p, a, tuple(m))
        for p, a, m in zip(packets, attentions, policy["modes"], strict=True)
    )
    new = backing_bytes(tuple(t for p in results for t in p.private()))
    if new > old + policy["maximum_added_bytes"]:
        raise RuntimeError("hybrid representation exceeded its charged hard storage cap")
    return results, dict(
        original_private_bytes=old,
        constructed_private_bytes=new,
        maximum_added_bytes=policy["maximum_added_bytes"],
        changed_components=sum(m != "source" for row in policy["modes"] for m in row),
        no_selection_or_positions_changed=True,
        parent_KV_bitwise=True,
        no_labels_or_answers_accessed=True,
    )
