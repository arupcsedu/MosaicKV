from itertools import product

import pytest
import torch

from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.hybrid_reader import (
    HybridFormat,
    ReaderOption,
    choose,
    convert,
    replace_packet,
)
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.sequential_kv_lease import fill_component


def packet():
    # Include signed zero and unusual BF16 words: float comparisons would miss
    # signed-zero changes and reject otherwise preserved NaN payloads.
    bits = torch.arange(2 * 17 * 128, dtype=torch.int16).reshape(1, 2, 17, 128)
    return ExactFormat(
        "kv",
        tuple(bits.shape),
        tuple(
            PackedBF16(tuple(bits.shape), True, ((bits + offset).view(torch.bfloat16),))
            for offset in (0, 100)
        ),
    )


@pytest.mark.parametrize("modes", list(product(("source", "encoded", "resident"), repeat=2)))
def test_each_component_mode_preserves_bits_strided_history_and_ownership(modes):
    base = packet()
    result = replace_packet(base, None, modes)
    for component in (0, 1):
        storage = torch.full((1, 2, 26, 128), -71, dtype=torch.bfloat16)
        fill_component(result, None, component, storage[..., 2:24, :])
        assert torch.equal(
            storage[..., 2:19, :].view(torch.int16),
            base.pair[component].buffers[0].view(torch.int16),
        )
        assert (storage[..., :2, :] == -71).all() and (storage[..., 19:, :] == -71).all()
    if "source" not in modes:
        assert result.base is None
        original_ptrs = {t.untyped_storage().data_ptr() for t in base.tensors()}
        assert not original_ptrs.intersection(
            t.untyped_storage().data_ptr() for t in result.tensors()
        )


def test_request_replacements_are_isolated_and_labels_are_rejected():
    base = packet()
    policy = dict(
        modes=[["resident", "encoded"]],
        maximum_added_bytes=10**6,
        base_private_bytes=backing_bytes(base.private()),
    )
    a, _ = convert((base,), (None,), policy)
    b, _ = convert((base,), (None,), policy)
    assert a[0].bodies[0].buffers[0].data_ptr() != b[0].bodies[0].buffers[0].data_ptr()
    with pytest.raises(ValueError, match="strictly scoped"):
        convert((base,), (None,), dict(policy, reference="forbidden"))
    with pytest.raises(ValueError, match="different input"):
        convert((base,), (None,), dict(policy, base_private_bytes=1))


def test_hard_cap_checks_constructed_storage_not_predicted_cost():
    base = packet()
    with pytest.raises(RuntimeError, match="hard storage cap"):
        convert(
            (base,),
            (None,),
            dict(
                modes=[["source", "resident"]],
                maximum_added_bytes=0,
                base_private_bytes=backing_bytes(base.private()),
            ),
        )


def test_lossless_code_mapper_preserves_direct_bodies_and_avoids_source_visits():
    from mosaickv.evaluation.layer_code_delta import map_bodies

    base = replace_packet(packet(), None, ("resident", "resident"))

    def forbidden(*args):
        raise AssertionError("no source body exists")

    assert map_bodies(base, forbidden) is base


def test_exact_frontier_agrees_with_exhaustive_and_uses_construction_cost():
    options = [
        [
            ReaderOption(("source", "source"), 4, 0, 10),
            ReaderOption(("encoded", "encoded"), 7, 20, 1),
        ],
        [
            ReaderOption(("source", "source"), 5, 0, 8),
            ReaderOption(("resident", "resident"), 9, 30, 0),
        ],
    ]
    for cap in range(9, 18):
        for horizon in (1, 127):
            feasible = [
                (
                    sum(o.construction_ns + horizon * o.read_ns for o in pair),
                    sum(o.private_bytes for o in pair),
                    tuple(o.modes for o in pair),
                )
                for pair in product(*options)
                if sum(o.private_bytes for o in pair) <= cap
            ]
            result = choose(options, private_cap_bytes=cap, decode_steps=horizon)
            assert (result["predicted_ns"], result["private_bytes"], result["modes"]) == min(
                feasible
            )
            assert result == choose(
                [list(reversed(r)) for r in options], private_cap_bytes=cap, decode_steps=horizon
            )


def test_planner_fails_closed_on_bad_metadata_or_state_explosion():
    with pytest.raises(ValueError):
        ReaderOption(("source", "source"), True, 0, 0)
    with pytest.raises(ValueError):
        choose(
            [[ReaderOption(("source", "source"), 10, 0, 0)]], private_cap_bytes=9, decode_steps=127
        )
    with pytest.raises(ValueError, match="state bound"):
        choose(
            [
                [
                    ReaderOption(("source", "source"), 1, 0, 2),
                    ReaderOption(("resident", "resident"), 2, 0, 1),
                ]
            ],
            private_cap_bytes=2,
            decode_steps=127,
            maximum_states=1,
        )
    with pytest.raises(ValueError):
        HybridFormat((1, 2, 3, 128), ("source", "source"), None, (None, None), ())
