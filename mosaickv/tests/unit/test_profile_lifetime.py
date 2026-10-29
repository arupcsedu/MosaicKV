"""Property-based arithmetic checks only; no benchmark answers or model loads."""

import random

import pytest

from mosaickv.selection.profile_lifetime import allocate_profile_lifetime


def test_joint_allocation_redistributes_not_increases_total_history():
    from mosaickv.selection.profile_lifetime import allocate_profile_history

    values = allocate_profile_history([0.1, 0.8, 0.5], 128, 69)
    assert sum(values) == 384 and min(values) >= 69
    assert values[0] > values[2] > values[1]
    plan = allocate_profile_lifetime(
        [0.1, 0.8, 0.5],
        [600] * 3,
        [25] * 3,
        [16384] * 3,
        cap_bytes=24 * 2**20,
        generated_capacity=128,
        history_allocation="profile_weighted",
        minimum_history_capacity=69,
    )
    assert plan.history_positions == values and plan.history_bytes == 384 * 16384
    assert plan.charged_ceiling_bytes <= plan.cap_bytes


@pytest.mark.parametrize("ratios", [[], [True], [float("nan")], [1], [-0.1], ["bad"]])
def test_joint_history_rejects_invalid_profiles(ratios):
    from mosaickv.selection.profile_lifetime import allocate_profile_history

    with pytest.raises(ValueError, match="profile"):
        allocate_profile_history(ratios, 128, 69)


def test_joint_history_accepts_single_pass_iterable():
    from mosaickv.selection.profile_lifetime import allocate_profile_history

    assert allocate_profile_history(iter([0.1, 0.8]), 128, 69) == allocate_profile_history(
        [0.1, 0.8], 128, 69
    )


@pytest.mark.parametrize("seed", range(100))
def test_profiled_capacity_never_exceeds_cap_or_loses_mandatory(seed):
    rng = random.Random(seed)
    lengths = [rng.randint(1, 100) for _ in range(rng.randint(1, 8))]
    history, extra = rng.randint(0, 40), rng.randint(0, 20000)
    layers = len(lengths)
    minimum = layers * (1 + history) * 16 + (1 + history) * 16
    cap = minimum + extra
    ratios = tuple(i / (layers + 1) for i in range(layers))
    result = allocate_profile_lifetime(
        ratios, lengths, [1] * layers, [16] * layers, cap_bytes=cap, generated_capacity=history
    )
    assert result == allocate_profile_lifetime(
        ratios, lengths, [1] * layers, [16] * layers, cap_bytes=cap, generated_capacity=history
    )
    assert all(1 <= p <= length for p, length in zip(result.prompt_positions, lengths, strict=True))
    assert result.charged_ceiling_bytes <= cap
    assert result.prompt_bytes + result.history_bytes == result.resident_ceiling_bytes
    assert result.charged_ceiling_bytes + result.capacity_shortfall_bytes == cap
    assert not result.largest_single_increment_feasible


def test_profile_shape_is_preserved_instead_of_uniform_layer_capacity():
    result = allocate_profile_lifetime(
        [0.1, 0.8], [1000, 1000], [10, 10], [16, 16], cap_bytes=12000, generated_capacity=32
    )
    assert result.prompt_positions[0] > 3 * result.prompt_positions[1]
    assert result.growth_scratch_bytes == max((p + 32) * 16 for p in result.prompt_positions)


def test_impossible_mandatory_geometry_fails_closed():
    with pytest.raises(ValueError, match="cannot fit"):
        allocate_profile_lifetime(
            [0.5, 0.5], [10, 10], [5, 5], [16, 16], cap_bytes=10, generated_capacity=32
        )
    with pytest.raises(ValueError, match="invalid"):
        allocate_profile_lifetime(
            [float("nan")], [10], [1], [16], cap_bytes=1000, generated_capacity=0
        )
