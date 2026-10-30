import copy

import pytest

from mosaickv.evaluation.precision_outcome import FIELDS, check


def fixture():
    row = dict(
        reference=dict(storage=dict.fromkeys(FIELDS, 100)),
        controls={
            "3": dict(
                candidate=dict(storage=dict.fromkeys(FIELDS, 80)),
                clone_repack_bitwise=True,
                lossless_exact_codec_128_bitwise=True,
                numerical_gate=True,
            )
        },
        trials=[
            dict(trial=i, warmup=False, variant=v, decode_wall_seconds=t)
            for i in range(20)
            for v, t in (("reference", 10.0), ("3", 9.0))
        ],
    )
    corrections = dict(
        layers=[
            dict(
                layer=2,
                exception_bytes=6,
                saved_text_and_map_bytes=10,
                corrected_component_bytes=80,
                raw_text_equivalent_bytes=84,
                fraction=0.005,
                components=[dict(squared_error_before=1, squared_error_after=0.5)],
            )
        ]
    )
    return (
        row,
        corrections,
        dict.fromkeys(FIELDS, 85),
        {cap: {"reference": 0.8, "3": 0.8} for cap in (32, 64, 128)},
    )


def test_checks_all_storage_boundaries_and_preserves_inputs():
    args = fixture()
    before = copy.deepcopy(args)
    assert check(*args)["diagnostic_all_checks_passed"]
    assert args == before


@pytest.mark.parametrize("failure", ["funding", "peak", "late_quality", "latency", "numerical"])
def test_never_hide_long_answer_peak_funding_or_latency_failures(failure):
    row, corrections, ceiling, quality = fixture()
    if failure == "funding":
        corrections["layers"][0]["exception_bytes"] = 11
    elif failure == "peak":
        row["controls"]["3"]["candidate"]["storage"][FIELDS[-1]] = 86
    elif failure == "late_quality":
        quality[128]["3"] = 0.79
    elif failure == "numerical":
        row["controls"]["3"]["numerical_gate"] = False
    else:
        row["trials"][1]["decode_wall_seconds"] = 11
        for trial in row["trials"]:
            if trial["variant"] == "3":
                trial["decode_wall_seconds"] = 11
    assert not check(row, corrections, ceiling, quality)["diagnostic_all_checks_passed"]


def test_incomplete_or_nonfinite_timings_fail_closed():
    args = fixture()
    args[0]["trials"].pop()
    with pytest.raises(ValueError, match="20 randomized paired"):
        check(*args)
    args = fixture()
    args[0]["trials"][0]["decode_wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite positive"):
        check(*args)
