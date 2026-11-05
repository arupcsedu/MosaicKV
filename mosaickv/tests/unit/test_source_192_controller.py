import evicastkv_source_192_reader as control
import pytest


def test_failed_zero_correction_operator_cannot_release_runtime():
    with pytest.raises(ValueError, match="corrected operator"):
        control.check_operator(dict(complete=True, numerically_compatible=False, records=[]))


def test_both_geometries_and_real_counted_corrections_are_required():
    result = dict(
        complete=True,
        numerically_compatible=True,
        all_operator_medians_faster=True,
        records=[
            dict(
                source_length=n,
                bitwise_after_actual_counted_correction=True,
                lossless_corrections_explicitly_registered=True,
                actual_added_numerical_correction_bytes=100,
                different_incoming_words=[0, 0],
            )
            for n in (600, 604)
        ],
    )
    control.check_operator(result)
    result["records"][1]["different_incoming_words"] = [1, 0]
    with pytest.raises(ValueError, match="corrected operator"):
        control.check_operator(result)


def test_original_and_candidate_geometry_are_explicit_with_common_baseline_offer(monkeypatch):
    monkeypatch.setattr(
        control.prior.prior,
        "POLICIES",
        dict(segment=True, rotary=True, copy=False, frozen=True, overlap=True),
    )
    monkeypatch.setattr(control.prior, "NORM", True)
    monkeypatch.setattr(control, "REUSE", False)
    records = dict(
        mosaickv_exact={"graph-lt": {}}, prefixkv_reimpl={"graph": {}, "graph-original": {}}
    )
    configs = control.configurations(records)
    assert configs["mosaickv_exact", "joint-previous"]["source_tile_rows"] == 128
    for key, record in configs.items():
        assert record["source_tile_rows"] == (
            128 if key == ("mosaickv_exact", "joint-previous") else 192
        )
        assert record["input_norm_segment"] and record["post_attention_segment"]
        assert record["native_history_reader"] is False
