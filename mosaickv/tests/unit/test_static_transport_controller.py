import evicastkv_static_transport_construction as control
import pytest


def test_method_agnostic_offer_keeps_all_original_baseline_controls():
    records = {
        "mosaickv_exact": {"graph": {}},
        "prefixkv_reimpl": {"graph": {}, "graph-original": {}},
    }
    variants = control.variants(records)
    assert len(variants) == 3 and len(control.ARMS) == 6
    assert all(record["static_transport_patches"] for _, _, record in variants)
    assert all(
        "static_transport_patches" not in record
        for group in records.values()
        for record in group.values()
    )


@pytest.mark.parametrize("method,count", [("mosaickv_exact", 64), ("prefixkv_reimpl", 0)])
def test_actual_eligibility_and_zero_retained_temporaries_are_enforced(monkeypatch, method, count):
    proof = dict(
        enabled=True,
        committed=True,
        constructor_patches=count,
        dynamic_transport_nonzero_calls_avoided=count,
        count_barrier_CPU_bytes=8 * count,
        constructor_temporary_objects_alive=0,
        canonical_packets_have_no_temporary_patches=True,
    )
    configured = []

    def execute(*args, **kwargs):
        configured.append(args[-1])
        return dict(static_transport_construction=proof)

    monkeypatch.setattr(control.previous, "execute", execute)
    record = dict(static_transport_patches=True)
    arguments = (None, None, None, None, None, method, record)
    control.execute(*arguments)
    assert configured[-1]["reuse_patch_dispatch"]
    assert "reuse_patch_dispatch" not in record
    proof["constructor_temporary_objects_alive"] = 1
    with pytest.raises(RuntimeError, match="retirement"):
        control.execute(*arguments)


def test_three_repetitions_never_establish_formal_latency_acceptance():
    rows = [
        dict(
            method=m,
            arm=a,
            trial=t,
            warmup=t < 0,
            measurement=dict(
                total_wall_seconds=0.9 if m == "mosaickv_exact" else 1.0,
                logits_bitwise=True,
                no_FullKV_fallback=True,
            ),
        )
        for m, a in control.ARMS
        for t in range(-5, 3)
    ]
    result = control.summary(rows)
    assert result["four_percent_POINT_only"]
    assert not result["formal_latency_acceptance"]
    assert len(result["ratios_vs_original_and_optimized_PrefixKV"]) == 4
    with pytest.raises(ValueError):
        control.summary(rows[:-1])
