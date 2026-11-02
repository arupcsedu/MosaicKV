from copy import deepcopy

from evicastkv_row_geometry import select_geometry


def fixtures():
    records = {
        i: [
            dict(
                geometry=list(geometry),
                status="completed",
                layers_checked=32,
                generation_128_logits_bitwise=True,
                max_spills=0,
            )
            for geometry in ((16, 4), (32, 4), (64, 4), (32, 2))
        ]
        for i in (0, 1)
    }
    trials = {
        i: [
            dict(
                geometry=f"r{rows}_h{heads}",
                layer=layer,
                trial=t,
                cuda_stream_ms=1.0 if rows == 16 else (0.8 if rows == 32 else 0.9),
            )
            for rows, heads in ((16, 4), (32, 4), (64, 4), (32, 2))
            for layer in range(32)
            for t in range(20)
        ]
        for i in (0, 1)
    }
    return records, trials


def test_one_common_geometry_uses_paired_trials_and_deterministic_ties():
    records, trials = fixtures()
    selected, summaries = select_geometry(records, trials, 3)
    assert selected == [32, 2]
    assert len(summaries) == 6
    assert all(r["familywise_upper_ratio"] <= 0.95 for r in summaries)
    assert select_geometry(deepcopy(records), deepcopy(trials), 3)[0] == selected


def test_bit_change_or_spill_increase_rejects_geometry_on_both_inputs():
    records, trials = fixtures()
    for record in records[0][1:]:
        record["generation_128_logits_bitwise"] = False
    assert select_geometry(records, trials, 3)[0] is None
    records, trials = fixtures()
    for record in records[1][1:]:
        record["max_spills"] = 1
    assert select_geometry(records, trials, 3)[0] is None


def test_less_than_registered_gain_stops_without_timing_promotion():
    records, trials = fixtures()
    for row in trials[1]:
        if row["geometry"] != "r16_h4":
            row["cuda_stream_ms"] = 0.98
    assert select_geometry(records, trials, 3)[0] is None
