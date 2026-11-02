from evicastkv_text_verification import profile_passes


def decision():
    return dict(
        complete=True,
        failures=[],
        summaries=[
            dict(
                input_index=i,
                method="mosaickv_exact",
                phases={"text_source.verification_equal": dict(inclusive_wall_seconds=0.21)},
            )
            for i in (0, 1)
        ],
    )


def test_admit_only_complete_two_input_measured_dominant_phase():
    assert profile_passes(decision())
    for key, value in [("complete", False), ("failures", ["missing"]), ("summaries", [])]:
        d = decision()
        d[key] = value
        assert not profile_passes(d)
    d = decision()
    d["summaries"][0]["phases"]["text_source.verification_equal"]["inclusive_wall_seconds"] = 0.001
    assert not profile_passes(d)
