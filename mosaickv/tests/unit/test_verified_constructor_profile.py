import evicastkv_verified_constructor_profile as control


def test_attribution_is_deterministic_and_does_not_claim_latency():
    stats = {
        ("/repo/verified_constructor.py", 85, "capture"): (1, 64, 0.2, 0.5, {}),
        ("~", 0, "<built-in method equal of type object>"): (1, 64, 0.1, 0.1, {}),
        ("other.py", 2, "other"): (1, 1, 0.3, 0.3, {}),
    }
    value = control.attribution(stats, 2)
    assert value["top_exclusive"][0]["function"] == "other"
    assert value["constructor"][0]["calls"] == 64
    assert value["torch_equal"][0]["calls"] == 64
    assert value["instrumented_NOT_latency_acceptance"]
    assert value["background_thread_CPU_not_attributed"]


def test_runtime_cannot_change_and_only_consumed_arms_are_profiled():
    assert len(control.APPROVED) == 4
    assert all("/src/" not in path for path in control.APPROVED)
    expected = (*control.base.ARMS[:3], control.base.ARMS[3])
    assert expected == control.ARMS
    assert len(control.ARMS) == 4
