import prepared_address_baselines as baseline


def test_private_prefill_policy_identical_and_address_transport_common_to_every_baseline():
    for method in baseline.METHODS:
        a, b = (baseline.policy(method + ":" + flag) for flag in ("reference", "private"))
        assert a["private_prefill_scores"] is b["private_prefill_scores"] is True
        assert a["prepared_addresses"] is False and b["prepared_addresses"] is True
