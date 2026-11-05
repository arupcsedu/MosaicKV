from types import SimpleNamespace

import evicastkv_optimized_runtime_profile as controller
import pytest

from mosaickv.evaluation.optimized_runtime_profile import collect


def test_policies_shared_and_records_immutable():
    original = dict(prompt_graph_reader=True)
    native = dict(original, lt_source_reader=True, lt_fit_policy="full")
    assert controller.optimized_record(original)["graph_accounting"] == "event_sum"
    assert controller.optimized_record(native)["lt_fit_policy"] == "direct"
    assert native["lt_fit_policy"] == "full" and "graph_accounting" not in original


@pytest.mark.parametrize("fault", [None, "steps", "parity", "policy", "enumeration"])
def test_scopes_and_failure_controls(tmp_path, fault):
    adapter = SimpleNamespace(prepare_inputs=lambda: None, decode_one_token=lambda: None)
    original = adapter.prepare_inputs, adapter.decode_one_token

    def call():
        adapter.prepare_inputs()
        for _ in range(126 if fault == "steps" else 127):
            adapter.decode_one_token()
        return dict(
            logits_bitwise=fault != "parity",
            no_FullKV_fallback=True,
            graph_cache_accounting=dict(
                policy="original" if fault == "policy" else "event_sum",
                final_independent_tensor_enumeration_preserved=fault != "enumeration",
            ),
        )

    if fault:
        with pytest.raises(RuntimeError, match="trajectory"):
            collect(adapter, call, tmp_path)
    else:
        report = collect(adapter, call, tmp_path)
        assert set(report["phases"]) == {"construction", "decode"}
        assert report["decode_updates"] == 126
        assert report["initial_decode_capture_excluded"]
        assert report["instrumentation_NOT_request_latency_acceptance"]
    assert original == (adapter.prepare_inputs, adapter.decode_one_token)
