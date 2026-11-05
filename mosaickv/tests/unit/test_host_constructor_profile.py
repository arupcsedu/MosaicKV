from types import SimpleNamespace

import pytest

from mosaickv.evaluation.host_constructor_profile import collect


def test_constructor_boundary_excludes_decode_and_restores_methods(tmp_path):
    adapter = SimpleNamespace(prepare_inputs=lambda: None, decode_one_token=lambda: None)
    old = adapter.prepare_inputs, adapter.decode_one_token

    def call():
        adapter.prepare_inputs()
        for _ in range(127):
            adapter.decode_one_token()
        return dict(logits_bitwise=True, no_FullKV_fallback=True)

    row = collect(adapter, call, tmp_path)
    assert row["decode_excluded"] and row["not_pure_compression_time"]
    assert row["parent_logits_bitwise"] and row["instrumentation_NOT_request_latency_acceptance"]
    assert old == (adapter.prepare_inputs, adapter.decode_one_token)


@pytest.mark.parametrize("fault", ["missing_prepare", "missing_step", "parity"])
def test_incomplete_trajectory_fails_closed(tmp_path, fault):
    adapter = SimpleNamespace(prepare_inputs=lambda: None, decode_one_token=lambda: None)

    def call():
        if fault != "missing_prepare":
            adapter.prepare_inputs()
        for _ in range(126 if fault == "missing_step" else 127):
            adapter.decode_one_token()
        return dict(logits_bitwise=fault != "parity", no_FullKV_fallback=True)

    with pytest.raises(RuntimeError, match="trajectory"):
        collect(adapter, call, tmp_path)
