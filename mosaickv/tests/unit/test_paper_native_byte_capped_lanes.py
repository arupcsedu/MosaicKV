from __future__ import annotations

from pathlib import Path

import yaml


def test_byte_capped_protocol_is_execution_disabled_and_fail_closed() -> None:
    path = (
        Path(__file__).parents[2]
        / "configs/experiments/paper_native_byte_capped_lanes_v1.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["classification"] == "unified_comparison_physical_byte_curve"
    assert payload["execution_authorized"] is False
    controls = payload["shared_controls"]
    assert controls["budget_unit"] == "bytes"
    assert controls["require_per_sample_exact_active_byte_match"] is True
    assert controls["fail_if_above_configured_byte_cap"] is True
    assert controls["require_same_prompts_media_processor_chat_template"] is True

    vl_cache = payload["lanes"]["vl_cache_byte_capped"]
    assert vl_cache["methods"] == ["full_kv", "mosaickv_exact", "vl_cache_reimpl"]
    assert vl_cache["budgets_mib"] == [10, 20, 40, 80, 120]
    assert vl_cache["retention_ratio_upper_bound"] == 0.99

    prefixkv = payload["lanes"]["prefixkv_byte_capped"]
    assert prefixkv["methods"] == ["full_kv", "mosaickv_exact", "prefixkv_reimpl"]
    assert prefixkv["prefixkv"]["rescale_offline_profile_to_byte_budget"] is True
    assert prefixkv["retention_ratio_upper_bound"] == 0.99
