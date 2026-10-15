"""CPU random-weight HF tests, never pretrained quality/performance evidence."""

import json
import math
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_hf_tiny_models import (  # noqa: E402
    _baseline_runtime_config,
    _llava,
    _runtime_request,
    _TextOnlyProcessor,
)

from mosaickv.backends.hf_runtime import (  # noqa: E402
    HuggingFaceMosaicKVModel,
    compare_runtime_retention_one,
)
from mosaickv.config import (  # noqa: E402
    OutputPreservingConfig,
    RunConfig,
    SelectionConfig,
    canonical_config,
)
from mosaickv.evaluation.shared_byte_contract import validate_configuration  # noqa: E402
from mosaickv.types import BudgetUnit, ForecastMode, MosaicKVMethod  # noqa: E402


class LongerProcessor(_TextOnlyProcessor):
    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, *range(3, 18)]]),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }


def candidate_config(*, swaps=2, retention=0.5, draft_steps=0):
    base = _baseline_runtime_config(MosaicKVMethod.PROMPT_ATTENTION_TOPK, retention)
    return replace(
        base,
        method=MosaicKVMethod.MOSAICKV_EXACT,
        cache=replace(
            base.cache,
            budget_value=1024,
            budget_unit=BudgetUnit.BYTES,
            physical_budget_block_size=1,
            physical_budget_mandatory_policy="first_and_last",
            packing_mode="per_head_reference",
        ),
        generation=replace(base.generation, max_new_tokens=16),
        forecasting=replace(
            base.forecasting,
            enabled=True,
            mode=(ForecastMode.HYBRID if draft_steps else ForecastMode.PROMPT_WINDOW),
            draft_steps=draft_steps,
        ),
        selection=SelectionConfig(),
        output_preserving=OutputPreservingConfig(
            max_swaps_per_head=swaps,
            shortlist_size=4,
            draft_validation=bool(draft_steps),
        ),
    )


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def run(adapter, config, tmp_path, name):
    result = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path).generate(
        _runtime_request(name)
    )
    trace = json.loads(next((tmp_path / name).glob("*.json")).read_text())
    return result, trace


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_anchor_matches_topk_and_preserves_gqa_queries(tmp_path, kv_heads):
    torch.manual_seed(32)
    adapter = _llava(kv_heads=kv_heads)
    adapter.processor = LongerProcessor()
    config = candidate_config(swaps=0)
    baseline = replace(
        config,
        method=MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        forecasting=replace(config.forecasting, enabled=False),
        selection=SelectionConfig(enabled=False),
        output_preserving=None,
    )
    _, baseline_trace = run(adapter, baseline, tmp_path, "baseline")
    result, trace = run(adapter, config, tmp_path, "anchor")
    assert trace["generated_token_ids"] == baseline_trace["generated_token_ids"]
    assert trace["packed_slots"] == baseline_trace["packed_slots"]
    for boundary in ("initial", "final", "peak"):
        key = f"{boundary}_active_kv_storage_bytes"
        assert trace["cache_lifecycle"][key] == baseline_trace["cache_lifecycle"][key]
    assert trace["selected_blocks"] == trace["anchor_selected_blocks"]
    audit = trace["output_reconstruction"]
    assert all(
        r["query_heads"] == 2 // kv_heads and r["objective_query_rows"] == 4 // kv_heads
        for r in audit["routes"]
    )
    assert audit["accepted_swaps"] == 0
    assert result.metrics.generated_tokens == 16
    assert trace["prototypes"] == [] and trace["repair_events"] == []


def test_candidate_identity_fullkv_and_repeated_request_isolation(tmp_path):
    torch.manual_seed(19)
    adapter = _llava(kv_heads=1)
    adapter.processor = LongerProcessor()
    config = candidate_config()
    _, a = run(adapter, config, tmp_path, "candidate-a")
    identity = replace(config, cache=replace(config.cache, retention_ratio=1.0, budget_value=2048))
    parity = compare_runtime_retention_one(
        adapter,
        adapter.prepare_inputs(_runtime_request("parity").messages),
        identity,
    )
    assert parity.token_agreement == 1.0 and parity.maximum_logit_difference <= 1e-6
    _, untouched = run(
        adapter,
        replace(
            identity,
            method=MosaicKVMethod.FULL_KV,
            forecasting=replace(identity.forecasting, enabled=False),
            selection=SelectionConfig(enabled=False),
            output_preserving=None,
        ),
        tmp_path,
        "full",
    )
    _, full = run(adapter, identity, tmp_path, "identity")
    _, again = run(adapter, config, tmp_path, "candidate-a-again")
    assert full["generated_token_ids"] == untouched["generated_token_ids"]
    assert a["generated_token_ids"] == again["generated_token_ids"]
    assert a["selected_blocks"] == again["selected_blocks"]
    assert a["active_kv_bytes"] == 1024
    assert a["output_reconstruction"]["final_error"] <= a["output_reconstruction"]["initial_error"]
    assert a["graph_edges"] == [] and a["prototypes"] == []
    assert a["effective_method"] == "mosaickv_exact__output_preserving_v1"
    assert all(
        math.isfinite(v) and v >= 0
        for v in a["timing_breakdown"].values()
        if isinstance(v, (int, float))
    )


def test_draft_gated_candidate_is_isolated_deterministic_and_audited(tmp_path):
    torch.manual_seed(57)
    adapter = _llava(kv_heads=1)
    adapter.processor = LongerProcessor()
    config = candidate_config(draft_steps=2)
    _, first = run(adapter, config, tmp_path, "draft-first")
    _, second = run(adapter, config, tmp_path, "draft-second")
    assert first["generated_token_ids"] == second["generated_token_ids"]
    assert first["selected_blocks"] == second["selected_blocks"]
    assert first["active_kv_bytes"] == 1024
    assert first["effective_method"].endswith("__draft_gate")
    assert first["forecast_statistics"]["draft_steps"] == 2
    assert first["forecast_statistics"]["draft_cache_isolated"] is True
    audit = first["output_reconstruction"]
    assert audit["draft_validation"] is True
    assert audit["final_error"] <= audit["initial_error"]
    assert audit["draft_final_error"] <= audit["draft_initial_error"]
    assert first["timing_breakdown"]["forecast"] > 0


def test_new_config_roundtrip_legacy_hash_payload_and_frozen_lane_rejection():
    config = candidate_config()
    assert RunConfig.from_mapping(asdict(config)) == config
    legacy = _baseline_runtime_config(MosaicKVMethod.PROMPT_ATTENTION_TOPK, 0.5)
    assert "output_preserving" not in canonical_config(legacy)
    with pytest.raises(ValueError, match="not the frozen lane"):
        validate_configuration({}, "vl_cache", "coco_caption", config)


def test_failed_selector_writes_explicit_variant_trace(tmp_path, monkeypatch):
    import mosaickv.backends.output_preserving as bridge

    def reject(*_args, **_kwargs):
        raise ValueError("insufficient query mass fixture")

    monkeypatch.setattr(bridge, "select_output_preserving", reject)
    adapter = _llava()
    adapter.processor = LongerProcessor()
    runtime = HuggingFaceMosaicKVModel(adapter, candidate_config(), trace_directory=tmp_path)
    with pytest.raises(ValueError, match="insufficient query mass"):
        runtime.generate(_runtime_request("failed"))
    trace = json.loads(next((tmp_path / "failed").glob("*.json")).read_text())
    assert trace["status"] == "failed"
    assert trace["effective_method"] == "mosaickv_exact__output_preserving_v1"
    assert trace["output_preserving_config"]["query_evidence"] == "individual"
    assert trace["end_to_end_time"] >= 0
