"""Random-weight CPU architecture tests, NOT real-checkpoint answer quality."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_hf_tiny_models import (  # noqa: E402
    _baseline_runtime_config,
    _llava,
    _runtime_config,
    _runtime_request,
    _VLCacheProcessor,
)

from mosaickv.adapters.huggingface.bounded_history_cache import (  # noqa: E402
    install_bounded_history,
)
from mosaickv.adapters.huggingface.types import DecodeState  # noqa: E402
from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel  # noqa: E402
from mosaickv.config import (  # noqa: E402
    BoundedDecodeConfig,
    CacheConfig,
    ConfigurationError,
    GenerationConfig,
    LookMConfig,
    PrefixKVConfig,
    PrototypeConfig,
    PublishedKVConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    canonical_config,
    config_sha256,
    load_config,
)
from mosaickv.measurements.cache_lifecycle import CacheLifecycle  # noqa: E402
from mosaickv.types import BudgetUnit, MosaicKVMethod, RepairPolicy  # noqa: E402


@pytest.mark.parametrize(
    "method",
    [
        MosaicKVMethod.RANDOM_KV,
        MosaicKVMethod.UNIFORM_KV,
        MosaicKVMethod.VALUE_TOPK,
        MosaicKVMethod.LOOKM_REIMPL,
        MosaicKVMethod.VL_CACHE_REIMPL,
        MosaicKVMethod.H2O_REIMPL,
        MosaicKVMethod.STREAMINGLLM_REIMPL,
        MosaicKVMethod.PYRAMIDKV_REIMPL,
        MosaicKVMethod.SNAPKV_REIMPL,
        MosaicKVMethod.ROCO_REIMPL,
    ],
)
def test_baselines_share_exact_lifetime_storage_without_native_relabeling(tmp_path, method):
    cfg = _config()
    kwargs = {"method": method}
    if method is MosaicKVMethod.LOOKM_REIMPL:
        kwargs["lookm"] = LookMConfig(
            enabled=True, recent_ratio=0.25, important_ratio=cfg.cache.retention_ratio - 0.25
        )
    elif method is MosaicKVMethod.VL_CACHE_REIMPL:
        from mosaickv.config import VLCacheConfig

        kwargs["vl_cache"] = VLCacheConfig(enabled=True)
    elif method.is_published_eviction_reimplementation:
        kwargs["published_kv"] = PublishedKVConfig(
            enabled=True, sink_tokens=1, observation_window=1
        )
    cfg = replace(cfg, **kwargs)
    adapter = _llava(kv_heads=2 if method is MosaicKVMethod.LOOKM_REIMPL else 1)
    expected_bytes = 1024 if method is MosaicKVMethod.LOOKM_REIMPL else 512
    cfg = replace(cfg, cache=replace(cfg.cache, budget_value=expected_bytes))
    if method is MosaicKVMethod.VL_CACHE_REIMPL:
        adapter.processor = _VLCacheProcessor()
    result = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(
        _runtime_request(method.value)
    )
    assert result.metrics.active_kv_bytes == expected_bytes
    assert result.metrics.initial_active_kv_storage_bytes == expected_bytes
    assert result.metrics.final_active_kv_storage_bytes == expected_bytes
    assert result.metrics.peak_active_kv_storage_bytes == expected_bytes
    assert "__bounded_generated_history_v1" in result.effective_method
    if method.is_published_reimplementation:
        assert result.effective_method.endswith("_reimpl")


def test_prefixkv_profile_remains_layer_ragged_without_padding_or_double_eviction(tmp_path):
    from mosaickv.baselines.prefixkv import PrefixKVOfflineProfile

    cfg = _config()
    profile = PrefixKVOfflineProfile(
        model_id=cfg.model.id,
        model_revision=cfg.model.revision,
        target_retention_ratio=0.625,
        layer_forget_ratios=(0.0, 0.75),
        calibration_dataset_id="synthetic",
        calibration_dataset_revision="b" * 40,
        calibration_split="train",
        calibration_sample_ids=("cal-only",),
        calibration_seed=0,
        start_size=0,
        protect_size=1,
    )
    path = profile.write(tmp_path / "profile.json")
    cfg = replace(
        cfg,
        method=MosaicKVMethod.PREFIXKV_REIMPL,
        prefixkv=PrefixKVConfig(
            enabled=True,
            profile_path=str(path),
            start_size=0,
            hf_cache_transport="ragged_llama_eager_v1",
            rescale_offline_profile_to_byte_budget=True,
            profile_reference_retention_ratio=0.625,
            decode_policy="append_only",
        ),
    )
    result = HuggingFaceMosaicKVModel(_llava(), cfg, trace_directory=tmp_path / "traces").generate(
        _runtime_request("bounded-prefix")
    )
    trace = json.loads(next((tmp_path / "traces").rglob("*.json")).read_text())
    lengths = trace["bounded_decode"]["protected_prompt_positions_per_layer"]
    assert len(set(lengths)) == 2
    assert sum(lengths) == 4
    assert trace["bounded_decode"]["cross_layer_padding_bytes"] == 0
    assert trace["prefixkv_decode_events"] == []
    assert trace["bounded_decode"]["evicted_generated_positions_per_head"] == 17
    assert result.metrics.peak_active_kv_storage_bytes == 512
    assert result.effective_method.endswith("__bounded_generated_history_v1_reimpl")


def test_vl_cache_adaptive_layer_lengths_use_no_padding_and_keep_exact_lifetime_cap(
    tmp_path, monkeypatch
):
    import mosaickv.baselines.vl_cache as vl
    from mosaickv.config import VLCacheConfig

    monkeypatch.setattr(
        vl, "paper_layer_retention_ratios", lambda *_a, **_k: ((0.75, 0.25), (0.75, 0.25))
    )
    adapter = _llava()
    adapter.processor = _VLCacheProcessor()
    cfg = replace(
        _config(), method=MosaicKVMethod.VL_CACHE_REIMPL, vl_cache=VLCacheConfig(enabled=True)
    )
    result = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(
        _runtime_request("vl-ragged-regression")
    )
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["bounded_decode"]["protected_prompt_positions_per_layer"] == [3, 1]
    assert trace["bounded_decode"]["cross_layer_padding_bytes"] == 0
    assert trace["packed_cache_all_slots_valid"] is True
    assert result.metrics.initial_active_kv_storage_bytes == 512
    assert result.metrics.final_active_kv_storage_bytes == 512
    assert result.metrics.peak_active_kv_storage_bytes == 512


def test_evidence_allocation_is_a_labeled_exact_byte_runtime_variant(tmp_path):
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(
            generated_capacity=2,
            sink_tokens=1,
            history_fraction=0.2,
            layer_allocation="output_impact_v1",
        ),
    )
    result = HuggingFaceMosaicKVModel(_llava(), cfg, trace_directory=tmp_path).generate(
        _runtime_request("layer-impact")
    )
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["evidence_layer_allocation"]["padding_bytes"] == 0
    assert trace["evidence_layer_allocation"]["benchmark_labels_accessed"] is False
    assert sum(trace["evidence_layer_allocation"]["layer_capacities"]) == 4
    assert trace["hard_prompt_capacity"] is None
    assert (
        trace["selected_layer_prompt_capacities"]
        == trace["evidence_layer_allocation"]["layer_capacities"]
    )
    assert trace["byte_budget_materialization"]["dense_quantization_underfill_bytes"] == 0
    assert result.metrics.peak_active_kv_storage_bytes == 512
    assert "output_impact_layers" in result.effective_method
    assert result.effective_method.endswith("bounded_generated_history_v2")


@pytest.mark.parametrize("policy", ["output_impact_guarded_v1", "direct_error_exchange_v1"])
def test_guarded_allocation_reports_final_errors_and_keeps_lifetime_cap(tmp_path, policy):
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(2, 1, 0.2, policy),
    )
    result = HuggingFaceMosaicKVModel(_llava(), cfg, trace_directory=tmp_path).generate(
        _runtime_request("guarded-layer-impact")
    )
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    allocation = trace["evidence_layer_allocation"]
    guard = allocation["reconstruction_guard"]
    assert allocation["padding_bytes"] == 0
    assert allocation["benchmark_labels_accessed"] is False
    assert len(guard["final_mse_by_layer_head"]) == 2
    assert guard["used_fullkv_fallback"] is False
    assert guard["host_wall_seconds_including_device_transfers"] >= 0
    for before, after in zip(
        guard["anchor_mse_by_layer_head"], guard["final_mse_by_layer_head"], strict=True
    ):
        assert all(
            b <= a + guard["absolute_tolerance"] + a * guard["relative_tolerance"]
            for a, b in zip(before, after, strict=True)
        )
    assert sum(allocation["layer_capacities"]) == 4
    assert result.metrics.initial_active_kv_storage_bytes == 512
    assert result.metrics.final_active_kv_storage_bytes == 512
    assert result.metrics.peak_active_kv_storage_bytes == 512
    if policy == "direct_error_exchange_v1":
        assert "__direct_error_exchange_layers" in result.effective_method
        assert guard["exchange_search"]["verified_pairs"] <= 64
        assert guard["exchange_search"]["committed_exchanges"] <= 16
    else:
        assert "__guarded_output_impact_layers" in result.effective_method


def test_one_token_teacher_forcing_ragged_cache_never_preallocates(tmp_path, monkeypatch):
    import mosaickv.selection.direct_error_exchange as exchange
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.selection.evidence_layer_allocation import EvidenceLayerPlan

    adapter = _llava()
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(2, 1, 0.2, "direct_error_exchange_v1"),
    )
    cfg = replace(cfg, cache=replace(cfg.cache, budget_value=384))
    # Synthetic transport regression: force feasible 2/4 layer lengths, not a
    # measured algorithm claim. Layer zero is deliberately not the longest.
    monkeypatch.setattr(
        exchange,
        "build_direct_error_layer_plan",
        lambda *_a, **_k: EvidenceLayerPlan(
            (((2, 3),), ((0, 1, 2, 3),)), (2, 4), 384, (0.0, 0.0), (1.0, 1.0), (1, 1)
        ),
    )

    def forbidden(*_a, **_k):
        pytest.fail("zero decode updates must never preallocate")

    monkeypatch.setattr(adapter, "preallocate_decode_cache", forbidden)
    request = _runtime_request("one-token-ragged")
    request = EvaluationRequest(
        request.run_id,
        request.sample_id,
        request.task,
        request.messages,
        {"_mosaickv_teacher_forced_token_ids": (2,)},
    )
    result = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(request)
    assert result.metrics.generated_tokens == 1
    assert result.metrics.initial_active_kv_storage_bytes == 384
    assert result.metrics.final_active_kv_storage_bytes == 384
    assert result.metrics.peak_active_kv_storage_bytes == 384
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["execution_phase"] == "teacher_forced"
    assert trace["decode_runtime_policy"]["planned_decode_updates"] == 0
    assert trace["bounded_decode"]["status"] == "no_decode_updates"
    assert trace["teacher_forced_evaluation"]["evaluated_tokens"] == 1


def test_rejected_reallocation_reuses_packed_anchor_without_gathering(monkeypatch):
    from types import SimpleNamespace

    import mosaickv.adapters.huggingface.layer_local_cache as gathering
    import mosaickv.selection.evidence_layer_allocation as selection
    from mosaickv.backends.hf_runtime import _reallocate_evidence_layers

    anchor = SimpleNamespace(
        prompt_capacity=2,
        slot_records=(),
        snapshot=SimpleNamespace(
            layers=(SimpleNamespace(key=torch.zeros(1, 1, 2, 2), value=torch.zeros(1, 1, 2, 2)),)
        ),
    )
    plan = SimpleNamespace(
        prompt_bytes=32,
        reconstruction={"accepted": False},
        trace=lambda: {"reconstruction_guard": {"accepted": False}},
    )
    monkeypatch.setattr(selection, "build_evidence_layer_plan", lambda *_a, **_k: plan)

    def forbidden_gather(*_a, **_k):
        pytest.fail("rollback must reuse the existing same-byte anchor")

    monkeypatch.setattr(gathering, "gather_layer_local_snapshot", forbidden_gather)
    returned, trace = _reallocate_evidence_layers(
        None, None, None, anchor, {}, 32, guard_reconstruction=True
    )
    assert returned is anchor
    assert trace["reconstruction_guard"]["accepted"] is False


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(204)
    yield
    torch.set_num_threads(previous)


def test_label_blind_logit_diagnostic_reuses_prefill_and_isolates_states(tmp_path, monkeypatch):
    from mosaickv.evaluation.direct_error_diagnostic import DirectErrorLogitDiagnostic

    adapter = _llava()
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(2, 1, 0.2, "direct_error_exchange_v1"),
    )
    prefill = adapter.prefill
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return prefill(*args, **kwargs)

    monkeypatch.setattr(adapter, "prefill", counted)
    runtime = DirectErrorLogitDiagnostic(adapter, cfg, trace_directory=tmp_path, horizon=2)
    request = _runtime_request("isolated-diagnostic")
    record = runtime.diagnose(request)
    assert len(calls) == 1
    assert record["measured_source_cache_unchanged"]
    assert record["logical_positions_unchanged"]
    assert record["benchmark_labels_accessed"] is False
    assert record["selection_modified_by_probe"] is False
    assert record["promotion_authorized"] is False
    assert len(record["steps"]) == 2
    for step in record["steps"]:
        for name in ("anchor", "candidate"):
            assert step[name]["active_bytes"] == 512
            assert step[name]["logit_kl"] >= -1e-6
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["status"] == "completed_evaluation_only_prefill_diagnostic"
    assert trace["not_a_benchmark_prediction_or_performance_row"] is True
    from mosaickv.evaluation.model import EvaluationRequest

    with pytest.raises(ValueError, match="prompt/media only"):
        runtime.diagnose(
            EvaluationRequest(
                request.run_id,
                request.sample_id,
                request.task,
                request.messages,
                {"target": "forbidden"},
            )
        )


@pytest.mark.parametrize("kind", ["ppl", "diagnostic"])
def test_recovery_worker_uses_metric_only_path_and_writes_manifest(tmp_path, monkeypatch, kind):
    import importlib
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    recovery = importlib.import_module("recover_bounded_history_v4")
    # Load aliases before monkeypatching the native loader; otherwise a later
    # import could retain a mocked loader beyond this test's fixture lifetime.
    importlib.import_module("mosaickv.hf_cli")
    import mosaickv.adapters.huggingface as adapters
    import mosaickv.evaluation.paper_native_lanes as native
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel

    adapter = _llava()
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(2, 1, 0.2, "direct_error_exchange_v1"),
    )
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    job = dict(kind=kind, sample_ids=["synthetic-consumed"], horizon=2)
    original = dict(
        index=13,
        task="synthetic_smoke",
        ledger="not-read",
        ledger_sha="c" * 64,
        config_sha="d" * 64,
        reference=str(reference),
        method="mosaickv_direct_error_v4",
    )
    monkeypatch.setattr(
        recovery, "validate", lambda *_: (dict(source_sha="a" * 40), job, {}, original, cfg)
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    # Do not leak the production worker's 12-thread setting into other CPU tests.
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "CPU mock A100; not GPU evidence")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(adapters, "load_hf_adapter", lambda *_a, **_k: adapter)
    monkeypatch.setattr(
        native,
        "load_paper_native_ledger",
        lambda *_a, **_k: SimpleNamespace(
            samples=[
                SimpleNamespace(
                    sample_id="synthetic-consumed",
                    prompt="tiny",
                    media=(),
                    teacher_forcing_target="synthetic target",
                )
            ]
        ),
    )
    monkeypatch.setattr(HuggingFaceMosaicKVModel, "_teacher_forced_target_ids", lambda *_: (2,))
    recovery.worker(tmp_path, "b" * 64, 0)
    folder = tmp_path / "workers/00"
    completed = json.loads((folder / "completed.json").read_text())
    assert completed["status"] == "passed"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["execution"]["generation_predictions_rerun"] is False
    assert manifest["execution"]["new_timing_comparison"] is False
    records = json.loads((folder / "raw.json").read_text())
    assert len(records) == 1 and records[0]["status"] == "completed"
    if kind == "ppl":
        assert records[0]["result"]["tokens"] == 1
    else:
        assert records[0]["result"]["benchmark_labels_accessed"] is False


def test_v5_runtime_uses_isolated_later_prefix_guard_and_preserves_byte_cap(tmp_path, monkeypatch):
    import mosaickv.backends.hf_runtime as runtime
    from mosaickv.adapters.huggingface.layer_local_cache import gather_layer_local_snapshot
    from mosaickv.backends.hf_allocation_guard import (
        validate_draft_guard_trace,
    )
    from mosaickv.config import DraftAllocationGuardConfig
    from mosaickv.evaluation.model import EvaluationRequest

    # Force one exact equal-byte exchange in this RANDOM-WEIGHT fixture so the
    # real transport/probe path is exercised even if the planner finds no move.
    def proposal(adapter, _source, original, anchor, _mass, _budget, **_kwargs):
        grouped = [[[] for _ in range(int(x.key.shape[1]))] for x in anchor.snapshot.layers]
        for r in anchor.slot_records:
            physical = dict(
                (logical, index)
                for index, logical in enumerate(_source.logical_positions.physical_to_logical)
            )
            grouped[r["layer"]][r["kv_head"]].append(physical[r["logical_position"]])
        chosen = grouped[0][0]
        if len(chosen) < 4:
            omitted = next(p for p in range(3) if p not in chosen)
            chosen[0] = omitted
        snapshot = gather_layer_local_snapshot(
            adapter, original, tuple(tuple(tuple(sorted(h)) for h in layer) for layer in grouped)
        )
        records = tuple(
            dict(
                layer=layer_index,
                kv_head=h,
                slot=s,
                physical_position=p,
                logical_position=_source.logical_positions.physical_to_logical[p],
                tier="exact",
            )
            for layer_index, heads in enumerate(grouped)
            for h, positions in enumerate(heads)
            for s, p in enumerate(sorted(positions))
        )
        packed = runtime.PackedRuntimeCache(
            snapshot, (), snapshot.active_sequence_length, True, (), (), records
        )
        return packed, dict(padding_bytes=0, benchmark_labels_accessed=False, fixture_only=True)

    monkeypatch.setattr(runtime, "_reallocate_evidence_layers", proposal)
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(
            2, 1, 0.2, "draft_logit_guard_v1", DraftAllocationGuardConfig(8, (1, 4, 8))
        ),
    )
    adapter = _llava()
    prefill_calls = []
    original_prefill = adapter.prefill

    def prefill(*args, **kwargs):
        prefill_calls.append(1)
        return original_prefill(*args, **kwargs)

    monkeypatch.setattr(adapter, "prefill", prefill)
    request = _runtime_request("V5-generation")
    result = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(request)
    generation = json.loads(next((tmp_path / request.run_id).glob("*.json")).read_text())
    guard = generation["draft_logit_allocation_guard"]
    assert len(prefill_calls) == 1 and guard["probe_updates"] == 24
    assert validate_draft_guard_trace(guard)
    assert guard["later_checkpoints_completed"] and len(guard["steps"]) == 8
    assert result.metrics.peak_active_kv_storage_bytes == 512
    assert "__draft_logit_guard_layers_v5" in result.effective_method
    assert generation["evidence_layer_allocation"]["role"].startswith("proposal_only")
    assert generation["timing_breakdown"]["logit_stability"] > 0
    # Teacher-forced labels change decode inputs, never the guard's draft prefix.
    teacher = EvaluationRequest(
        "V5-ppl",
        request.sample_id,
        request.task,
        request.messages,
        {"_mosaickv_teacher_forced_token_ids": (2,) * 20},
    )
    HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(teacher)
    forced = json.loads(next((tmp_path / teacher.run_id).glob("*.json")).read_text())[
        "draft_logit_allocation_guard"
    ]
    assert forced["steps"] == guard["steps"]
    assert forced["final_positions_sha256"] == guard["final_positions_sha256"]
    assert len(prefill_calls) == 2


def test_v6_single_exchange_uses_real_transport_and_no_label_input(tmp_path, monkeypatch):
    import mosaickv.selection.direct_error_exchange as planner
    from mosaickv.backends.hf_single_exchange_guard import validate_single_exchange_trace
    from mosaickv.config import DraftAllocationGuardConfig, DraftExchangeSearchConfig
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.selection.evidence_layer_allocation import EvidenceLayerPlan

    def singles(source, mass, records, *, prompt_budget_bytes, max_candidates):
        assert max_candidates == 2 and mass
        physical = {p: i for i, p in enumerate(source.logical_positions.physical_to_logical)}
        grouped = [[[] for _ in range(layer.kv_heads)] for layer in source.layers]
        for row in records:
            grouped[row["layer"]][row["kv_head"]].append(physical[row["logical_position"]])
        counts = [len(heads[0]) for heads in grouped]
        d, r = 0, 1
        removed, added = [], []
        for head in range(len(grouped[d])):
            removed.append(grouped[d][head].pop(0))
            added.append(
                next(
                    p for p in range(source.layers[r].sequence_length) if p not in grouped[r][head]
                )
            )
            grouped[r][head].append(added[-1])
        positions = tuple(tuple(tuple(sorted(head)) for head in layer) for layer in grouped)
        capacities = tuple(len(layer[0]) for layer in positions)
        costs = source.layers[d].byte_size // source.layers[d].sequence_length
        event = dict(
            donor_layer=d,
            receiver_layer=r,
            transferred_bytes=costs,
            initial_layer_capacities=counts,
            removed_positions=removed,
            added_positions=added,
        )
        return (
            EvidenceLayerPlan(
                positions,
                capacities,
                prompt_budget_bytes,
                (0.0,) * len(counts),
                (0.0,) * len(counts),
                (0,) * len(counts),
                {"single_exchange": event},
            ),
        )

    monkeypatch.setattr(planner, "build_single_exchange_layer_plans", singles)
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(
            2,
            1,
            0.2,
            "draft_single_exchange_v1",
            DraftAllocationGuardConfig(8, (1, 4, 8)),
            DraftExchangeSearchConfig(2, 1),
        ),
    )
    adapter = _llava()
    calls = []
    original = adapter.prefill

    def prefill(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "prefill", prefill)
    request = _runtime_request("V6-generation")
    result = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(request)
    generation = json.loads(next((tmp_path / request.run_id).glob("*.json")).read_text())
    guard = generation["draft_logit_allocation_guard"]
    assert validate_single_exchange_trace(guard)
    assert len(calls) == 1 and 0 < guard["probe_updates"] <= 24
    assert result.metrics.peak_active_kv_storage_bytes == 512
    assert "__draft_single_exchange_layers_v6" in result.effective_method
    assert "V6_single_exchange" in generation["selection_provenance"]
    assert generation["timing_breakdown"]["logit_stability"] > 0
    teacher = EvaluationRequest(
        "V6-ppl",
        request.sample_id,
        request.task,
        request.messages,
        {"_mosaickv_teacher_forced_token_ids": (2,) * 20},
    )
    HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(teacher)
    forced = json.loads(next((tmp_path / teacher.run_id).glob("*.json")).read_text())[
        "draft_logit_allocation_guard"
    ]
    assert forced["attempts"] == guard["attempts"]
    assert forced["final_positions_sha256"] == guard["final_positions_sha256"]
    assert len(calls) == 2


def test_one_head_diagnostic_uses_one_prefill_private_states_and_complete_horizon(
    tmp_path, monkeypatch
):
    from mosaickv.evaluation.head_substitution_diagnostic import (
        HeadDiagnosticPolicy,
        HeadSubstitutionDiagnostic,
    )
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.selection.draft_shortlist import validate_shortlist_trace

    adapter = _llava()
    cfg = _config(MosaicKVMethod.MOSAICKV_EXACT)
    calls = []
    original = adapter.prefill

    def prefill(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "prefill", prefill)
    runtime = HeadSubstitutionDiagnostic(
        adapter, cfg, policy=HeadDiagnosticPolicy(pool_size=2), trace_directory=tmp_path
    )
    request = _runtime_request("one-head-diagnostic")
    result = runtime.diagnose(request)
    assert len(calls) == 1
    assert result["measured_source_cache_unchanged"] and result["logical_positions_unchanged"]
    assert result["live_private_states_after_diagnostic"] == 0
    assert result["active_cap_bytes"] == 512 and result["padding_bytes"] == 0
    assert not result["fresh_cohort_authorized"] and not result["paper_campaign_authorized"]
    assert result["one_head_replacement_only"] and not result["layer_capacities_changed"]
    validate_shortlist_trace(result["selection"])
    assert all(
        len(event["steps"]) == 32
        for event in result["selection"]["events"]
        if event["status"] == "full_horizon_pass"
    )
    with pytest.raises(ValueError, match="never targets"):
        runtime.diagnose(
            EvaluationRequest(
                request.run_id,
                request.sample_id,
                request.task,
                request.messages,
                {"reference": "forbidden"},
            )
        )
    saved = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert saved["not_a_benchmark_prediction_or_performance_row"]


def test_identical_anchor_followup_null_controls_on_real_cpu_transport(tmp_path, monkeypatch):
    from mosaickv.backends.hf_allocation_guard import packed_positions_sha
    from mosaickv.evaluation.head_substitution_followup import HeadSubstitutionFollowup

    adapter = _llava()
    calls = []
    original_prefill = adapter.prefill

    def prefill(*args, **kwargs):
        calls.append(1)
        return original_prefill(*args, **kwargs)

    monkeypatch.setattr(adapter, "prefill", prefill)

    class FixtureFollowup(HeadSubstitutionFollowup):
        def _observe_exact_anchor(self, prefill, full_state, anchor, attention, budget):
            # Synthetic CPU fixture only; real runs obtain this from the hashed V7 artifact.
            self.followup["anchor_positions_sha256"] = packed_positions_sha(anchor)
            return super()._observe_exact_anchor(prefill, full_state, anchor, attention, budget)

    runtime = FixtureFollowup(
        adapter,
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        trace_directory=tmp_path,
        followup={"anchor_positions_sha256": "fixture-only", "survivors": []},
    )
    result = runtime.diagnose(_runtime_request("identical-anchor-controls"))
    assert len(calls) == 1 and result["fullkv_prefills"] == 0
    assert result["controls_passed"] and result["extensions"] == []
    assert result["candidate_updates"] == 64 and result["reference_updates"] == 32
    assert result["probe_updates"] == 128
    assert result["active_cap_bytes"] == 512 and result["padding_bytes"] == 0
    assert result["measured_source_cache_unchanged"] and result["logical_positions_unchanged"]
    assert result["live_private_states_after_diagnostic"] == 0
    assert not result["selection_modified_by_probe"] and not result["existing_pool_exhausted"]
    assert not result["fresh_cohort_authorized"] and not result["paper_campaign_authorized"]
    assert set(runtime.saved_logits) == {"fullkv", "anchor", "null_clone", "null_repack"}
    for control in result["controls"].values():
        assert control["registered_guard"]["all_prefixes_checked"]
        assert len(control["steps"]) == 32
        assert all("logit_kl_fp64" in r["candidate"] for r in control["steps"])
    for role in ("null_clone", "null_repack"):
        assert len(runtime.saved_logits[role]) == 32
        assert all(
            torch.equal(runtime.saved_logits[role][step], tensor)
            for step, tensor in runtime.saved_logits["anchor"].items()
        )


@pytest.mark.parametrize("version", [5, 6])
def test_draft_one_token_ppl_skips_all_probes_and_does_not_preallocate(
    tmp_path, monkeypatch, version
):
    from mosaickv.config import DraftAllocationGuardConfig, DraftExchangeSearchConfig
    from mosaickv.evaluation.model import EvaluationRequest

    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(
            2,
            1,
            0.2,
            "draft_single_exchange_v1" if version == 6 else "draft_logit_guard_v1",
            DraftAllocationGuardConfig(4, (1, 4)),
            DraftExchangeSearchConfig() if version == 6 else None,
        ),
    )
    from mosaickv.config import ForecastingConfig
    from mosaickv.types import ForecastMode

    cfg = replace(
        cfg,
        forecasting=ForecastingConfig(
            enabled=True,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=2,
            draft_steps=0,
            centroid_count=2,
        ),
    )
    adapter = _llava()

    def forbidden(*_a, **_k):
        pytest.fail("zero-update PPL must not decode or preallocate")

    monkeypatch.setattr(adapter, "decode_one_token", forbidden)
    monkeypatch.setattr(adapter, "preallocate_decode_cache", forbidden)
    request = _runtime_request("V5-single-token")
    request = EvaluationRequest(
        request.run_id,
        request.sample_id,
        request.task,
        request.messages,
        {"_mosaickv_teacher_forced_token_ids": (2,)},
    )
    result = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(request)
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    guard = trace["draft_logit_allocation_guard"]
    assert guard["reason"] == "no_decode_updates" and guard["probe_updates"] == 0
    assert (
        result.metrics.initial_active_kv_storage_bytes
        == result.metrics.peak_active_kv_storage_bytes
        == 512
    )


@pytest.mark.parametrize("version", [5, 6])
def test_draft_retention_one_bypasses_probes_and_matches_fullkv(tmp_path, monkeypatch, version):
    import mosaickv.backends.hf_allocation_guard as guard_module
    import mosaickv.backends.hf_single_exchange_guard as single_module
    from mosaickv.config import DraftAllocationGuardConfig, DraftExchangeSearchConfig

    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, ratio=1.0),
        bounded_decode=BoundedDecodeConfig(
            2,
            1,
            0.2,
            "draft_single_exchange_v1" if version == 6 else "draft_logit_guard_v1",
            DraftAllocationGuardConfig(4, (1, 4)),
            DraftExchangeSearchConfig() if version == 6 else None,
        ),
    )
    full = replace(
        cfg,
        method=MosaicKVMethod.FULL_KV,
        bounded_decode=None,
        forecasting=replace(cfg.forecasting, enabled=False),
        graph=replace(cfg.graph, enabled=False),
        selection=replace(cfg.selection, enabled=False),
    )
    adapter = _llava()

    def forbidden(*_a, **_k):
        pytest.fail("retention-1 identity must not probe or change the cache")

    monkeypatch.setattr(guard_module, "guard_joint_allocation", forbidden)
    monkeypatch.setattr(single_module, "guard_single_exchanges", forbidden)
    request = _runtime_request("V5-identity")
    reference = HuggingFaceMosaicKVModel(adapter, full, trace_directory=tmp_path / "full").generate(
        request
    )
    result = HuggingFaceMosaicKVModel(
        adapter, cfg, trace_directory=tmp_path / "candidate"
    ).generate(request)
    assert result.answer == reference.answer
    left = json.loads(next((tmp_path / "full").rglob("*.json")).read_text())
    right = json.loads(next((tmp_path / "candidate").rglob("*.json")).read_text())
    assert left["generated_token_ids"] == right["generated_token_ids"]
    assert "draft_logit_allocation_guard" not in right


def _config(method=MosaicKVMethod.PROMPT_ATTENTION_TOPK, *, ratio=0.9999999999999999):
    base = (
        _baseline_runtime_config(method, ratio)
        if method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
        else _runtime_config(method, ratio)
    )
    return replace(
        base,
        generation=GenerationConfig(max_new_tokens=20),
        cache=CacheConfig(
            512,
            BudgetUnit.BYTES,
            ratio,
            1,
            physical_budget_block_size=1,
            physical_budget_mandatory_policy="last_token",
        ),
        bounded_decode=BoundedDecodeConfig(generated_capacity=2, sink_tokens=1),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE),
    )


@pytest.mark.parametrize(
    "method", [MosaicKVMethod.MOSAICKV_EXACT, MosaicKVMethod.PROMPT_ATTENTION_TOPK]
)
def test_full_runtime_caps_initial_final_peak_and_writes_variant_trace(tmp_path, method):
    runtime = HuggingFaceMosaicKVModel(_llava(), _config(method), trace_directory=tmp_path)
    result = runtime.generate(_runtime_request(f"bounded-{method.value}"))
    assert result.metrics.generated_tokens == 20
    assert result.metrics.active_kv_bytes == 512
    assert result.metrics.initial_active_kv_storage_bytes == 512
    assert result.metrics.final_active_kv_storage_bytes == 512
    assert result.metrics.peak_active_kv_storage_bytes == 512
    assert result.effective_method.endswith("__bounded_generated_history_v1")
    trace_path = next(tmp_path.rglob("*.json"))
    trace = json.loads(trace_path.read_text())
    record = trace["bounded_decode"]
    assert record["protected_prompt_positions_per_head"] == 2
    assert record["evicted_generated_positions_per_head"] == 17
    assert record["next_decode_position"] == 23
    assert set(record["retained_generated_logical_positions_physical_order"]) == {4, 22}
    assert trace["cache_lifecycle"]["peak_active_kv_storage_bytes"] == 512
    assert record["budget_reservation"]["reserved_generated_bytes"] == 256


@pytest.mark.parametrize(
    "allocation", ["uniform", "output_impact_guarded_v1", "direct_error_exchange_v1"]
)
def test_retention_one_bypasses_ring_and_matches_untouched_fullkv(tmp_path, allocation):
    adapter = _llava()
    bounded = HuggingFaceMosaicKVModel(
        adapter,
        replace(
            _config(MosaicKVMethod.MOSAICKV_EXACT, ratio=1.0),
            bounded_decode=BoundedDecodeConfig(2, 1, layer_allocation=allocation),
        ),
        trace_directory=tmp_path / "bounded",
    )
    reference_config = replace(
        _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0),
        generation=GenerationConfig(max_new_tokens=20),
    )
    reference = HuggingFaceMosaicKVModel(
        adapter, reference_config, trace_directory=tmp_path / "reference"
    )
    assert (
        bounded.generate(_runtime_request("identity")).answer
        == reference.generate(_runtime_request("reference")).answer
    )
    trace = json.loads(next((tmp_path / "bounded").rglob("*.json")).read_text())
    ref_trace = json.loads(next((tmp_path / "reference").rglob("*.json")).read_text())
    assert trace["generated_token_ids"] == ref_trace["generated_token_ids"]
    assert trace["bounded_decode"]["status"] == "bypassed_retention_one_identity"
    assert trace["effective_method"].endswith("__retention_one_exact")


def test_ring_logits_equal_explicit_chronological_eviction_reference():
    """Checks actual HF RoPE/attention before/after many ring wraps, not mocks."""
    adapter = _llava(kv_heads=2)
    prepared = adapter.prepare_inputs(_runtime_request("rope").messages)
    prefill = adapter.prefill(prepared)
    prompt = adapter._legacy_layers(prefill.state.past_key_values)[0]
    next_position = prefill.state.next_decode_position
    state = DecodeState(
        adapter.inject_past_key_values(
            adapter.extract_past_key_values(prefill.state.past_key_values)
        ),
        torch.ones(1, 4, dtype=torch.long),
        4,
        4,
        4,
        (),
        {"mosaickv_packed_all_valid": True, "mosaickv_cache_all_valid": True},
    )
    install_bounded_history(adapter, state, generated_capacity=3, sink_tokens=1, cap_bytes=7 * 256)
    reference_layers = tuple((k.clone(), v.clone()) for k, v in prompt)
    generated_positions = []
    token = prefill.next_token_id
    for step in range(16):
        # Before an append, remove only the oldest non-sink generated slot.
        if len(generated_positions) == 3:
            keep = torch.tensor([0, 1, 2, 3, 4, 6])
            reference_layers = tuple(
                (k.index_select(-2, keep), v.index_select(-2, keep)) for k, v in reference_layers
            )
            generated_positions.pop(1)
        from transformers import DynamicCache

        reference_cache = DynamicCache.from_legacy_cache(reference_layers)
        position = torch.tensor([next_position + step])
        with torch.inference_mode():
            output = adapter.model(
                input_ids=token,
                past_key_values=reference_cache,
                position_ids=position.reshape(1, 1),
                cache_position=position,
                use_cache=True,
                return_dict=True,
            )
        result = adapter.decode_one_token(token, state, capture_queries=False)
        torch.testing.assert_close(result.logits, output.logits[:, -1, :], rtol=1e-5, atol=1e-6)
        assert torch.equal(result.next_token_id, output.logits[:, -1, :].argmax(-1, keepdim=True))
        reference_layers = adapter._legacy_layers(output.past_key_values)[0]
        generated_positions.append(next_position + step)
        token = result.next_token_id
        assert state.next_decode_position == next_position + step + 1
        assert CacheLifecycle.count(adapter._legacy_layers(state.past_key_values)[0])[1] == 7 * 256


def test_teacher_forcing_uses_bounded_path_and_complete_targets(tmp_path):
    adapter = _llava()
    target_ids = tuple(range(3, 23))
    adapter.processor.tokenizer = lambda *_args, **_kwargs: {
        "input_ids": torch.tensor([target_ids])
    }
    runtime = HuggingFaceMosaicKVModel(adapter, _config(), trace_directory=tmp_path)
    request = _runtime_request("teacher-bounded")
    result = runtime.teacher_forced_nll(request, "a synthetic target")
    assert result.evaluated_tokens == 20
    assert result.negative_log_likelihood_sum >= 0
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["generated_token_ids"] == list(target_ids)
    assert trace["teacher_forced_evaluation"]["evaluated_tokens"] == 20
    assert trace["teacher_forced_evaluation"]["target_was_not_truncated"]
    assert trace["bounded_decode"]["evicted_generated_positions_per_head"] == 17


def test_unsupported_attention_and_unsafe_state_apis_fail_explicitly():
    adapter = _llava(attention="sdpa")
    prepared = adapter.prepare_inputs(_runtime_request("unsupported").messages)
    state = adapter.prefill(prepared).state
    with pytest.raises(RuntimeError, match=r"LLaVA-1\.5 eager"):
        install_bounded_history(adapter, state, generated_capacity=2, sink_tokens=0, cap_bytes=768)
    adapter.set_attention_implementation("eager")
    state.model_state["mosaickv_packed_all_valid"] = True
    install_bounded_history(adapter, state, generated_capacity=2, sink_tokens=0, cap_bytes=768)
    with pytest.raises(NotImplementedError, match="generic snapshot/reinjection"):
        adapter.extract_past_key_values(state.past_key_values)
    with pytest.raises(RuntimeError, match="erase bounded-history"):
        adapter.preallocate_decode_cache(state, 8)


def test_mandatory_underflow_preserves_failure_trace(tmp_path):
    config = _config()
    config = replace(
        config,
        cache=replace(
            config.cache, budget_value=384, physical_budget_mandatory_policy="first_and_last"
        ),
    )
    runtime = HuggingFaceMosaicKVModel(_llava(), config, trace_directory=tmp_path)
    with pytest.raises(RuntimeError, match="mandatory"):
        runtime.generate(_runtime_request("mandatory-underflow"))
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["status"] == "failed"
    assert trace["bounded_decode"]["budget_reservation"]["reserved_generated_bytes"] == 256


def test_development_preset_has_reproducible_strict_configuration():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs/development/llava_bounded_history_v1.yaml"
    config = load_config(path)
    assert config.bounded_decode == BoundedDecodeConfig(32, 4)
    assert config_sha256(RunConfig.from_mapping(canonical_config(config))) == config_sha256(config)
    with pytest.raises(ConfigurationError, match="bounded_decode requires"):
        replace(config, cache=replace(config.cache, total_storage_budget_bytes=1))


def test_bounded_native_metrics_share_a_policy_bound_full_reference(tmp_path):
    import hashlib

    from PIL import Image

    from mosaickv.config import DatasetConfig
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import record_hf_run

    adapter = _llava()
    config = _config()
    spec = default_task_registry().resolve("llava_description")
    config = replace(config, dataset=DatasetConfig(spec.dataset_id, "b" * 40, spec.split))
    full = replace(
        config,
        method=MosaicKVMethod.FULL_KV,
        bounded_decode=None,
        cache=replace(config.cache, retention_ratio=1.0, budget_value=2**63 - 1),
    )
    media = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(media)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            dict(
                schema_version=1,
                task=spec.name,
                dataset_id=spec.dataset_id,
                dataset_revision="b" * 40,
                split=spec.split,
                samples=[
                    dict(
                        sample_id="native-fixture",
                        prompt="Describe.",
                        references=["target"],
                        teacher_forcing_target="human target",
                        media=[
                            dict(
                                kind="image",
                                path=str(media),
                                sha256=hashlib.sha256(media.read_bytes()).hexdigest(),
                            )
                        ],
                    )
                ],
            )
        )
    )
    reference = tmp_path / "reference.json"
    policy = dict(
        name="common_bounded_history_v1_fullkv_reference_exempt",
        generated_capacity=2,
        sink_tokens=1,
    )
    common = dict(
        adapter=adapter,
        task=spec.name,
        output_root=tmp_path / "output",
        ledger_path=ledger,
        ledger_sha=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        sample_ids=("native-fixture",),
        phase="synthetic_cpu_not_quality_evidence",
        repetition=0,
        order=0,
        export_metrics=True,
        materialized_prefixkv=True,
        comparison_decode_policy=policy,
    )
    first = record_hf_run(full, **common, run_id="full", fullkv_reference_output=reference)
    second = record_hf_run(
        config,
        **common,
        run_id="topk",
        fullkv_reference_input=reference,
        fullkv_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
    )
    assert first["rows"][0]["status"] == second["rows"][0]["status"] == "completed"
    assert second["rows"][0]["active_kv_bytes"] == 512
