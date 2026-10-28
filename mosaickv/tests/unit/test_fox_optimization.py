import ast
from pathlib import Path

import pytest

from mosaickv.evaluation.fox_campaign import digest, registered_arms, sha
from mosaickv.evaluation.fox_optimization import (
    budget_function_tree,
    budget_implementation,
    factorial_order,
    reference_budget,
    without_unit_reuse,
)


def test_implementation_factorial_has_no_duplicated_fullkv_or_changed_budgets():
    arms = registered_arms("coco", "optimization")
    assert len(arms) == 9
    assert arms.count(("full_kv", None, "complete")) == 1
    for method in ("prompt_attention_topk", "mosaickv_exact"):
        assert {(cap, variant) for m, cap, variant in arms if m == method} == {
            (cap, v) for cap in (20, 40) for v in ("reference-budget", "reuse-budget")
        }
    for method in ("prompt_attention_topk", "mosaickv_exact"):
        for cap in (20, 40):
            orders = [factorial_order(arms, i, r) for i in range(16) for r in range(3)]
            assert all(sorted(order, key=str) == sorted(arms, key=str) for order in orders)
            assert (
                sum(
                    o.index((method, cap, "reference-budget"))
                    < o.index((method, cap, "reuse-budget"))
                    for o in orders
                )
                == 24
            )


def test_pipeline_limit_is_distinct_from_six_gpu_aggregate_ceiling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from evicastkv_campaign import admitted_array_concurrency

    resources = dict(maximum_a100s=6, array_concurrency=2)
    assert [admitted_array_concurrency(resources, n) for n in (0, 3, 4, 5, 6)] == [2, 2, 2, 1, 0]
    with pytest.raises(ValueError):
        admitted_array_concurrency(dict(maximum_a100s=8), 0)


def test_reference_function_is_hash_bound_and_context_is_restored(tmp_path, monkeypatch):
    import mosaickv.backends.hf_runtime as runtime
    import mosaickv.evaluation.fox_optimization as module

    monkeypatch.setattr(module, "clean_source", lambda root, revision: revision)
    path = tmp_path / "mosaickv/src/mosaickv/backends/hf_runtime.py"
    path.parent.mkdir(parents=True)
    path.write_text("def _dense_physical_cache_budget(full, config):\n    return 'fixture'\n")
    contract = dict(
        reference_source_root=str(tmp_path),
        reference_source_sha="fixture",
        reference_runtime_sha256=sha(path),
        reference_function_sha256=digest(
            ast.dump(budget_function_tree(path), include_attributes=False)
        ),
    )
    original = runtime._dense_physical_cache_budget
    with budget_implementation(
        dict(optimization_protocol=contract), "reference-budget", required=True
    ):
        assert runtime._dense_physical_cache_budget(None, None) == "fixture"
    assert runtime._dense_physical_cache_budget is original
    with pytest.raises(ValueError, match="changed"):
        reference_budget(
            str(tmp_path), "fixture", "incorrect", contract["reference_function_sha256"]
        )
    with (
        pytest.raises(ValueError, match="unregistered"),
        budget_implementation({}, "arbitrary-fallback", required=True),
    ):
        pass


def test_semantic_gate_ignores_phase_timing_but_rejects_changed_selected_ids(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from evicastkv_optimization import semantic_check

    from mosaickv.evaluation.fox_campaign import write_once

    fields = dict(
        input_sha="fixture",
        processor_sha="fixture",
        scorer_sha="fixture",
        config_sha="fixture",
        token_ids=[1, 2],
        forward_steps=1,
        initial_kv_bytes=8,
        final_kv_bytes=12,
        peak_kv_bytes=12,
    )
    trace = dict(
        selected_blocks=dict(sha256="same"),
        packed_slots=dict(sha256="same"),
        forecast_statistics=dict(
            heads=dict(sha256="same"), provenance=dict(mode="prompt"), timing=dict(total=1)
        ),
    )
    write_once(tmp_path / "a.json", trace)
    trace["forecast_statistics"]["timing"]["total"] = 2
    write_once(tmp_path / "b.json", trace)
    left, right = (
        dict(fields, trace_path=str(tmp_path / "a.json")),
        dict(fields, trace_path=str(tmp_path / "b.json")),
    )
    assert semantic_check(left, right)["token_agreement"] == 1
    trace["selected_blocks"]["sha256"] = "different"
    write_once(tmp_path / "c.json", trace)
    with pytest.raises(ValueError, match="selected_blocks"):
        semantic_check(left, dict(fields, trace_path=str(tmp_path / "c.json")))


def test_unit_reference_reuse_matches_unmodified_slow_allocation_for_random_layouts():
    from dataclasses import replace

    import numpy as np

    import mosaickv.backends.hf_runtime as runtime
    from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.fox_campaign import source_config

    function = without_unit_reuse(budget_function_tree(Path(runtime.__file__)))
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    namespace = dict(vars(runtime))
    exec(compile(ast.fix_missing_locations(tree), "unchanged-slow-reference", "exec"), namespace)
    slow = namespace["_dense_physical_cache_budget"]
    rng = np.random.default_rng(20271028)
    base = RunConfig.from_mapping(source_config("coco"))
    for i in range(160):
        length, heads, batch = (
            int(rng.integers(4, 25)),
            int(rng.integers(1, 5)),
            int(rng.integers(1, 3)),
        )
        logical = tuple(range(length)) if i % 2 else tuple(range(0, length * 2, 2))
        original = logical[-1] + 1
        mandatory = (logical[-1],) if i % 3 else (logical[0], logical[-1])
        layers = tuple(
            (
                np.zeros((batch, heads, length, 4), np.float16),
                np.zeros((batch, heads, length, 4), np.float32),
            )
            for _ in range(1 + i % 3)
        )
        full = FullKVState.from_tensors(
            layers,
            block_size=1,
            logical_positions=logical,
            original_logical_sequence_length=original,
            mandatory_logical_positions=mandatory,
            modality_spans=(
                ModalitySpan(0, original // 2, Modality.IMAGE),
                ModalitySpan(original // 2, original, Modality.TEXT),
            ),
        )
        if i % 5 == 0:
            full = replace(full, blocks=tuple(reversed(full.blocks)))
        width = full.active_bytes // length
        config = replace(
            base,
            cache=replace(
                base.cache,
                budget_value=int(rng.integers(1, length + 1)) * width + i % width,
                retention_ratio=(0.1, 0.4, 0.9999999999999999, 1.0)[i % 4],
                physical_budget_mandatory_policy=("last_token", "first_and_last", "method")[i % 3],
            ),
        )
        try:
            expected = slow(full, config)
        except runtime.HuggingFaceRuntimeError as error:
            with pytest.raises(type(error), match=__import__("re").escape(str(error))):
                runtime._dense_physical_cache_budget(full, config)
        else:
            actual = runtime._dense_physical_cache_budget(full, config)
            assert actual == expected
            assert list(actual.position_limits) == list(expected.position_limits)


def test_reuse_does_not_reblock_identical_unit_policy_but_keeps_different_policy_fallback(
    monkeypatch,
):
    from dataclasses import replace

    import numpy as np

    import mosaickv.backends.hf_runtime as runtime
    from mosaickv.cache_state import FullKVState
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.fox_campaign import source_config

    full = FullKVState.from_tensors(
        ((np.zeros((1, 2, 8, 4)), np.zeros((1, 2, 8, 4))),),
        block_size=1,
        mandatory_logical_positions=(7,),
    )
    base = RunConfig.from_mapping(source_config("coco"))
    config = replace(base, cache=replace(base.cache, budget_value=full.active_bytes // 2))
    calls = []
    original = FullKVState.from_tensors

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(FullKVState, "from_tensors", observed)
    runtime._dense_physical_cache_budget(full, config)
    assert calls == []
    runtime._dense_physical_cache_budget(
        full,
        replace(
            config, cache=replace(config.cache, physical_budget_mandatory_policy="first_and_last")
        ),
    )
    assert len(calls) == 1
