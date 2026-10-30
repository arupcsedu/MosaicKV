from types import SimpleNamespace

import evicastkv_paired_storage_baselines as module
import pytest
import torch

import mosaickv.evaluation.storage_runs as runtime


def test_common_choice_is_storage_only_per_layer(monkeypatch):
    def f(size, name):
        return SimpleNamespace(component_bytes=size, kind=name)
    monkeypatch.setattr(
        module,
        "original_choices",
        lambda *a, **k: (
            {"one": (f(10, "a"), f(20, "b")), "two": (f(15, "c"), f(12, "d"))},
            ["proof"],
        ),
    )
    result, proof = module.choices()
    assert [x.component_bytes for x in result["common_min_storage"]] == [10, 12]
    assert proof == ["proof"]


def test_all_compressed_methods_paired_and_fullkv_never_timed(tmp_path, monkeypatch):
    reference = torch.ones(2, 3)

    def fake_study(
        adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan
    ):
        method = job["method"]
        if method != "full_kv":
            module.base.timing(None, None, {"common": (method,)}, reference, method, plan)
        return [dict(method=method)]

    seen = []

    def fake_run(adapter, prefill, formats, **kwargs):
        seen.append(formats[0])
        return reference, dict(storage={}, decode_wall_seconds=1.0)

    monkeypatch.setattr(module, "original_study", fake_study)
    monkeypatch.setattr(runtime, "run", fake_run)
    original_timing = module.base.timing
    result = module.paired_study(
        None,
        None,
        None,
        None,
        {},
        None,
        None,
        None,
        tmp_path,
        dict(warmups=1, timed_repetitions=2, timing_seed=42),
    )
    assert module.base.timing is original_timing
    assert len(result[0]["baseline_controls"]) == 6
    assert len(result[0]["trials"]) == 15
    assert "full_kv" not in seen
    assert set(seen) == set(module.METHODS) - {"full_kv"}
    for trial in (-1, 0, 1):
        block = [r for r in result[0]["trials"] if r["trial"] == trial]
        assert len(block) == 5
        assert sorted(r["order_index"] for r in block) == list(range(5))


def test_modified_logits_fail_closed_and_restore_timing(tmp_path, monkeypatch):
    reference = torch.ones(2, 3)

    def fake_study(
        adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan
    ):
        if job["method"] != "full_kv":
            module.base.timing(
                None, None, {"common": (job["method"],)}, reference, job["method"], plan
            )
        return []

    monkeypatch.setattr(module, "original_study", fake_study)
    monkeypatch.setattr(runtime, "run", lambda *a, **k: (torch.zeros_like(reference), {}))
    original_timing = module.base.timing
    with pytest.raises(RuntimeError, match="changed"):
        module.paired_study(
            None,
            None,
            None,
            None,
            {},
            None,
            None,
            None,
            tmp_path,
            dict(warmups=1, timed_repetitions=2, timing_seed=42),
        )
    assert module.base.timing is original_timing
    assert (tmp_path / "nested_evidence.json").exists()
