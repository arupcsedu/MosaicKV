import evicastkv_batched_history_study as module
import torch

import mosaickv.evaluation.storage_runs as runtime


def test_history_study_preserves_control_horizons_and_excludes_fullkv_timing(tmp_path, monkeypatch):
    def variants(*args, **kwargs):
        return {"dual": ("format",)}, []

    monkeypatch.setattr(module, "dual_choices", variants)
    reference = torch.ones(32, 4)

    def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
        reps, _ = module.base.choices()
        if job["method"] != "full_kv":
            module.base.timing(None, None, reps, reference, job["method"], plan)
        return [dict(method=job["method"])]

    monkeypatch.setattr(module.paired, "original_study", study)
    calls = []

    def run(*args, tokens=32, batched_history=False, **kwargs):
        calls.append((tokens, batched_history))
        return torch.ones(tokens, 4), dict(
            storage=dict(
                initial_active_kv_storage_bytes=10,
                final_active_kv_storage_bytes=20,
                peak_active_kv_storage_bytes=30,
            ),
            batched_history_seals=(tokens - 2) // 16,
            decode_wall_seconds=1.0,
        )

    monkeypatch.setattr(runtime, "run", run)
    plan = dict(
        history_control_horizons=[32, 128],
        evicast_initial_cap_bytes=100,
        evicast_decode_owned_peak_cap_bytes=100,
        timing_seed=3,
        warmups=0,
        timed_repetitions=1,
    )
    result = module.study(None, None, None, None, {}, None, reference, None, tmp_path, plan)[0]
    assert len(result["history_controls"]) == 12
    assert len(result["trials"]) == 10
    assert all(r["method"] != "full_kv" for r in result["trials"])
    assert sum(t == 128 for t, b in calls) == 12
    assert (tmp_path / "nested_evidence.json").exists()
