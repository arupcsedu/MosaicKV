import pytest
import torch

from mosaickv.evaluation.exact_history_timing import measure, summarize, trial_order


def fixture_rows():
    return [
        dict(
            **r,
            decode_wall_seconds=2.0 if r["variant"] == "raw_history" else 1.5,
            logits_bitwise=True,
            tensor_result_auditing=False,
        )
        for r in trial_order(seed=19)
    ]


def test_method_workers_preserve_input_routing_and_complete_denominators():
    from evicastkv_shared_source_runtime import expected_row_count, history_method_jobs

    parents = [dict(index=i, inputs=[f"immutable-{i}"]) for i in range(2)]
    variants = ("evicastkv", "topk", "prefixkv_reimpl", "vl_cache_reimpl", "lookm_reimpl")
    jobs = history_method_jobs(parents, variants, seed=19)
    assert jobs == history_method_jobs(parents, variants, seed=19)
    assert [j["index"] for j in jobs] == list(range(10))
    assert expected_row_count(dict(jobs=jobs, variants=variants)) == 10
    assert expected_row_count(dict(jobs=parents, variants=variants)) == 10
    for i in range(2):
        group = [j for j in jobs if j["input_index"] == i]
        assert {j["evaluation_variant"] for j in group} == set(variants)
        assert all(j["inputs"] == [f"immutable-{i}"] for j in group)
    assert all("evaluation_variant" not in p for p in parents)


def test_order_is_deterministic_randomized_paired_and_warmups_remain_visible():
    rows = list(trial_order(seed=19))
    assert rows == list(trial_order(seed=19))
    assert len(rows) == 50 and sum(r["warmup"] for r in rows) == 10
    for index in range(0, 50, 2):
        assert {r["variant"] for r in rows[index : index + 2]} == {
            "raw_history", "exact_source_history"
        }
    assert len({r["variant"] for r in rows if r["order_index"] == 0}) == 2


@pytest.mark.parametrize("args", [{"warmups": 4}, {"repetitions": 19}, {"seed": True}])
def test_invalid_registration_rejected(args):
    with pytest.raises(ValueError):
        list(trial_order(**dict({"seed": 19}, **args)))


def test_summary_is_paired_and_preserves_both_timing_distributions():
    result = summarize(fixture_rows(), seed=19)
    assert result["timed_pairs"] == 20 and result["warmup_pairs"] == 5
    assert result["mean_time_ratio"] == 0.75
    assert result["paired_time_ratio_CI95"] == [0.75, 0.75]
    assert result["paired_difference_CI95"] == [-0.5, -0.5]
    assert result["summaries"]["raw_history"]["p95"] == 2.0
    assert not result["whole_request_latency"] and result["independent_quality_N"] == 0


@pytest.mark.parametrize("case", ["missing", "duplicate", "changed_logits", "audited"])
def test_incomplete_or_invalid_timing_never_becomes_a_pass(case):
    rows = fixture_rows()
    if case == "missing":
        rows.pop()
    elif case == "duplicate":
        rows[-1] = rows[0]
    elif case == "changed_logits":
        rows[-1]["logits_bitwise"] = False
    else:
        rows[-1]["tensor_result_auditing"] = True
    with pytest.raises(ValueError):
        summarize(rows, seed=19)


@pytest.mark.parametrize("changed", [False, True])
def test_measure_isolated_common_policies_and_every_logit_checked(monkeypatch, changed):
    reference = torch.zeros(128, 3)
    calls, records = [], []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 32)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 64)
    monkeypatch.setattr("mosaickv.evaluation.exact_history_timing.gc.collect", lambda: None)

    def run(*args, **kwargs):
        calls.append(kwargs)
        return reference + int(changed), dict(decode_wall_seconds=1.0, decode_cuda_ms=1000.0)

    if changed:
        with pytest.raises(RuntimeError, match="original greedy logit bit"):
            measure(None, None, (), reference, seed=19, record=records.append, runner=run)
        assert not records
    else:
        rows = measure(None, None, (), reference, seed=19, record=records.append, runner=run)
        assert len(rows) == len(records) == 50
        assert all(not c["audit"] and c["sequential_lease"] for c in calls)
        assert all(r["storage_record_not_peak_evidence"] for r in rows)
