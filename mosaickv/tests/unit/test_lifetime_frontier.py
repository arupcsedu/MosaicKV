"""Geometry-only capacity invariance; no labels, real samples or GPU access."""

import json

import evicastkv_lifetime_frontier as module
import pytest

from mosaickv.evaluation.fox_campaign import sha


@pytest.mark.parametrize("source_length", [593, 600, 606, 700])
def test_history_frontier_preserves_prompt_while_accounting_extra_history(source_length):
    ratios = tuple(i / 40 for i in range(32))
    plans = [
        module.capacity_plan(ratios, [source_length] * 32, [25] * 32, [16384] * 32, g)
        for g in module.HISTORY_GRID
    ]
    assert len({p.prompt_positions for p in plans}) == 1
    assert len({p.prompt_bytes for p in plans}) == 1
    assert [p.history_bytes for p in plans] == [g * 524288 for g in module.HISTORY_GRID]
    assert all(p.charged_ceiling_bytes <= p.cap_bytes for p in plans)
    assert [p.cap_bytes // 2**20 for p in plans] == [223, 289, 355, 421]


def test_frontier_rejects_heterogeneous_geometry_or_unregistered_capacity():
    with pytest.raises(ValueError, match="geometry"):
        module.capacity_plan([0.1, 0.8], [600] * 2, [1] * 2, [16, 32], 128)
    with pytest.raises(ValueError, match="history"):
        module.capacity_plan([0.1, 0.8], [600] * 2, [1] * 2, [16, 16], 192)


def test_geometry_reads_only_explicit_kv_shapes():
    shapes = dict(
        key_shape=[1, 32, 600, 128],
        value_shape=[1, 32, 600, 128],
        key_element_size=2,
        value_element_size=2,
    )
    assert module.geometry(dict(source_kv_layout=[shapes])) == ((600,), (16384,))


@pytest.mark.parametrize(
    "quality_loss,byte_mismatch", [(False, False), (True, False), (False, True)]
)
def test_frontier_analyzer_keeps_both_metrics_and_memory_gates(
    tmp_path, monkeypatch, quality_loss, byte_mismatch
):
    jobs = []
    for task in ("llava_description", "mmvet_prefixkv"):
        native = tmp_path / f"native-{task}.json"
        native.write_text(
            json.dumps(
                dict(
                    teacher_forced_ppl=3.0,
                    rouge_l_f1=0.5,
                    metrics=dict(zip(module.common.STORAGE, (100, 200, 300), strict=True)),
                    audited_decode_transaction_peak_bytes=300,
                )
            )
        )
        for method in ("prompt_attention_topk", "mosaickv_exact"):
            for history in module.HISTORY_GRID:
                index = len(jobs)
                label = f"{task}-{method}-{history}"
                worker = tmp_path / f"workers/{index}"
                (worker / "rows").mkdir(parents=True)
                trace = worker / "trace.json"
                trace.write_text(
                    json.dumps(
                        dict(
                            execution_phase="generation",
                            evidence_layer_allocation=dict(
                                capacity=dict(prompt_positions=[10, 20]),
                                selected_positions_sha256=f"fixed-{method}",
                            ),
                        )
                    )
                )
                row = dict(
                    task=task,
                    sample_id=task,
                    label=label,
                    method=method,
                    status="completed",
                    teacher_forced_ppl=3.1 if quality_loss and method == "mosaickv_exact" else 2.9,
                    rouge_l_f1=0.6,
                    nll_sum=10.0,
                    evaluated_tokens=10,
                    audited_decode_transaction_peak_bytes=80,
                    audited_request_owned_peak_bytes=400,
                    pytorch_request_peak_allocated_bytes=1000,
                    memory_observer_token_parity=True,
                    traces={str(trace): sha(trace)},
                    memory_audit_traces={str(trace): sha(trace)},
                    metrics=dict(
                        zip(module.common.STORAGE, (40, 60, 80), strict=True),
                        end_to_end_time=1.0,
                        compression_time=0.2,
                    ),
                )
                if byte_mismatch and method == "mosaickv_exact":
                    row["metrics"][module.common.STORAGE[0]] = 41
                (worker / f"rows/{label}.json").write_text(json.dumps(row))
                (worker / "result.json").write_text(json.dumps(dict(failures=[])))
                jobs.append(
                    dict(
                        index=index,
                        task=task,
                        native_control_row=str(native),
                        planned_prompt_positions=[10, 20],
                        diagnostic_history_capacity=history,
                        arms=[dict(label=label)],
                    )
                )
    monkeypatch.setattr(module.common, "inputs", lambda *_: dict(source_sha="fixture", jobs=jobs))
    module.analyze(tmp_path, "fixture")
    gate = module.common.read(tmp_path / "analysis/GATE.json")
    assert len(gate["checks"]) == 8
    assert bool(gate["point_eligible_history_capacities"]) == (not (quality_loss or byte_mismatch))
    assert gate["status"] == ("failed" if byte_mismatch else "complete_consumed_diagnostic")
    assert not gate["superiority_established"] and not gate["paper_campaign_authorized"]
