"""Synthetic CPU records only; no model execution or new scientific evidence."""

from copy import deepcopy

import evicastkv_storage_boundary_reconciliation as study
import pytest


def rows():
    result = []
    for case in range(2):
        for target in study.TARGETS:
            for arm in study.ARMS:
                extra = 10 if arm == study.ARMS[1] else 0
                final, peak = target + 600 + extra, target + 1000 + extra * 2
                result.append(
                    dict(
                        case=case,
                        target_initial_bytes=target,
                        arm=arm,
                        status="completed",
                        actual_initial_bytes=target,
                        actual_final_bytes=final,
                        decode_owned_peak_bytes=peak,
                        original_logical_length=600,
                        final_next_position=727,
                        used_fullkv_fallback=False,
                        same_initial_bytes_only=True,
                        diagnostics={
                            name: dict(registered_guard_pass=False)
                            for name in (
                                "registered_guard_32",
                                "cache_dependent_guard_32",
                                "prefix_128",
                            )
                        },
                        storage=dict(
                            initial_active_kv_storage_bytes=target,
                            final_active_kv_storage_bytes=final,
                            peak_active_kv_storage_bytes=peak,
                            prompt_construction=[
                                dict(layer=i, format="synthetic", component_bytes=target)
                                for i in range(32)
                            ],
                            generated_history_storage=[
                                dict(sealed_chunks=7, raw_tail_used=15) for _ in range(32)
                            ],
                            history_updates_per_layer=[127] * 32,
                            generated_history_lossless_and_append_only=True,
                            unchanged_original_selected_memberships=True,
                            source_history=True,
                            source_history_constructor="direct",
                            source_history_normalized_tail_bytes=100,
                            active_workspace_bytes=0,
                            outstanding_layer_leases=0,
                            workspace_allocations=8,
                            maximum_workspace_capacity=728,
                            boundary="declared bounded decode-owned storage",
                            includes="prompt/history/lease/workspace",
                            excludes="source construction and internal GEMM",
                            reconstruction_tensor_results_observed=False,
                            source_history_constructor_tensor_results_observed=False,
                        ),
                    )
                )
    return result


def parent(tmp_path):
    root = tmp_path / "parent"
    root.mkdir()
    raw_rows = rows()
    study.write_once(root / "plan.json", dict(source_sha=study.PRODUCER_SHA, expected_rows=8))
    study.write_once(
        root / "worker/manifest.json",
        dict(
            source_sha=study.PRODUCER_SHA,
            source_dirty=False,
            plan_sha256=study.sha(root / "plan.json"),
            result_category="unified_comparison",
        ),
    )
    (root / "worker/raw.jsonl").write_text("\n".join(study.json.dumps(r) for r in raw_rows) + "\n")
    (root / "worker/logits.pt").write_bytes(b"hashed only, never tensor-loaded")
    study.write_once(
        root / "worker/result.json",
        dict(
            source_sha=study.PRODUCER_SHA,
            rows=8,
            status="completed",
            errors=[],
            plan_sha256=study.sha(root / "plan.json"),
            bindings={
                str(root / "worker" / n): study.sha(root / "worker" / n)
                for n in ("manifest.json", "raw.jsonl", "logits.pt")
            },
        ),
    )
    study.write_once(
        root / "GATE_constructed_shared_bytes.json",
        dict(
            complete=True,
            status="passed_constructed_initial_bytes",
            failures=[],
            final_and_peak_byte_match_claim=False,
            superiority_established=False,
            original_guard_thresholds=dict(maximum_KL_FP32=0.02, logit_difference=1.0),
            rows=raw_rows,
        ),
    )
    return root


def test_complete_pairs_keep_unknowns_and_do_not_infer_peak_categories():
    record = study.reconcile(rows())
    assert record["row_count"] == 8 and record["pair_count"] == 4
    assert record["complete_row_pairing"] is True
    assert record["valid_unified_comparison_evidence"] is False
    assert record["superiority_established"] is False
    assert record["saved_guard_pass_counts"]["prefix_128"] == 0
    pair = record["pairs"][0]
    assert pair["differences"] == dict(initial_bytes=0, final_bytes=10, observed_peak_bytes=20)
    assert pair["matched_initial"] and not pair["matched_final"]
    component = record["records"][0]
    assert (
        component["initial_components"]["gross_layer_sum_bytes"]
        > component["boundaries"]["initial_bytes"]
    )
    assert component["peak_components"]["category_bytes"] is None
    assert component["final_components"]["sealed_history_bytes"] is None
    assert component["final_components"]["unclassified_final_backing_bytes"] == (
        component["boundaries"]["final_bytes"] - 100
    )


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "unregistered"])
def test_no_intersection_or_duplicate_analysis(change):
    source = rows()
    if change == "missing":
        source.pop()
    elif change == "duplicate":
        source[-1] = deepcopy(source[0])
    elif change == "extra":
        source.append(deepcopy(source[0]))
    else:
        source[0]["case"] = 2
    with pytest.raises(ValueError, match=r"eight|duplicate, missing"):
        study.reconcile(source)


@pytest.mark.parametrize("change", ["endpoint", "history", "lease", "workspace", "fallback"])
def test_rejects_inconsistent_or_unregistered_records(change):
    source = rows()
    if change == "endpoint":
        source[0]["storage"]["final_active_kv_storage_bytes"] += 1
    elif change == "history":
        source[0]["storage"]["history_updates_per_layer"].pop()
    elif change == "lease":
        source[0]["storage"]["outstanding_layer_leases"] = 1
    elif change == "workspace":
        source[0]["storage"]["active_workspace_bytes"] = 1
    else:
        source[0]["used_fullkv_fallback"] = True
    with pytest.raises(ValueError):
        study.reconcile(source)


def test_parent_hash_binding_without_tensor_loading(tmp_path):
    root = parent(tmp_path)
    loaded, hashes, preserved = study.load_parent(root)
    assert len(loaded) == 8 and len(hashes) == 6
    assert preserved["category_is_historical_label_not_current_admission"] is True
    (root / "worker/logits.pt").write_bytes(b"altered")
    with pytest.raises(ValueError, match="artifact changed"):
        study.load_parent(root)


def test_gate_raw_disagreement_and_foreign_binding_are_rejected(tmp_path):
    root = parent(tmp_path)
    p = root / "GATE_constructed_shared_bytes.json"
    gate = study.read(p)
    gate["rows"][0]["actual_final_bytes"] += 1
    p.write_text(study.json.dumps(gate))
    with pytest.raises(ValueError, match="raw rows differ"):
        study.load_parent(root)
    p = root / "worker/result.json"
    result = study.read(p)
    result["bindings"][str(root / "plan.json")] = study.sha(root / "plan.json")
    p.write_text(study.json.dumps(result))
    with pytest.raises(ValueError, match="exactly its raw"):
        study.load_parent(root)


def test_run_requires_current_cpu_gate_and_write_once(tmp_path, monkeypatch):
    source = parent(tmp_path)
    monkeypatch.setattr(study, "clean_source", lambda _: "current")
    monkeypatch.setattr(study, "pinned_source_proofs", lambda: {"historical": "proof"})
    monkeypatch.setattr(study, "cpu_gate", lambda r, s: dict(status="passed", source_sha=s))
    out = tmp_path / "derived"
    record = study.run(out, cpu_gate_root=tmp_path / "cpu", parent=source)
    assert record["source_sha"] == "current"
    assert (out / study.GATE).is_file()
    assert record["saved_numerical_failures_preserved"] is True
    with pytest.raises(FileExistsError):
        study.run(out, cpu_gate_root=tmp_path / "cpu", parent=source)

    def bad_gate(*_):
        raise ValueError("current-source CPU missing")

    monkeypatch.setattr(study, "cpu_gate", bad_gate)
    with pytest.raises(ValueError, match="CPU missing"):
        study.run(tmp_path / "never", cpu_gate_root=tmp_path / "cpu", parent=source)
    assert not (tmp_path / "never" / study.GATE).exists()
