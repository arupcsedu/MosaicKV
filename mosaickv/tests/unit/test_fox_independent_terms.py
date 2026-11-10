"""Independent objective terms retain fixed routing, graph and byte controls."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import evicastkv_independent_terms as study
import pytest

from mosaickv.config import RunConfig, canonical_config
from mosaickv.evaluation.fox_campaign import source_config


def config():
    value = RunConfig.from_mapping(source_config("coco"))
    return replace(value, cache=replace(value.cache, budget_value=study.CAP))


@pytest.mark.parametrize(
    "variant,zeroed",
    [
        ("term-full", ()),
        ("term-attention-only", ("lambda_v", "lambda_o", "lambda_g", "lambda_m")),
        ("term-attention-plus-value", ("lambda_o", "lambda_g", "lambda_m")),
        ("term-value-off", ("lambda_v",)),
        ("term-novelty-off", ("lambda_o",)),
        ("term-graph-coverage-off", ("lambda_g",)),
        ("term-modality-coverage-off", ("lambda_m",)),
        ("term-cross-modal-off", ()),
    ],
)
def test_terms_are_independent_not_grouped(variant, zeroed):
    base = config()
    changed = study.term_config(base, variant)
    assert study.coefficients(changed) == {
        key: 0.0 if key in zeroed else value for key, value in study.coefficients(base).items()
    }
    assert changed.cache is base.cache
    assert changed.forecasting is base.forecasting
    assert changed.graph.uncertain_head_fraction == 1 / 16
    assert changed.prototypes.uncertain_head_fraction == 1 / 16
    assert changed.graph.cross_modal_weight == (
        0 if variant == "term-cross-modal-off" else base.graph.cross_modal_weight
    )


def test_grid_complete_consumed_deterministic_and_no_random_reruns():
    jobs = study.jobs()
    assert jobs == study.jobs()
    assert len(jobs) == 4
    assert [index for job in jobs for index in job["indices"]] == list(range(16))
    assert sum(len(job["indices"]) * len(job["arms"]) for job in jobs) == 128
    assert len(study.comparisons()) == 7
    assert all(method == "mosaickv_exact" and cap == 40 for method, cap, _ in study.arms())
    assert not any("random" in variant for variant in study.VARIANTS)


@pytest.mark.parametrize("variant", ["complete", "wrong", "random-13", "route-1.0"])
def test_unknown_terms_fail(variant):
    with pytest.raises(ValueError, match="registered independent term"):
        study.term_config(config(), variant)


def test_wrong_cap_fails():
    base = config()
    with pytest.raises(ValueError, match="40 MiB"):
        study.term_config(
            replace(base, cache=replace(base.cache, budget_value=20 * 2**20)), "term-full"
        )


def test_overlay_restores_globals_and_uses_native_crossmodal_audit(monkeypatch):
    import evicastkv_worker as worker

    import mosaickv.evaluation.fox_mechanisms as audits

    base, seen = config(), []

    def original(lock, workload, method, cap, *, variant="complete"):
        return base

    def audit(variant, *args, **kwargs):
        seen.append(variant)
        return variant

    monkeypatch.setattr(worker, "resolved_config", original)
    monkeypatch.setattr(audits, "mechanism_audit", audit)
    with study.config_overlay():
        current = worker.resolved_config({}, "coco", "mosaickv_exact", 40, variant="term-value-off")
        assert current.utility.lambda_v == 0
        assert current.utility.lambda_o == base.utility.lambda_o
        assert (
            worker.resolved_config({}, "coco", "mosaickv_exact", 40).graph.uncertain_head_fraction
            == 1 / 16
        )
        assert worker.resolved_config({}, "coco", "full_kv", None) is base
        assert audits.mechanism_audit("term-cross-modal-off", "id") == "cross-modal-off"
        assert audits.mechanism_audit("term-value-off", "id") == "mechanism-complete"
        with pytest.raises(ValueError, match="unregistered"):
            worker.resolved_config({}, "milebench", "mosaickv_exact", 40, variant="term-full")
        with pytest.raises(ValueError, match="unregistered"):
            audits.mechanism_audit("complete", "id")
    assert worker.resolved_config is original
    assert audits.mechanism_audit is audit
    assert seen == ["cross-modal-off", "mechanism-complete"]


@pytest.fixture
def valid_rows(tmp_path):
    base = study.term_config(config(), "term-full")
    contract = dict(sample_ids=["consumed"], full_config=canonical_config(base))
    edge = tmp_path / "edges.parquet"
    terms = tmp_path / "raw_marginal_terms.parquet"
    edge.write_bytes(b"synthetic edge fixture, not empirical data")
    terms.write_bytes(b"synthetic raw term fixture, not empirical data")
    rows = []
    for index, variant in enumerate(study.VARIANTS):
        resolved = tmp_path / f"resolved-{index}.json"
        resolved.write_text(json.dumps(canonical_config(study.term_config(base, variant))))
        activity = {field: "fixed" for field in study.DOMAIN_FIELDS}
        activity.update(
            graph_routes=[[0, 0]],
            planner_routes=[[0, 0]],
            graph_nodes=5,
            graph_edges=6,
            selected_source_ids=[0, 1, 2],
            raw_terms={str(edge): study.sha(edge), str(terms): study.sha(terms)},
        )
        audit = tmp_path / f"audit-{index}.json"
        audit.write_text(json.dumps(activity))
        rows.append(
            dict(
                sample_id="consumed",
                method="mosaickv_exact",
                effective_method="mosaickv_exact",
                status="completed",
                variant=variant,
                budget_bytes=study.CAP,
                initial_kv_bytes=study.CAP,
                final_kv_bytes=study.CAP + 100,
                peak_kv_bytes=study.CAP + 100,
                emitted_tokens=64,
                forward_steps=63,
                input_sha="same-input",
                processor_sha="same-processor",
                scorer_sha="same-scorer",
                references=["synthetic reference"],
                media_group_ids=["synthetic media"],
                resolved_config_path=str(resolved),
                resolved_config_file_sha256=study.sha(resolved),
                mechanism_audit_path=str(audit),
                mechanism_audit_sha256=study.sha(audit),
            )
        )
    return rows, contract


def rewrite(row, key, data):
    path = Path(row[key + "_path"])
    path.write_text(json.dumps(data))
    hash_key = {
        "resolved_config": "resolved_config_file_sha256",
        "mechanism_audit": "mechanism_audit_sha256",
    }[key]
    row[hash_key] = study.sha(path)


def test_all_activities_keep_fixed_domain_and_actual_coefficients(valid_rows):
    rows, contract = valid_rows
    records = study.validate_rows(rows, contract)
    assert len(records) == 8
    assert all(record["selected_jaccard_vs_full"] == 1 for record in records)
    assert all(record["changed_source_memberships_vs_full"] == 0 for record in records)
    assert records[3]["coefficients"]["lambda_v"] == 0
    assert records[3]["coefficients"]["lambda_o"] != 0


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_incomplete_or_duplicate_grid_blocks(valid_rows, kind):
    rows, contract = valid_rows
    values = rows[:-1] if kind == "missing" else [*rows, rows[0]]
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        study.validate_rows(values, contract)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("status", "failed", "failed sample"),
        ("effective_method", "full_kv", "fallback"),
        ("initial_kv_bytes", study.CAP - 1, "physical cap"),
        ("final_kv_bytes", study.CAP + 1, "final/peak"),
        ("references", ["changed annotation"], "reference"),
        ("input_sha", "different prompt", "controls changed"),
        ("emitted_tokens", 63, "generation changed"),
    ],
)
def test_common_controls_fail_closed(valid_rows, field, value, match):
    rows, contract = valid_rows
    rows[-1][field] = value
    with pytest.raises(ValueError, match=match):
        study.validate_rows(rows, contract)


@pytest.mark.parametrize("field", study.DOMAIN_FIELDS)
def test_rehashed_candidate_or_quota_change_is_rejected(valid_rows, field):
    rows, contract = valid_rows
    row = rows[-1]
    activity = json.loads(Path(row["mechanism_audit_path"]).read_text())
    activity[field] = "changed"
    rewrite(row, "mechanism_audit", activity)
    with pytest.raises(ValueError, match="domain changed"):
        study.validate_rows(rows, contract)


def test_rehashed_combined_value_novelty_ablation_cannot_masquerade_as_value_only(valid_rows):
    rows, contract = valid_rows
    row = rows[3]
    config = json.loads(Path(row["resolved_config_path"]).read_text())
    config["utility"]["lambda_o"] = 0
    rewrite(row, "resolved_config", config)
    with pytest.raises(ValueError, match="unregistered coefficient"):
        study.validate_rows(rows, contract)


def test_graph_coverage_off_is_not_allowed_to_change_edges(valid_rows, tmp_path):
    rows, contract = valid_rows
    row = rows[5]
    activity = json.loads(Path(row["mechanism_audit_path"]).read_text())
    other = tmp_path / "changed" / "edges.parquet"
    other.parent.mkdir()
    other.write_bytes(b"different synthetic edges")
    activity["raw_terms"] = {
        path: value
        for path, value in activity["raw_terms"].items()
        if Path(path).name != "edges.parquet"
    }
    activity["raw_terms"][str(other)] = study.sha(other)
    rewrite(row, "mechanism_audit", activity)
    with pytest.raises(ValueError, match="changed graph edges"):
        study.validate_rows(rows, contract)


def test_crossmodal_intervention_can_delete_edges_without_being_graph_coverage_off(
    valid_rows, tmp_path
):
    rows, contract = valid_rows
    row = rows[-1]
    activity = json.loads(Path(row["mechanism_audit_path"]).read_text())
    other = tmp_path / "crossmodal" / "edges.parquet"
    other.parent.mkdir()
    other.write_bytes(b"synthetic reduced graph")
    activity["raw_terms"] = {
        path: value
        for path, value in activity["raw_terms"].items()
        if Path(path).name != "edges.parquet"
    }
    activity["raw_terms"][str(other)] = study.sha(other)
    activity["graph_edges"] = 4
    activity["cross_modal_edges_removed"] = 2
    rewrite(row, "mechanism_audit", activity)
    records = study.validate_rows(rows, contract)
    assert records[-1]["cross_modal_edges_removed"] == 2
    assert records[-1]["coefficients"]["lambda_g"] == records[0]["coefficients"]["lambda_g"]


def test_empty_activity_does_not_count_as_success(valid_rows):
    rows, contract = valid_rows
    for row in rows:
        activity = json.loads(Path(row["mechanism_audit_path"]).read_text())
        activity["selected_source_ids"] = []
        rewrite(row, "mechanism_audit", activity)
    with pytest.raises(ValueError, match="nonzero routing/selection"):
        study.validate_rows(rows, contract)


def test_mutation_fixture_is_independent(valid_rows):
    rows, contract = valid_rows
    preserved = copy.deepcopy(rows)
    study.validate_rows(rows, contract)
    assert rows == preserved
