"""Frozen-constructor and launch controls; synthetic CPU evidence only."""

import copy
import json

import anchor_compaction_hf_v3 as gate
import pytest
import torch
import yaml

from mosaickv.evaluation.anchor_compaction_diagnostic import (
    METHOD,
    VARIANT,
    construct_anchor_heads,
    construction_policy,
)
from mosaickv.prototypes.attention_compaction import AttentionCompactionPolicy


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    rng = torch.Generator().manual_seed(21)
    source = (
        (
            torch.randn(1, 2, 6, 4, generator=rng, dtype=torch.float16),
            torch.randn(1, 2, 6, 4, generator=rng, dtype=torch.float16),
        ),
    )
    p = torch.rand(2, 32, 6, generator=rng)
    p = p / p.sum(-1, keepdim=True) * 0.8
    y = torch.randn(2, 32, 4, generator=rng)
    evidence = (
        dict(
            prompt_probabilities=p,
            target=y,
            generated_mass=torch.full((2, 32), 0.2),
            generated_numerator=torch.zeros_like(y),
        ),
    )
    dictionary = (((0, 2, 5), (0, 3, 5)),)
    return source, evidence, dictionary


def serialized(anchor, fitted):
    return [
        dict(
            layer=layer,
            head=h,
            positions=row.logical_positions,
            keys=row.keys,
            original_keys=anchor[layer][h].keys,
            values=row.values,
            log_bias=row.log_bias,
            original_values=anchor[layer][h].values,
        )
        for layer, rows in enumerate(fitted)
        for h, row in enumerate(rows)
    ]


def test_frozen_constructor_and_replay_use_first_16_only_and_preserve_actual_constraints():
    source, evidence, dictionary = fixture()
    saved = copy.deepcopy((source, evidence))
    core = AttentionCompactionPolicy(True)
    anchor, fitted, records = construct_anchor_heads(
        source, evidence, 3, core, (0, 5), fixed_dictionary=dictionary
    )
    changed = copy.deepcopy(evidence)
    changed[0]["target"][:, 16:] += 100
    changed[0]["prompt_probabilities"][:, 16:] = torch.flip(
        changed[0]["prompt_probabilities"][:, 16:], [-1]
    )
    other = construct_anchor_heads(source, changed, 3, core, (0, 5), fixed_dictionary=dictionary)[1]
    assert len(records) == 2
    for h, row in enumerate(fitted[0]):
        assert torch.equal(row.values, other[0][h].values) and torch.equal(
            row.log_bias, other[0][h].log_bias
        )
        assert row.logical_positions == dictionary[0][h]
        assert torch.equal(row.keys, anchor[0][h].keys)
        assert row.diagnostics["coefficient_fit_prefixes"] == 16
        norm = anchor[0][h].values.double().norm(dim=-1)
        assert (
            (row.values.double() - anchor[0][h].values.double()).norm(dim=-1) <= 0.1 * norm
        ).all()
    reviewed = gate.review_coefficients(serialized(anchor, fitted), evidence, core, dictionary)
    assert reviewed["first_16_fit_bitwise_reproduced"]
    assert reviewed["maximum_delta_norm_ratio"] <= 0.1 and reviewed["maximum_value_norm_ratio"] <= 4
    assert all(
        torch.equal(a, b)
        for pair, old in zip(source, saved[0], strict=True)
        for a, b in zip(pair, old, strict=True)
    )
    assert all(torch.equal(evidence[0][key], saved[1][0][key]) for key in evidence[0])
    corrupt = serialized(anchor, fitted)
    corrupt[0]["values"] = corrupt[0]["values"].clone()
    corrupt[0]["values"][1, 0] += 1
    with pytest.raises(ValueError, match=r"coefficients|mandatory"):
        gate.review_coefficients(corrupt, evidence, core, dictionary)


def test_constructor_rejects_profile_changes_and_invalid_mandatory_dictionary():
    source, evidence, dictionary = fixture()
    with pytest.raises(ValueError, match="unchanged"):
        construct_anchor_heads(
            source,
            evidence,
            3,
            AttentionCompactionPolicy(True, value_ridge=0.01),
            (0, 5),
            fixed_dictionary=dictionary,
        )
    with pytest.raises(ValueError, match="mandatory"):
        construct_anchor_heads(
            source,
            evidence,
            3,
            AttentionCompactionPolicy(True),
            (0, 5),
            fixed_dictionary=(((0, 1, 2), (0, 3, 5)),),
        )
    with pytest.raises(TypeError):
        construct_anchor_heads(
            source,
            evidence,
            3,
            AttentionCompactionPolicy(True),
            (0, 5),
            fixed_dictionary=dictionary,
            benchmark_answers=["forbidden"],
        )


def test_frozen_gpu_protocol_rejects_all_scope_and_fit_changes():
    original = yaml.safe_load(gate.PROTOCOL.read_text())
    gate.validate_protocol(original)
    assert original["construction"] == construction_policy()
    for key, value in (
        ("effective_method", "mosaickv_exact"),
        ("workers", 3),
        ("prompt_positions", 40),
        ("cap_bytes", 40 * 2**20),
        ("dictionary", "new_query_ranking"),
        ("calibration_authorized", True),
        ("fresh_cohort_authorized", True),
    ):
        altered = copy.deepcopy(original)
        altered[key] = value
        with pytest.raises(ValueError):
            gate.validate_protocol(altered)
    for key, value in (("coefficient_fit_prefixes", 32), ("component", "value_only")):
        altered = copy.deepcopy(original)
        altered["construction"][key] = value
        with pytest.raises(ValueError, match="frozen"):
            gate.validate_protocol(altered)
    altered = copy.deepcopy(original)
    altered["construction"]["value_policy"]["maximum_delta_norm_ratio"] = 0.2
    with pytest.raises(ValueError, match="frozen"):
        gate.validate_protocol(altered)
    altered = copy.deepcopy(original)
    altered["guard"]["horizon"] = 16
    with pytest.raises(ValueError):
        gate.validate_protocol(altered)


def test_worker_explicitly_selects_v3_and_cannot_silently_run_old_constructor(
    monkeypatch, tmp_path
):
    plan = dict(source_sha="a" * 40)
    monkeypatch.setattr(gate, "validate", lambda *_: plan)
    calls = []
    monkeypatch.setattr(
        gate.base, "execute_worker", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    gate.worker(tmp_path, "b" * 64, 0)
    assert calls[0][1]["method"] == METHOD and calls[0][1]["construction_variant"] == VARIANT
    assert calls[0][1]["plan"] == plan


def test_missing_gpu_results_are_negative_and_write_once(tmp_path, monkeypatch):
    plan = dict(jobs=[dict(index=i, task=f"synthetic-{i}") for i in range(2)], source_sha="a" * 40)
    monkeypatch.setattr(gate, "validate", lambda *_: plan)
    gate.analyze(tmp_path, "b" * 64)
    report = json.loads((tmp_path / "analysis/report.json").read_text())
    assert len(report["errors"]) == 2 and not report["diagnostic_passed"]
    assert not report["benchmark_quality_established"] and report["status"] == "incomplete"
    assert all(report[key] is False for key in gate.BLOCKED)
    with pytest.raises(FileExistsError):
        gate.analyze(tmp_path, "b" * 64)


def test_v3_submission_reuses_two_worker_scheduler_without_watch(monkeypatch, tmp_path):
    calls = []

    def prepare(root):
        plan = dict(source_sha="a" * 40)
        gate.write_once(root / "plan.json", plan)
        return plan

    monkeypatch.setattr(gate, "prepare", prepare)
    monkeypatch.setattr(gate, "validate", lambda *_: None)
    monkeypatch.setattr(
        gate.base, "submit_prepared", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    gate.submit(tmp_path)
    assert calls[0][1]["worker_script"] == "anchor_compaction_hf_v3.sbatch"
    assert calls[0][1]["analysis_script"] == "anchor_compaction_hf_analysis_v3.sbatch"
    gpu = (gate.SOURCE / "mosaickv/slurm/anchor_compaction_hf_v3.sbatch").read_text()
    assert "gpu:a100:1" in gpu and "--cpus-per-task=12" in gpu
    assert "bi_fox_dgx" in gpu and "--time=00:30:00" in gpu
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in gpu and "cache_env.sh" in gpu
