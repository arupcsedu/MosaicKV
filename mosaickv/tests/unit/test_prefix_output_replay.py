"""Synthetic fixtures only; no checkpoint or benchmark-quality evidence."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from mosaickv.backends.hf_prefix_outputs import frozen_query_prediction
from mosaickv.evaluation.head_substitution_diagnostic import HeadDiagnosticPolicy
from mosaickv.evaluation.prefix_output_replay import frozen_trajectories
from mosaickv.selection.head_substitution import HeadSubstitution


def frozen_fixture():
    p = HeadSubstitution(0, 0, 1, 3, 16)
    raw = asdict(p)
    key = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    record = dict(
        policy=asdict(HeadDiagnosticPolicy()),
        selection=dict(accepted=False, events=[dict(proposal_id=key, steps=[dict(step=1)])]),
        controls_passed=True,
        proposals={key: raw},
    )
    return record, key, p


def test_frozen_descriptors_guard_and_prefix_coverage_cannot_change():
    record, key, p = frozen_fixture()
    assert frozen_trajectories(record) == ((key, p, {1: dict(step=1)}),)
    # JSON roundtrip changes tuple checkpoints to lists, not policy semantics.
    assert frozen_trajectories(json.loads(json.dumps(record)))[0][1] == p
    for mutate in (
        lambda r: r["policy"]["guard"].update(absolute_tolerance=1e-3),
        lambda r: r["proposals"][key].update(added_position=4),
        lambda r: r["selection"].update(accepted=True),
        lambda r: r["selection"]["events"][0].update(steps=[dict(step=2)]),
        lambda r: r.update(proposals={}),
    ):
        changed = deepcopy(record)
        mutate(changed)
        with pytest.raises(ValueError):
            frozen_trajectories(changed)


def test_frozen_query_prediction_keeps_prefixes_heads_and_projection_explicit():
    rng = torch.Generator().manual_seed(9)
    p = torch.softmax(torch.randn(2, 64, 4, generator=rng), -1) * 0.8
    v = torch.randn(1, 2, 4, 3, generator=rng)
    data = dict(
        prompt_probabilities=p,
        generated_mass=torch.full((2, 64), 0.2),
        generated_numerator=torch.zeros(2, 64, 3),
        target=torch.randn(2, 64, 3, generator=rng),
    )
    source = SimpleNamespace(
        layers=[SimpleNamespace(value=v)],
        logical_positions=SimpleNamespace(physical_to_logical=(10, 20, 30, 40)),
    )
    anchor = SimpleNamespace(
        slot_records=[
            dict(layer=0, kv_head=h, logical_position=pos) for h in (0, 1) for pos in (10, 20)
        ]
    )
    projection = torch.nn.Linear(12, 5)
    proposal = HeadSubstitution(0, 1, 1, 3, 24)
    before = v.clone()
    out = frozen_query_prediction(source, anchor, proposal, (data,), projection)
    actual = (p[1, :, [0, 3]] @ v[0, 1, [0, 3]]) / (p[1, :, [0, 3]].sum(-1, keepdim=True) + 0.2)
    torch.testing.assert_close(out["predicted_candidate"].flatten(0, 1), actual)
    delta = (out["predicted_candidate"] - out["predicted_anchor"]).flatten(1).double()
    torch.testing.assert_close(
        out["predicted_projected_change_fp64"], delta @ projection.weight[:, 6:12].double().T
    )
    assert out["query_heads_per_kv_head"] == 2 and torch.equal(v, before)
    assert out["numerical_role"].startswith("sideband_")


def test_output_replay_protocol_consumes_no_new_identities_and_opens_no_campaign():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[3]
    protocol = yaml.safe_load(
        (root / "mosaickv/configs/experiments/bounded_prefix_output_v8_1.yaml").read_text()
    )
    assert protocol["parent_source_sha"] == "0eb7a72329c4cdbcd2fd31374febeb94b0312316"
    assert protocol["diagnostic"]["guard"]["horizon"] == 32
    assert protocol["diagnostic"]["guard"]["absolute_tolerance"] == 1e-7
    assert protocol["diagnostic"]["guard"]["relative_tolerance"] == 1e-6
    assert all(
        not protocol[k]
        for k in ("fresh_cohort_authorized", "calibration_authorized", "paper_campaign_authorized")
    )


def test_measured_comparison_does_not_confuse_vector_change_with_improvement():
    from analyze_prefix_output_replay import comparison, cosine

    reference = torch.tensor([0.0, 0.0])
    anchor = torch.tensor([1.0, 1.0])
    candidate = torch.tensor([2.0, 2.0])
    result = comparison(reference, anchor, candidate)
    assert result["improvement"] == -3 and result["observed_change_rms"] == 1
    assert cosine(anchor, -anchor) == pytest.approx(-1.0)
    assert cosine(reference, anchor) is None
    with pytest.raises(ValueError, match="nonfinite"):
        comparison(reference, anchor, candidate * float("nan"))
    with pytest.raises(ValueError, match="geometry"):
        comparison(reference, anchor, torch.zeros(3))
