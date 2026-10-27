"""Small CPU-only analytic fixtures; no checkpoint results or fitted weights."""

import math
from dataclasses import asdict

import pytest
import torch

from mosaickv.evaluation.readout_audit import (
    PROSPECTIVE_SCORES,
    SCORE_PROVENANCE,
    FrozenReadout,
    ReadoutAuditPolicy,
    audit_ranking,
    kl_divergence,
    projected_residual_gain,
    rank_correlation,
)


def fixture():
    rng = torch.Generator().manual_seed(41)
    norm = torch.rand(5, generator=rng) + 0.5
    weight = torch.randn(7, 5, generator=rng)
    readout = FrozenReadout(norm, weight, 1e-5)
    full = torch.randn(5, generator=rng)
    anchor = full + torch.randn(5, generator=rng) * 0.1
    delta = torch.randn(5, generator=rng) * 0.01
    return readout, full, anchor, delta


def test_frozen_readout_matches_rmsnorm_and_linear_formula_and_keeps_weights_fixed():
    readout, full, anchor, _ = fixture()
    norm, weight = readout.norm.clone(), readout.weight.clone()
    states = torch.stack([full, anchor])
    expected = (
        states.double() * (states.double().square().mean(-1, keepdim=True) + 1e-5).rsqrt() * norm
    )
    torch.testing.assert_close(readout.normalize(states), expected, rtol=0, atol=0)
    torch.testing.assert_close(readout.logits(states), expected @ weight.T, rtol=0, atol=0)
    assert not readout.weight.requires_grad and not readout.norm.requires_grad
    assert torch.equal(readout.norm, norm) and torch.equal(readout.weight, weight)


def test_analytic_kl_gradient_and_directional_score_match_finite_difference():
    readout, full, anchor, delta = fixture()
    z = readout.logits(full)
    h = anchor.double()
    step = 1e-5
    finite = (
        kl_divergence(z, readout.logits(h + step * delta))
        - kl_divergence(z, readout.logits(h - step * delta))
    ) / (2 * step)
    gradient = readout.kl_gradient(z, h)
    linear, quadratic = readout.directional_gains(z, h, delta)
    torch.testing.assert_close(gradient @ delta.double(), finite, rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(linear, -finite, rtol=1e-6, atol=1e-10)
    assert quadratic <= linear + 1e-12
    jvp = (readout.normalize(h + step * delta) - readout.normalize(h - step * delta)) / (2 * step)
    torch.testing.assert_close(readout.norm_jvp(h, delta), jvp, rtol=1e-6, atol=1e-10)


def test_directional_batch_preserves_per_prefix_observations():
    readout, full, anchor, delta = fixture()
    states = torch.stack([anchor, anchor * 1.1])
    change = torch.stack([delta, delta * 0.3])
    z = torch.stack([readout.logits(full), readout.logits(full * 0.8)])
    batched = readout.directional_gains(z, states, change)
    separate = [readout.directional_gains(z[i], states[i], change[i]) for i in range(2)]
    for j in (0, 1):
        torch.testing.assert_close(batched[j], torch.stack([s[j] for s in separate]))


def test_residual_directional_gain_accounts_for_error_cancellation():
    full = torch.tensor([0.0, 0.0])
    anchor = torch.tensor([1.0, 0.0])
    closer = torch.tensor([-0.25, 0.0])
    orthogonal = torch.tensor([0.0, 0.25])
    assert projected_residual_gain(full, anchor, closer) > 0
    assert projected_residual_gain(full, anchor, orthogonal) < 0


def test_readout_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        FrozenReadout(torch.ones(2), torch.ones(3, 4), 1e-5)
    with pytest.raises(ValueError):
        FrozenReadout(torch.ones(2), torch.ones(3, 2), float("nan"))
    readout, full, _, _ = fixture()
    with pytest.raises(ValueError):
        readout.normalize(torch.ones(6))
    with pytest.raises(ValueError):
        readout.normalize(full * float("nan"))


def test_ranks_handle_ties_constants_and_nonfinite_values():
    assert rank_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert rank_correlation([1, 1, 2], [2, 2, 3]) == pytest.approx(1)
    assert rank_correlation([1, 1], [2, 3]) is None
    with pytest.raises(ValueError):
        rank_correlation([1, 2], [1, math.nan])


def test_post_probe_features_cannot_qualify_as_prospective_signal():
    rows = []
    for i in range(16):
        # Perfect post-probe score, but every pre-probe score ranks in reverse.
        rows.append(
            dict(
                prefix=1,
                registered_kl_gain=(i + 1) * 1e-4,
                registered_kl_before=0.1,
                **{name: 16 - i for name in PROSPECTIVE_SCORES},
                actual_readout_linear=i + 1,
            )
        )
    report = audit_ranking(rows, ReadoutAuditPolicy())
    assert set(report) == set(PROSPECTIVE_SCORES)
    assert not any(r["descriptive_signal"] for r in report.values())
    assert SCORE_PROVENANCE["actual_readout_linear"].startswith("post_probe")
    with pytest.raises(ValueError, match="prefix one"):
        audit_ranking([dict(rows[0], prefix=2)], ReadoutAuditPolicy())


def test_fixed_screen_can_identify_a_descriptive_not_promotion_signal():
    rows = []
    for i in range(16):
        rows.append(
            dict(
                prefix=1,
                registered_kl_gain=(i + 1) * 1e-4,
                registered_kl_before=0.1,
                **{name: i + 1 for name in PROSPECTIVE_SCORES if name != "head_reconstruction"},
                head_reconstruction=16 - i,
            )
        )
    report = audit_ranking(rows, ReadoutAuditPolicy())
    assert report["projected_residual"]["descriptive_signal"]
    assert not report["head_reconstruction"]["descriptive_signal"]
    assert asdict(ReadoutAuditPolicy())["minimum_positive_precision"] == 0.75


def test_complete_task_audit_preserves_unexecuted_prefixes_as_predictions_only(tmp_path):
    import json

    from audit_readout_v8_2 import task_audit

    readout, full, anchor, delta = fixture()
    worker = tmp_path / "workers/0"
    worker.mkdir(parents=True)
    destination = tmp_path / "derived"
    destination.mkdir()
    outputs = {"fullkv": {}, "anchor": {}, "p1": {}, "p2": {}}
    for step in range(1, 33):
        for role, h in [("fullkv", full), ("anchor", anchor)]:
            outputs[role][step] = (dict(layer_output=h.clone(), post_attention_residual=h.clone()),)
    predictions = {}
    replay = []
    logits = {
        role: {
            step: readout.logits(layers[-1]["layer_output"]).unsqueeze(0)
            for step, layers in steps.items()
        }
        for role, steps in outputs.items()
    }
    for key, sign, count in [("p1", 1, 2), ("p2", -1, 1)]:
        target = full.repeat(32, 1, 1)
        before = anchor.repeat(32, 1, 1)
        after = (anchor + sign * delta).repeat(32, 1, 1)
        predictions[key] = dict(
            fullkv_query_target=target,
            predicted_anchor=before,
            predicted_candidate=after,
            predicted_projected_change_fp64=(sign * delta).repeat(32, 1).double(),
        )
        steps = []
        for step in range(1, count + 1):
            ch = anchor + sign * delta
            outputs[key][step] = (
                dict(layer_output=ch.clone(), post_attention_residual=ch.clone()),
            )
            logits[key][step] = readout.logits(ch).unsqueeze(0)
            a = float(kl_divergence(readout.logits(full), readout.logits(anchor)))
            c = float(kl_divergence(readout.logits(full), readout.logits(ch)))
            steps.append(
                dict(
                    step=step,
                    anchor=dict(logit_kl=a, maximum_logit_difference=1.0),
                    candidate=dict(
                        logit_kl=c, maximum_logit_difference=1.0, greedy_token_agreement=True
                    ),
                )
            )
        replay.append(dict(proposal_id=key, proposal=dict(layer=0), steps=steps))
    (worker / "diagnostic.json").write_text(
        json.dumps(dict(result=dict(replay=replay, accepted_proposals=0)))
    )
    (worker / "manifest.json").write_text(json.dumps(dict(dataset=dict(id="synthetic_fixture"))))
    torch.save(outputs, worker / "prefix-outputs.pt")
    torch.save(predictions, worker / "frozen-query-predictions.pt")
    torch.save(logits, worker / "logits.pt")
    report, observed, prospective, checks = task_audit(tmp_path, 0, readout, destination)
    assert report["observed_proposal_prefixes"] == 3 and report["proposals"] == 2
    assert len(prospective) == 64 and len(observed) == 3
    assert report["sideband_vs_GPU_logit_observations"] == 67
    assert len(checks) == 67
    # Batched versus single-vector CPU BLAS may differ at FP64 roundoff.
    # This synthetic assertion does not modify any registered GPU guard.
    assert report["sideband_vs_GPU_logit_maximum_difference"] < 1e-12
    assert report["no_new_forward_pass"] and not report["benchmark_labels_accessed"]
    assert {r["evidence_horizon"] for r in observed} == {32}
    assert (destination / "readout-sideband-0.pt").is_file()


def test_protocol_cannot_change_fidelity_or_downstream_permissions():
    from pathlib import Path

    import yaml
    from audit_readout_v8_2 import validate_protocol

    path = (
        Path(__file__).resolve().parents[3]
        / "mosaickv/configs/experiments/bounded_readout_v8_2.yaml"
    )
    protocol = yaml.safe_load(path.read_text())
    validate_protocol(protocol)
    for key in ("fresh_cohort_authorized", "calibration_authorized", "paper_campaign_authorized"):
        with pytest.raises(ValueError):
            validate_protocol(dict(protocol, **{key: True}))
    with pytest.raises(ValueError):
        validate_protocol(
            dict(
                protocol,
                registered_guard=dict(horizon=32, absolute_tolerance=1e-3, relative_tolerance=1e-6),
            )
        )
