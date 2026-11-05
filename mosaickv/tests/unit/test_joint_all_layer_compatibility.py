from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import joint_all_layer_compatibility as study
from mosaickv.evaluation.joint_kernel_compatibility import ENUMS


def rows():
    return [
        dict(
            layer=layer,
            step=step,
            component=component,
            kernels=[
                dict(
                    configuration=dict(
                        algorithm_id=31,
                        tile=15,
                        split_k=1,
                        reduction=0,
                        swizzle=0,
                        custom=0,
                        stages=18,
                        inner=4,
                    ),
                    incoming_row=dict(bitwise_equal=True),
                    prompt_rows=dict(bitwise_equal=True),
                )
            ],
        )
        for layer in range(32)
        for step in study.WINDOWS
        for component in (0, 1)
    ]


def test_all_192_observations_and_the_frozen_kernel_are_required():
    data = rows()
    assert study.admissible(data)
    data[-1]["kernels"][0]["configuration"]["inner"] = 0
    assert not study.admissible(data)


@pytest.mark.parametrize("field", ["incoming_row", "prompt_rows"])
def test_one_word_failure_in_any_layer_is_not_averaged_away(field):
    data = rows()
    data[128]["kernels"][0][field]["bitwise_equal"] = False
    assert not study.admissible(data)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_layer"])
def test_missing_layer_or_prefix_fails_closed(change):
    data = rows()
    if change == "missing":
        data.pop()
    elif change == "duplicate":
        data[-1] = data[0]
    else:
        data[-1]["layer"] = 32
    with pytest.raises(ValueError, match="every layer"):
        study.admissible(data)


def test_all_layer_hooks_return_original_outputs_and_restore_after_failure(monkeypatch):
    attentions = [
        SimpleNamespace(k_proj=torch.nn.Linear(4, 4), v_proj=torch.nn.Linear(4, 4))
        for _ in range(2)
    ]
    hidden = torch.zeros(1, 1, 4)

    def step():
        return [a.k_proj(hidden) for a in attentions]

    adapter = SimpleNamespace(
        decode_one_token=step,
        _language_layers=lambda: [SimpleNamespace(self_attn=a) for a in attentions],
    )
    state, seen = study.Probe(adapter), []
    monkeypatch.setattr(state, "observe_layer", lambda *args: seen.append(args[:2]))
    expected = step()
    with pytest.raises(RuntimeError, match="stop"), state.installed():
        assert study.current() is state
        assert all(torch.equal(a, b) for a, b in zip(step(), expected, strict=True))
        raise RuntimeError("stop")
    assert seen == [(0, 0), (1, 0)] and study.current() is None
    assert adapter.decode_one_token is step
    assert all(not a.k_proj._forward_hooks for a in attentions)


def test_frozen_geometry_matches_verified_installed_enum_values():
    assert study.FROZEN[1] == ENUMS["CUBLASLT_MATMUL_TILE_64x64"]
    assert study.FROZEN[-2] == ENUMS["CUBLASLT_MATMUL_STAGES_64x6"]
    assert study.FROZEN[-1] == ENUMS["CUBLASLT_MATMUL_INNER_SHAPE_MMA16816"]


@pytest.mark.parametrize("allowed", [True, False])
def test_no_all_layer_job_is_prepared_without_positive_frozen_operator_evidence(
    monkeypatch, tmp_path, allowed
):
    import evicastkv_joint_all_layer_compatibility as controller

    monkeypatch.setattr(
        controller.parent.driver,
        "read",
        lambda *args: dict(
            complete=True,
            failures=[],
            common_exact_configurations_both_inputs=[list(study.FROZEN)] if allowed else [],
        ),
    )
    seen = []
    monkeypatch.setattr(controller, "ORIGINAL_PREPARE", lambda *args: seen.append(args))
    if allowed:
        controller.prepare(tmp_path, "source")
        assert seen == [(tmp_path, "source")]
    else:
        with pytest.raises(ValueError, match="positive frozen"):
            controller.prepare(tmp_path, "source")
        assert not seen
