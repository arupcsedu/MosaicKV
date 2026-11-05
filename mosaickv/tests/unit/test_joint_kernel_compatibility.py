import ctypes as C
from types import SimpleNamespace

import pytest

from mosaickv.evaluation import joint_kernel_compatibility as study
from mosaickv.evaluation.zero_workspace_lt import Algo


def configuration(index):
    return dict(
        algorithm_id=index,
        tile=15,
        stages=18,
        custom=0,
        swizzle=0,
        split_k=1,
        reduction=0,
        inner=4,
        numerical_flags=0x20200,
    )


def rows():
    return [
        dict(
            step=s,
            component=c,
            kernels=[
                dict(
                    configuration=configuration(i),
                    incoming_row=dict(bitwise_equal=True),
                    prompt_rows=dict(bitwise_equal=True),
                )
                for i in (0, 1)
            ],
        )
        for s in study.original.WINDOWS
        for c in (0, 1)
    ]


def test_every_registered_enum_is_checked_against_installed_header_text():
    text = "\n".join(f"{name} = {value}," for name, value in study.ENUMS.items())
    study.verify_header(text)
    for name in study.ENUMS:
        with pytest.raises(RuntimeError, match="unverified"):
            study.verify_header(text.replace(name, "not_an_enum"))


def test_common_configuration_must_match_both_outputs_at_every_window():
    data = rows()
    assert len(study.common_exact(data)) == 2
    data[0]["kernels"][0]["incoming_row"]["bitwise_equal"] = False
    data[-1]["kernels"][1]["prompt_rows"]["bitwise_equal"] = False
    assert study.common_exact(data) == []


@pytest.mark.parametrize("kind", ["missing", "duplicate", "empty", "too_many"])
def test_no_partial_or_duplicate_configuration_evidence_is_accepted(kind):
    data = rows()
    if kind == "missing":
        data.pop()
    elif kind == "duplicate":
        data[0]["kernels"].append(data[0]["kernels"][0])
    elif kind == "empty":
        data[0]["kernels"].clear()
    else:
        data[0]["kernels"] = [dict(configuration=configuration(i)) for i in range(33)]
    with pytest.raises(ValueError):
        study.common_exact(data)


def test_config_fingerprint_cannot_hide_stage_custom_or_inner_differences():
    record = configuration(0)
    for field in record.keys() - {"numerical_flags"}:
        assert study.fingerprint(record) != study.fingerprint(
            dict(record, **{field: record[field] + 1})
        )


@pytest.mark.parametrize("status,workspace,expected", [(0, 0, True), (7, 0, False), (0, 1, False)])
def test_vendor_check_and_zero_workspace_are_required_for_both_lengths(status, workspace, expected):
    state = object.__new__(study.KernelPlans)
    state.candidates, state.rejections, state.handle, state.operation = [], [], None, None
    state.plans = {88: (None, None, None, None, None), 89: (None, None, None, None, None)}
    state.describe = lambda algo: configuration(0)
    checks = []

    def check(*args):
        result = C.cast(args[-1], C.POINTER(study.Heuristic)).contents
        result.workspaceSize = workspace
        checks.append(1)
        return status if len(checks) == 2 else 0

    state.library = SimpleNamespace(lib=SimpleNamespace(cublasLtMatmulAlgoCheck=check))
    state.consider(Algo(), "fixture")
    assert len(checks) == 2
    assert bool(state.candidates) is expected
    assert bool(state.rejections) is not expected


def test_configuration_swap_is_restored_even_if_actual_execution_fails():
    state = object.__new__(study.KernelPlans)
    original = (None, None, None, Algo(), {"original": True})
    state.plans = {88: original}

    def fail(*args):
        assert state.plans[88][-1] == {"new": True}
        raise RuntimeError("execution failed")

    state.execute = fail
    with pytest.raises(RuntimeError, match="execution failed"):
        state.run_configuration(None, None, None, 88, Algo(), {"new": True})
    assert state.plans[88] is not original
    assert state.plans[88] == original


def test_controller_keeps_no_runtime_and_no_quality_claim(monkeypatch, tmp_path):
    import evicastkv_joint_kernel_compatibility as controller

    seen = []
    monkeypatch.setattr(controller.parent.driver, "checked", lambda *a: {"source_sha": "sha"})
    monkeypatch.setattr(
        controller.parent.driver,
        "load_producers",
        lambda *a: (
            [
                dict(
                    records=[
                        dict(
                            common_exact_configurations=[list(study.fingerprint(configuration(0)))]
                        )
                    ]
                )
            ]
            * 2,
            [],
        ),
    )
    monkeypatch.setattr(controller, "write_once", lambda p, d: seen.append(d))
    controller.analyze(tmp_path, "plan")
    assert seen[0]["same_parent_integration_eligible_for_further_operator_checks"]
    assert seen[0]["no_runtime_integrated"] and seen[0]["no_latency_or_quality_acceptance"]
    assert not seen[0]["automatic_followup"]
