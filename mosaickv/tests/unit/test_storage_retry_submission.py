"""Offline scheduler argument tests; never submit a real Slurm job."""

import evicastkv_lifetime_v2 as launch
import pytest


@pytest.mark.parametrize("indices,spec", [(None, "--array=0-1%2"), ([1], "--array=1%1")])
def test_selective_retry_indices_keep_one_gpu_and_afterany_analysis(
    tmp_path, monkeypatch, indices, spec
):
    calls = []
    monkeypatch.setattr(
        launch.previous, "inputs", lambda *_args: {"jobs": [{"index": 0}, {"index": 1}]}
    )

    def submit(args, **_kwargs):
        calls.append(args)
        return str(100 + len(calls))

    monkeypatch.setattr(launch.subprocess, "check_output", submit)
    monkeypatch.setattr(launch, "tick", lambda *_args: {"synthetic_registration_fixture": True})
    launch.submit(tmp_path, "a" * 64, worker_indices=indices)
    assert spec in calls[0]
    assert "--gres=gpu:a100:1" in calls[0]
    assert "--cpus-per-task=12" in calls[0]
    assert "--dependency=afterany:101" in calls[1]


@pytest.mark.parametrize("indices", [[], [1, 1], [2], ["1"]])
def test_bad_retry_indices_fail_before_submission(tmp_path, monkeypatch, indices):
    monkeypatch.setattr(
        launch.previous, "inputs", lambda *_args: {"jobs": [{"index": 0}, {"index": 1}]}
    )
    monkeypatch.setattr(
        launch.subprocess, "check_output", lambda *_a, **_k: pytest.fail("submitted")
    )
    with pytest.raises(ValueError, match="worker indices"):
        launch.submit(tmp_path, "a" * 64, worker_indices=indices)


def test_sequence_length_does_not_create_constexpr_variants():
    from mosaickv.evaluation.lossless_bf16_kernel import _decode, _encode_headers, _encode_literals

    assert {_decode.arg_names[i] for i in _decode.constexprs} == {"DIM", "BLOCK"}
    for function in (_encode_headers, _encode_literals):
        assert {function.arg_names[i] for i in function.constexprs} == {"BLOCK"}
