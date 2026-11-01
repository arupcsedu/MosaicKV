import pytest
from evicastkv_common_streaming import optimized_controls_pass

from mosaickv.evaluation.common_streaming_timing import METHODS, request


def evidence():
    jobs, cases = [], []
    for i in range(2):
        for method in METHODS:
            index = len(jobs)
            jobs.append(
                dict(
                    input_index=i,
                    evaluation_variant="source3_direct_fp32"
                    if method == "mosaickv_exact"
                    else f"baseline_{method}__fp32",
                )
            )
            cases.append(
                dict(
                    case=index,
                    observations=dict(
                        payload_bitwise=True,
                        prefill_logits_bitwise=True,
                        physical_packet_bytes_identical=True,
                        memory=dict(
                            tensor_results_observed=True,
                            original_KV_tensor_objects_alive=0,
                            observed_prefill_cuda_peak_bytes=10
                            if method == "mosaickv_exact"
                            else 20,
                        ),
                    ),
                )
            )
    return dict(complete=True, failures=[], cases=cases), jobs


def test_timing_requires_complete_identity_and_strictly_lower_constructor_peak():
    proof, jobs = evidence()
    assert optimized_controls_pass(proof, jobs)
    proof["cases"][0]["observations"]["memory"]["observed_prefill_cuda_peak_bytes"] = 20
    assert not optimized_controls_pass(proof, jobs)
    proof, jobs = evidence()
    proof["cases"][4]["observations"]["payload_bitwise"] = False
    assert not optimized_controls_pass(proof, jobs)
    proof, jobs = evidence()
    proof["cases"].pop()
    assert not optimized_controls_pass(proof, jobs)


@pytest.mark.parametrize(
    "kwargs", [dict(componentwise_fitting=1), dict(code_delta=1), dict(input_packet_validator=1)]
)
def test_bad_execution_policy_is_rejected_before_cuda(kwargs):
    with pytest.raises(ValueError):
        request(None, None, (), "mosaickv_exact", **kwargs)


def test_continuation_cannot_widen_scope(monkeypatch, tmp_path):
    import evicastkv_common_streaming as module

    monkeypatch.setattr(module.previous.common, "inputs", lambda *a: {})
    with pytest.raises(ValueError, match="only the registered"):
        module.register_optimized_timing(tmp_path, "plan")
