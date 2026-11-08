"""Synthetic admission fixtures, never measured experiment values."""

import copy

import pytest

from mosaickv.evaluation.owner_preservation import assess_pair, select_inputs


class NoLabelAccess(dict):
    def __getitem__(self, key):
        if key not in ("sample_id", "prompt", "media"):
            raise AssertionError("label access is forbidden")
        return super().__getitem__(key)


def inputs():
    return [
        NoLabelAccess(
            sample_id=str(i),
            prompt="synthetic CI only",
            media=[dict(sha256=f"{i:064x}", path=f"/scratch/fixture/{i}")],
        )
        for i in range(20)
    ]


def test_deterministic_id_only_selection_excludes_ids_and_underlying_media():
    pool = inputs()
    alias = dict(pool[3], sample_id="media_alias")
    pool.append(NoLabelAccess(alias))
    excluded = [dict(pool[3], sample_id="old_other_id"), pool[1]]
    a = select_inputs(pool, excluded, count=16, seed=91)
    b = select_inputs(pool[::-1], excluded, count=16, seed=91)
    assert a == b
    assert len(a) == 16
    assert not {"1", "3", "media_alias"} & {r["sample_id"] for r in a}
    assert all(set(r) == {"sample_id", "prompt", "media"} for r in a)


def test_aliases_do_not_inflate_available_sample_count():
    pool = [dict(inputs()[0], sample_id=str(i)) for i in range(20)]
    with pytest.raises(ValueError, match="insufficient"):
        select_inputs(pool, [], count=16, seed=0)
    with pytest.raises(ValueError, match="duplicate"):
        select_inputs([inputs()[0]] * 2, [], count=2, seed=0)


@pytest.mark.parametrize("count", [0, 1, 17, True])
def test_invalid_scope_cannot_expand_to_campaign(count):
    with pytest.raises(ValueError, match="bounded"):
        select_inputs(inputs(), [], count=count, seed=0)


def pair():
    return [
        dict(
            sample_id="synthetic_CI_fixture",
            arm=arm,
            status="completed",
            logits_shape=[128, 32],
            logits_dtype="torch.float32",
            logits_sha256="a" * 64,
            all_128_logits_bitwise=True,
            maximum_logit_difference=0.0,
            token_agreement=1.0,
            detail=dict(
                used_fullkv_fallback=False,
                tensor_result_auditing=True,
                peak_measurement_admissible=True,
                initial_encoded_bytes=100 - a,
                final_encoded_bytes=200 - a,
                decode_owned_peak_bytes=300 - a,
                answer=dict(answer="synthetic"),
            ),
            codec=dict(
                all_shared_source_consumers_covered=True,
                records=[{}] * 32,
                additive_replacements=31,
                additive_original_owners_retired=31,
            ),
            mechanism_evidence=dict(extra_gpu_bytes=0, residual_cpu_bytes=0),
            whole_request_envelope=dict(
                kv_peak_upper_bound_certified=True, exact_kv_peak_certified=False
            ),
        )
        for a, arm in enumerate(("storage_only_common", "storage_only_complete_source_replacement"))
    ]


def test_valid_preservation_never_upgrades_peak_latency_or_six_mechanism_claims():
    result = assess_pair(pair())
    assert result["valid"] and result["parent_computation_preserved"]
    assert result["smaller_declared_KV_boundaries"]
    for key in (
        "exact_whole_request_KV_peak_certified",
        "new_benchmark_quality_improvement_established",
        "strict_latency_pass_established",
        "six_component_claim_supported",
        "paper_campaign_authorized",
    ):
        assert result[key] is False


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"maximum_logit_difference": 1e-8}, "parent_computation_or_answer_changed"),
        ({"all_128_logits_bitwise": False}, "parent_computation_or_answer_changed"),
        (
            {"logits_sha256": None},
            "storage_only_complete_source_replacement:incomplete_logit_evidence",
        ),
        (
            {"logits_shape": [127, 32]},
            "storage_only_complete_source_replacement:incomplete_logit_evidence",
        ),
        ({"status": "failed"}, "storage_only_complete_source_replacement:failed_or_incomplete"),
    ],
)
def test_failed_numerical_or_missing_evidence_cannot_pass(change, expected):
    data = pair()
    data[1].update(change)
    result = assess_pair(data)
    assert not result["valid"]
    assert expected in result["errors"]


@pytest.mark.parametrize(
    "invalid",
    ["fallback", "unaudited", "retained_owner", "overlay", "missing_envelope", "unsupported_peak"],
)
def test_ownership_accounting_and_scope_fail_closed(invalid):
    data = copy.deepcopy(pair())
    if invalid == "fallback":
        data[1]["detail"]["used_fullkv_fallback"] = True
    if invalid == "unaudited":
        data[1]["detail"]["tensor_result_auditing"] = False
    if invalid == "retained_owner":
        data[1]["codec"]["additive_original_owners_retired"] = 30
    if invalid == "overlay":
        data[1]["mechanism_evidence"]["residual_cpu_bytes"] = 1
    if invalid == "missing_envelope":
        data[1]["whole_request_envelope"]["kv_peak_upper_bound_certified"] = False
    if invalid == "unsupported_peak":
        data[1]["whole_request_envelope"]["exact_kv_peak_certified"] = True
    assert not assess_pair(data)["valid"]


def test_valid_but_larger_representation_remains_negative_memory_result():
    data = pair()
    data[1]["detail"]["initial_encoded_bytes"] = 101
    result = assess_pair(data)
    assert result["valid"] and result["parent_computation_preserved"]
    assert not result["smaller_declared_KV_boundaries"]
    assert result["memory"]["initial_encoded_bytes"]["saved_bytes"] == -1


def test_duplicate_or_mismatched_pairs_rejected():
    with pytest.raises(ValueError, match="unique"):
        assess_pair([pair()[0], pair()[0]])
    data = pair()
    data[1]["sample_id"] = "other"
    with pytest.raises(ValueError, match="identities"):
        assess_pair(data)
