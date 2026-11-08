"""Admission for a frozen lossless-codec preservation check, not superiority.

The holdout is relative to this codec's two-input development, not to the
project's historical benchmark use. No label or prediction affects selection.
"""

from collections.abc import Mapping, Sequence

from mosaickv.evaluation.fox_campaign import digest


def select_inputs(rows: Sequence[Mapping], excluded: Sequence[Mapping], *, count: int, seed: int):
    if type(count) is not int or not 2 <= count <= 16 or type(seed) is not int or seed < 0:
        raise ValueError("bounded 2--16 input preservation check and explicit seed required")
    seen_ids, seen_media, pool = set(), set(), []
    excluded_ids = {r["sample_id"] for r in excluded}
    excluded_media = {m["sha256"] for r in excluded for m in r["media"]}
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        sid, media = row["sample_id"], row["media"]
        if sid in seen_ids:
            raise ValueError("duplicate provider identity")
        seen_ids.add(sid)
        hashes = tuple(m["sha256"] for m in media)
        if (
            not hashes
            or len(set(hashes)) != len(hashes)
            or any(not isinstance(h, str) or len(h) != 64 for h in hashes)
        ):
            raise ValueError("nonempty immutable media identities required")
        # Do not count aliases as additional independent images.
        if sid in excluded_ids or set(hashes) & (excluded_media | seen_media):
            continue
        seen_media.update(hashes)
        pool.append(row)
    pool.sort(key=lambda r: (digest([seed, r["sample_id"]]), r["sample_id"]))
    if len(pool) < count:
        raise ValueError("insufficient codec-development-disjoint identities")
    # Copy only request inputs. References/answers never enter worker inputs.
    return [{k: r[k] for k in ("sample_id", "prompt", "media")} for r in pool[:count]]


def assess_pair(rows):
    """Report distinct validity/memory gates; allocator bounds are never exact KV."""
    expected = ("storage_only_common", "storage_only_complete_source_replacement")
    if len(rows) != 2 or {r["arm"] for r in rows} != set(expected):
        raise ValueError("one unique result per registered arm required")
    by_arm = {r["arm"]: r for r in rows}
    reference, candidate = (by_arm[a] for a in expected)
    if reference["sample_id"] != candidate["sample_id"]:
        raise ValueError("paired identities differ")
    errors = []
    for row in rows:
        if row.get("status") != "completed":
            errors.append(row["arm"] + ":failed_or_incomplete")
            continue
        detail, codec, envelope = row["detail"], row["codec"], row["whole_request_envelope"]
        if (
            not isinstance(row.get("logits_sha256"), str)
            or len(row["logits_sha256"]) != 64
            or row.get("logits_shape", [])[:1] != [128]
            or len(row.get("logits_shape", [])) != 2
            or type(row["logits_shape"][1]) is not int
            or row["logits_shape"][1] < 1
        ):
            errors.append(row["arm"] + ":incomplete_logit_evidence")
        if detail.get("used_fullkv_fallback") is not False:
            errors.append(row["arm"] + ":fallback")
        if (
            detail.get("tensor_result_auditing") is not True
            or detail.get("peak_measurement_admissible") is not True
        ):
            errors.append(row["arm"] + ":unaudited_decode")
        if (
            codec.get("all_shared_source_consumers_covered") is not True
            or len(codec["records"]) != 32
        ):
            errors.append(row["arm"] + ":incomplete_shared_source_coverage")
        if envelope.get("kv_peak_upper_bound_certified") is not True:
            errors.append(row["arm"] + ":incomplete_request_envelope")
        if envelope.get("exact_kv_peak_certified") is not False:
            errors.append(row["arm"] + ":unsupported_exact_peak_claim")
        for k in ("initial_encoded_bytes", "final_encoded_bytes", "decode_owned_peak_bytes"):
            if type(detail.get(k)) is not int or detail[k] <= 0:
                errors.append(row["arm"] + ":invalid_" + k)
        mechanisms = row["mechanism_evidence"]
        if mechanisms.get("extra_gpu_bytes") != 0 or mechanisms.get("residual_cpu_bytes") != 0:
            errors.append(row["arm"] + ":unregistered_overlay")
    if candidate.get("status") == "completed":
        codec = candidate["codec"]
        replacements = codec.get("additive_replacements", 0)
        if not replacements or codec.get("additive_original_owners_retired") != replacements:
            errors.append("complete_source_owner_retirement_failed")
        if reference.get("status") == "completed" and (
            candidate.get("all_128_logits_bitwise") is not True
            or candidate.get("maximum_logit_difference") != 0.0
            or candidate.get("token_agreement") != 1.0
            or candidate.get("logits_sha256") != reference.get("logits_sha256")
            or candidate.get("logits_shape") != reference.get("logits_shape")
            or candidate.get("logits_dtype") != reference.get("logits_dtype")
            or candidate["detail"]["answer"] != reference["detail"]["answer"]
        ):
            errors.append("parent_computation_or_answer_changed")
    memory = {}
    if not errors:
        for k in ("initial_encoded_bytes", "final_encoded_bytes", "decode_owned_peak_bytes"):
            a, b = reference["detail"][k], candidate["detail"][k]
            memory[k] = dict(parent=a, candidate=b, saved_bytes=a - b, reduction=1 - b / a)
    return dict(
        valid=not errors,
        errors=errors,
        parent_computation_preserved=not errors,
        smaller_declared_KV_boundaries=not errors
        and all(m["saved_bytes"] > 0 for m in memory.values()),
        memory=memory,
        exact_whole_request_KV_peak_certified=False,
        new_benchmark_quality_improvement_established=False,
        strict_latency_pass_established=False,
        six_component_claim_supported=False,
        paper_campaign_authorized=False,
    )
