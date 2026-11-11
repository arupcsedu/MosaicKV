"""V26 synthetic CPU orchestration fixtures, never pretrained/GPU evidence."""

import json
import sys
from types import SimpleNamespace

import evicastkv_bounded_reader_v26 as study
import evicastkv_review_source_prefix as fixed
import pytest
import torch

ARMS = ("original_source_bounded", "raw_topk_bounded", "raw_retention_1_bounded")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def allocator_stats(monkeypatch, *, extra_by_observation=(128, 128, 128)):
    """Run the REAL ledger on explicit synthetic snapshots, never fake totals."""
    from mosaickv.evaluation import kv_allocator_ledger as allocation

    requested = (1024, 2304, 2048)
    current = {"index": 0}

    def snapshot():
        index = current["index"]
        count = requested[index]
        rounded = count + extra_by_observation[index]
        return [
            dict(
                device=0,
                address=16,
                total_size=rounded,
                is_expandable=True,
                blocks=[
                    dict(address=16, size=rounded, requested_size=count, state="active_allocated")
                ],
            )
        ]

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        memory=SimpleNamespace(get_allocator_backend=lambda: "native"),
        memory_snapshot=snapshot,
    )
    with monkeypatch.context() as patch:
        patch.setattr(allocation, "torch", SimpleNamespace(device=torch.device, cuda=fake_cuda))
        patch.setattr(
            allocation,
            "tensor_owners",
            lambda _tensors, *, device: ((16, requested[current["index"]]),),
        )
        for variable in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            patch.setenv(variable, study.POLICY["allocator_configuration"])
        ledger = allocation.AllocatorLedger(device="cuda:0")
        for index, phase in enumerate(("initial", "growth_peak", "final")):
            current["index"] = index
            ledger.observe([], phase=phase)
        stats = ledger.summary()
        allocation.verify_ownership_deltas(stats["ownership_deltas"], expected_summary=stats)
    return stats


def seal_allocator(out, arm, stats):
    artifact = f"traces/{arm}.allocator_ownership.json"
    deltas = stats.pop("ownership_deltas")
    write(out / artifact, dict(ownership_deltas=deltas))
    stats["ownership_delta_artifact"] = artifact
    stats["ownership_delta_sha256"] = study.sha(out / artifact)
    return stats


@pytest.fixture
def cohort(tmp_path, monkeypatch, request):
    """Build complete, hashed two-worker outputs with real 128-by-5 tensors."""
    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "avx2")
    prior_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(prior_threads))
    torch.set_num_threads(12)
    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
    plan = dict(
        source_sha="a" * 40,
        policy=study.POLICY,
        policy_sha256=study.digest(study.POLICY),
        bindings={},
        expected_workers=2,
    )
    write(tmp_path / "plan.json", plan)
    monkeypatch.setattr(study, "clean_source", lambda *_args: plan["source_sha"])
    write(
        tmp_path / "operator/result.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=study.sha(tmp_path / "plan.json"),
            operator=dict(
                all_bitwise_passed=True,
                allocator=dict(snapshot_consistency_passed=True, overhead_budget_passed=True),
            ),
            failures=[],
        ),
    )
    reference = torch.zeros(128, 5)
    reference[:, 0], reference[:, 1] = 2, 1
    for index in range(2):
        out = tmp_path / f"workers/{index}"
        out.mkdir(parents=True)
        saved = dict(
            FullKV_reference=reference.clone(),
            raw_native_same_selected=reference.roll(1, -1),
            original_source_bounded=reference.clone(),
            raw_topk_bounded=reference.roll(1, -1),
            raw_retention_1_bounded=reference.clone(),
        )
        torch.save(saved, out / "logits.pt")
        rows = []
        for arm in ARMS:
            storage = dict(
                initial_active_kv_storage_bytes=1024,
                final_active_kv_storage_bytes=2048,
                peak_active_kv_storage_bytes=2304,
                history_updates_per_layer=[127] * 32,
                outstanding_incoming_KV=0,
                full_layer_KV_materializations=0,
                attention_calls_per_layer=[127] * 32,
                workspace_bytes=256,
                observations=3,
                allocator_blocks=seal_allocator(out, arm, allocator_stats(monkeypatch)),
            )
            row = dict(
                arm=arm,
                sample_id=f"synthetic-consumed-{index}",
                task="synthetic_only",
                source_sha=plan["source_sha"],
                plan_sha256=study.sha(tmp_path / "plan.json"),
                status="completed",
                fullkv_fallback=False,
                parity=dict(
                    prediction_agreement=128,
                    predictions=128,
                    maximum_logit_difference=0.0,
                    bitwise_logit_parity=True,
                    pass_128=True,
                ),
                measured=dict(
                    storage=storage,
                    predictions=128,
                    used_fullkv_fallback=False,
                    logical_positions_preserved=True,
                    original_KV_values_preserved=True,
                    decode_wall_seconds=1.0,
                    decode_cuda_ms=999.0,
                ),
                FullKV_diagnostic=fixed.comparisons(reference, saved[arm]),
            )
            rows.append(row)
            write(out / f"traces/{arm}.json", row)
        (out / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write(out / "manifest.json", dict(source_sha=plan["source_sha"], synthetic_only=True))
        write(
            out / "construction.json",
            dict(
                matched_packet_bytes=1024,
                no_padding_or_unused_fill=True,
                old_49_MiB_lossless_all_source_unsupported=True,
            ),
        )
        rehash(tmp_path, index, failures=[], actual_rows=3)
    return tmp_path


def rehash(root, index, **changes):
    out = root / f"workers/{index}"
    result = dict(
        source_sha="a" * 40,
        plan_sha256=study.sha(root / "plan.json"),
        failures=[],
        actual_rows=3,
    )
    if (out / "result.json").is_file():
        result = study.read(out / "result.json")
    result.update(changes)
    result["outputs"] = {
        str(path.relative_to(out)): study.sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "result.json"
    }
    write(out / "result.json", result)


def rewrite_rows(root, index, rows):
    out = root / f"workers/{index}"
    (out / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for row in rows:
        write(out / f"traces/{row['arm']}.json", row)
    rehash(root, index)


def rows(root, index=0):
    return [
        json.loads(line) for line in (root / f"workers/{index}/raw.jsonl").read_text().splitlines()
    ]


def test_policy_does_not_relabel_a_new_attainable_cap_as_the_old_49_mib_gate():
    assert study.POLICY["source"] == "original_BF16_normalized_input_with_original_KV_bit_patches"
    assert study.POLICY["original_49_MiB_all_source_lossless_claim"] is False
    assert study.POLICY["dummy_padding_or_unused_reservations"] is False
    assert study.POLICY["token_reduction_tiling"] is False
    assert study.POLICY["output_predictions"] == 128
    assert study.POLICY["logit_absolute_tolerance"] == 0.001
    assert not study.POLICY["fresh_cohort_authorized"]
    assert not study.POLICY["paper_campaign_authorized"]


def test_broadcast_dispatch_and_constant_workspace_are_preregistered_not_search_choices():
    assert study.POLICY["native_dispatch"] == "zero_stride_batch32_broadcast"
    assert study.POLICY["workspace_component_token_capacity"] == 1024
    assert study.POLICY["native_full_token_axis"] is True
    assert study.POLICY["token_reduction_tiling"] is False
    assert study.POLICY["logit_absolute_tolerance"] == 0.001
    assert (
        study.POLICY["dispatch_operator_evidence_SHA"] == "29805aa4e4f610368d6e1f4165a45e6920a3dc85"
    )
    assert study.POLICY["projection"] == "original_full_width_native_mm"
    assert study.POLICY["raw_baseline"] == "ragged_heads_no_stored_padding"
    assert study.POLICY["workspace_components"] == 1
    assert study.POLICY["workspace_single_head_scratch"] is True
    assert study.POLICY["raw_head_allocation"] == "joint_KV_complete_disjoint_views"
    assert study.POLICY["every_cache_observation_resolves_actual_allocator_blocks"] is True
    assert study.POLICY["allocator_configuration"] == "backend:native,expandable_segments:True"
    assert "roundup_power2_divisions:0" not in study.POLICY["allocator_configuration"]
    script = study.SOURCE / "mosaickv/slurm/fox_bounded_reader_v26.sbatch"
    assert "roundup_power2_divisions:0" not in script.read_text()
    assert study.POLICY["allocator_configuration"] in script.read_text()


def test_immutable_validation_rejects_a_rehashed_workspace_or_dispatch_change(cohort):
    plan = study.read(cohort / "plan.json")
    plan["policy"]["workspace_component_token_capacity"] = 2048
    plan["policy_sha256"] = study.digest(plan["policy"])
    write(cohort / "plan.json", plan)
    with pytest.raises(ValueError, match="immutable policy changed"):
        study.validate(cohort)


def test_immutable_input_binding_is_verified_not_merely_recorded(cohort):
    binding = cohort / "synthetic_bound_input.json"
    write(binding, dict(synthetic_only=True))
    plan = study.read(cohort / "plan.json")
    plan["bindings"] = {str(binding): study.sha(binding)}
    write(cohort / "plan.json", plan)
    assert study.validate(cohort)["bindings"]
    write(binding, dict(synthetic_only=False))
    with pytest.raises(ValueError, match="immutable binding changed"):
        study.validate(cohort)


def test_complete_hashed_cpu_fixture_passes_engineering_not_paper_gates(cohort):
    gate = study.analyze(cohort)
    assert gate["complete"]
    assert gate["source_fidelity_128_pass"]
    assert gate["constructed_decode_boundary_byte_match_pass"]
    assert gate["allocator_block_byte_match_pass"]
    assert gate["old_49_MiB_lossless_source_unsupported"] is True
    assert gate["whole_request_KV_peak_certified"] is False
    assert gate["independent_quality_confirmation"] is False
    assert gate["paper_campaign_authorized"] is False


def test_worker_failure_blocks_both_release_gates(cohort):
    rehash(cohort, 1, failures=[dict(error="synthetic worker failure")])
    gate = study.analyze(cohort)
    assert not gate["complete"]
    assert not gate["source_fidelity_128_pass"]
    assert not gate["constructed_decode_boundary_byte_match_pass"]
    assert gate["failures"][0]["worker"] == 1


def test_missing_raw_arm_is_not_accepted_from_declared_actual_rows(cohort):
    rewrite_rows(cohort, 0, rows(cohort)[:2])
    gate = study.analyze(cohort)
    assert not gate["complete"]
    assert not gate["source_fidelity_128_pass"]


def test_immutable_output_hashes_are_rechecked(cohort):
    out = cohort / "workers/0/manifest.json"
    out.write_text("{}")
    gate = study.analyze(cohort)
    assert not gate["complete"]
    assert "immutable output changed" in gate["failures"][0]["error"]


@pytest.mark.parametrize(
    "boundary",
    (
        "initial_active_kv_storage_bytes",
        "final_active_kv_storage_bytes",
        "peak_active_kv_storage_bytes",
    ),
)
def test_rehashed_unequal_actual_storage_fails_the_matching_gate(cohort, boundary):
    changed = rows(cohort)
    changed[1]["measured"]["storage"][boundary] += 1
    rewrite_rows(cohort, 0, changed)
    gate = study.analyze(cohort)
    assert not gate["constructed_decode_boundary_byte_match_pass"]


@pytest.mark.parametrize(
    "boundary",
    (
        "initial_allocator_block_bytes",
        "final_allocator_block_bytes",
        "peak_allocator_block_bytes",
    ),
)
def test_equal_tensor_backing_does_not_hide_different_real_allocator_blocks(
    cohort, boundary, monkeypatch
):
    changed = rows(cohort)
    indices = {
        "initial_allocator_block_bytes": 0,
        "peak_allocator_block_bytes": 1,
        "final_allocator_block_bytes": 2,
    }
    extra = [128, 128, 128]
    extra[indices[boundary]] += 512
    changed[1]["measured"]["storage"]["allocator_blocks"] = seal_allocator(
        cohort / "workers/0",
        changed[1]["arm"],
        allocator_stats(monkeypatch, extra_by_observation=extra),
    )
    rewrite_rows(cohort, 0, changed)
    gate = study.analyze(cohort)
    assert gate["complete"]
    assert gate["source_fidelity_128_pass"]
    assert gate["constructed_decode_boundary_byte_match_pass"]
    assert not gate["allocator_block_byte_match_pass"]


@pytest.mark.parametrize("damage", ("missing", "observation_count", "unresolved", "integer_true"))
def test_incomplete_or_nonboolean_allocator_owner_coverage_fails_closed(cohort, damage):
    changed = rows(cohort)
    storage = changed[0]["measured"]["storage"]
    if damage == "missing":
        storage.pop("allocator_blocks")
    elif damage == "observation_count":
        storage["allocator_blocks"]["observations"] -= 1
    else:
        storage["allocator_blocks"]["all_required_owners_resolved"] = (
            False if damage == "unresolved" else 1
        )
    rewrite_rows(cohort, 0, changed)
    gate = study.analyze(cohort)
    assert not gate["complete"]
    assert not gate["allocator_block_byte_match_pass"]


def test_rehashed_delta_totals_cannot_replace_replayable_allocator_observations(cohort):
    changed = rows(cohort)
    ledger = changed[0]["measured"]["storage"]["allocator_blocks"]
    artifact = cohort / "workers/0" / ledger["ownership_delta_artifact"]
    data = study.read(artifact)
    data["ownership_deltas"][1]["current_allocator_block_bytes"] += 512
    write(artifact, data)
    ledger["ownership_delta_sha256"] = study.sha(artifact)
    rewrite_rows(cohort, 0, changed)
    gate = study.analyze(cohort)
    assert not gate["complete"]
    assert not gate["allocator_block_byte_match_pass"]


def test_missing_complete_owner_delta_artifact_cannot_promote_byte_matching(cohort):
    changed = rows(cohort)
    ledger = changed[0]["measured"]["storage"]["allocator_blocks"]
    artifact = cohort / "workers/0" / ledger["ownership_delta_artifact"]
    artifact.unlink()
    rehash(cohort, 0)
    gate = study.analyze(cohort)
    assert not gate["complete"]
    assert not gate["allocator_block_byte_match_pass"]


def test_analyzer_recomputes_numerical_gate_instead_of_trusting_true_label(cohort):
    out = cohort / "workers/0"
    saved = torch.load(out / "logits.pt", weights_only=True)
    saved["original_source_bounded"][17, 4] = 0.125
    torch.save(saved, out / "logits.pt")
    changed = rows(cohort)
    # Preserve a malicious/stale true pass label, while all hashes and numeric
    # diagnostic fields correctly describe the changed tensor.
    changed[0]["parity"]["maximum_logit_difference"] = 0.125
    changed[0]["parity"]["bitwise_logit_parity"] = False
    changed[0]["FullKV_diagnostic"] = fixed.comparisons(
        saved["FullKV_reference"], saved["original_source_bounded"]
    )
    rewrite_rows(cohort, 0, changed)
    gate = study.analyze(cohort)
    assert not gate["source_fidelity_128_pass"]


@pytest.mark.parametrize("not_a_boolean", (1, "true", "false", None))
def test_truthy_or_missing_serialized_parity_flags_cannot_promote_a_worker(cohort, not_a_boolean):
    changed = rows(cohort)
    changed[0]["parity"]["pass_128"] = not_a_boolean
    rewrite_rows(cohort, 0, changed)
    gate = study.analyze(cohort)
    assert not gate["complete"]
    assert not gate["source_fidelity_128_pass"]


def test_operator_failure_prevents_any_pretrained_worker_load(cohort):
    write(
        cohort / "operator/result.json",
        dict(
            source_sha="a" * 40,
            failures=[dict(error="synthetic operator failure")],
            operator=dict(all_bitwise_passed=False),
        ),
    )
    with pytest.raises(ValueError, match="operator gate failed"):
        study.worker(cohort, 0)


def test_dispatch_launches_only_analysis_after_failed_operator(cohort, monkeypatch):
    write(
        cohort / "full-cpu/cpu-tests.json",
        dict(status="passed", source_sha="a" * 40),
    )
    write(
        cohort / "operator/result.json",
        dict(
            source_sha="a" * 40,
            failures=[dict(error="operator failed")],
            operator=dict(all_bitwise_passed=False),
        ),
    )
    submitted = []

    def submit(root, revision, label, mode, **kwargs):
        submitted.append((root, revision, label, mode, kwargs))
        return "123"

    monkeypatch.setattr(study, "submit", submit)
    study.dispatch(cohort)
    assert [request[3] for request in submitted] == ["analyze"]
    decision = study.read(cohort / "jobs/dispatch-result.json")
    assert decision["model_workers_blocked"] is True
    assert decision["workers"] is None


def test_operator_precheck_pass_starts_cpu_suite_before_any_model_worker(cohort, monkeypatch):
    submitted = []

    def submit(root, revision, label, mode, **kwargs):
        submitted.append((mode, kwargs))
        return str(123 + len(submitted))

    monkeypatch.setattr(study, "submit", submit)
    study.precheck(cohort)
    assert [mode for mode, _ in submitted] == ["tests", "dispatch"]
    assert not any(kwargs.get("gpu", False) for _, kwargs in submitted)
    assert submitted[1][1]["dependencies"] == ("124",)
    assert not (cohort / "full-cpu/cpu-tests.json").exists()


@pytest.mark.parametrize("failure", ("native", "mapping", "overhead"))
def test_operator_precheck_failure_skips_full_cpu_and_all_model_workers(
    cohort, monkeypatch, failure
):
    result = study.read(cohort / "operator/result.json")
    if failure == "native":
        result["operator"]["all_bitwise_passed"] = False
    else:
        field = "snapshot_consistency_passed" if failure == "mapping" else "overhead_budget_passed"
        result["operator"]["allocator"][field] = False
    write(cohort / "operator/result.json", result)
    submitted = []

    def submit(root, revision, label, mode, **kwargs):
        submitted.append(mode)
        return "123"

    monkeypatch.setattr(study, "submit", submit)
    study.precheck(cohort)
    assert submitted == ["analyze"]


def test_operator_precheck_missing_cuda_result_still_receives_cpu_failure_analysis(
    cohort, monkeypatch
):
    (cohort / "operator/result.json").unlink()
    submitted = []

    def submit(root, revision, label, mode, **kwargs):
        submitted.append(mode)
        return "123"

    monkeypatch.setattr(study, "submit", submit)
    study.precheck(cohort)
    assert submitted == ["analyze"]


def test_dispatch_submits_two_workers_then_afterany_analysis_only_after_both_gates(
    cohort, monkeypatch
):
    write(
        cohort / "full-cpu/cpu-tests.json",
        dict(status="passed", source_sha="a" * 40),
    )
    submitted = []

    def submit(root, revision, label, mode, **kwargs):
        submitted.append((mode, kwargs))
        return str(123 + len(submitted))

    monkeypatch.setattr(study, "submit", submit)
    study.dispatch(cohort)
    assert [mode for mode, _ in submitted] == ["worker", "analyze"]
    assert submitted[0][1] == dict(gpu=True, array=True)
    assert submitted[1][1]["dependencies"] == ("124",)
    decision = study.read(cohort / "jobs/dispatch-result.json")
    assert decision["model_workers_blocked"] is False


def test_command_line_worker_index_is_forwarded_without_defaulting_to_zero(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(study, "worker", lambda root, index: called.append((root, index)))
    monkeypatch.setattr(sys, "argv", ["v26", "worker", "--root", str(tmp_path), "--index", "1"])
    study.main()
    assert called == [(tmp_path, 1)]


def test_command_line_worker_requires_an_explicit_valid_index(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["v26", "worker", "--root", str(tmp_path)])
    with pytest.raises(SystemExit):
        study.main()
