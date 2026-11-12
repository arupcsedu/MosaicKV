#!/usr/bin/env python3
"""Two consumed FP16/GQA exact-cache controls using the original worker/oracle.

No model code is replaced. Twelve requests validate packing, not benchmark
quality, source reconstruction, latency, fresh confirmation, or whole-device
memory. VL-Cache participates in packing/bytes, not a fabricated fidelity pass.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import re
import shlex
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import evicastkv_coco_confirmation_admission_v35 as admission
import evicastkv_original_complete_admission_v36 as original_review
from evicastkv_analyze import collect, pairing
from evicastkv_campaign import admitted_array_concurrency, environment, load_lock, resolved_config
from evicastkv_review_followups import cpu_gate

from mosaickv.config import canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/fox-review-v4-native-coco-20261109")
PARENT_LOCK_SHA = "8d87bc59311c39c921cf32f7a217bf7de1f9671b16c4b0897058d9fce11169e3"
LEDGER = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-v3-discovery-20261109/manifests/coco-validation.json"
)
LEDGER_SHA = "a58c20178a87c93a4cc793a1b985d39f6c83783f103bd021bbbfbbe004e3d17e"
IDS = (
    "coco_caption:COCO_val2014_000000260440.jpg",
    "coco_caption:COCO_val2014_000000512467.jpg",
)
ARMS = tuple(
    (method, cap, "complete")
    for cap in (20, 40)
    for method in ("prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")
)
STAGE = "correctness"
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
HISTORICAL_PRODUCER = "cd5c3ca625c19c3df095dc58dccc9fa8bfc850f6"
SHARED_BINDINGS = {
    "GATE_scoring.json": "b095af44c2ef11655e9c68ffd720c3cb0eefab9fcdc6d98175f56a1cc48a7030",
    "scoring_fixtures.json": "b3b69494ae599244e56fe20e8dfd4cc02637d0447da9411632a707849122d2d8",
    "native/native_metrics.json": (
        "bc44efc7065d4cc62066a699b20a4a31240600680d0cbc4fa84e180cfdcf72e2"
    ),
    "native/analysis_manifest.json": (
        "b03ade42c59ab9a9f9009562cd359ddd6d870ec82bec31b7a11a2ef45414bdde"
    ),
    "baseline_fidelity.json": "f0fa331fd9e552fa368b67d495cc9d9899c843bb12c1ef39db7b14cb3dd1e66b",
    "observations.parquet": "18e38081c405f4ac26e1fc67c01a8b5cbbf9b6edba6a7e05558bc0b75ea5470e",
}
SOURCE_CLOSURE = (
    "mosaickv/src/mosaickv/baselines/vl_cache.py",
    "mosaickv/src/mosaickv/config.py",
    "mosaickv/src/mosaickv/cache_state.py",
    "mosaickv/src/mosaickv/types.py",
    "mosaickv/src/mosaickv/evaluation/coco_shared_bytes.py",
    "mosaickv/src/mosaickv/evaluation/fox_statistics.py",
    "mosaickv/src/mosaickv/evaluation/fox_cider_matrix.py",
    "mosaickv/src/mosaickv/evaluation/metrics/scorers.py",
    "mosaickv/src/mosaickv/evaluation/metrics/statistics.py",
    "mosaickv/src/mosaickv/evaluation/metrics/types.py",
    "mosaickv/src/mosaickv/evaluation/metrics/registry.py",
    "mosaickv/scripts/evicastkv_evidence.py",
    "mosaickv/tests/unit/test_paper_native_metrics.py",
    "mosaickv/tests/unit/test_vl_cache_reimpl.py",
    "mosaickv/tests/unit/test_coco_shared_bytes.py",
)
REQUIRED_FIXTURES = {
    "test_cider_matches_lmms_eval_coco_tokenization_and_scorer",
    "test_cider_reducer_counts_repeated_bootstrap_draws",
    "test_cached_tokens_preserve_official_cider_and_duplicate_draws",
    "test_threshold_filter_matches_paper_equation_one",
    "test_sparsity_and_layer_budget_match_paper_formulas",
    "test_denser_layer_receives_more_budget_and_hard_total_is_exact",
    "test_physical_byte_capacity_scales_layer_adaptation_without_overflow",
}
POLICY = dict(
    consumed_ids=list(IDS),
    arms=[list(arm) for arm in ARMS],
    predictions=64,
    append_steps=63,
    ordinary_and_retention_one_predictions=16,
    independent_layout_forced_steps=16,
    maximum_absolute_logit_error=1e-4,
    masked_unpruned_normwise_FP16_bound=8 * 2**-10,
    attention_layout_policy="native",
    source_reconstruction=False,
    GQA=True,
    fresh_samples=False,
    benchmark_scoring=False,
    headline_timing=False,
    workers=2,
    maximum_a100s=6,
    array_concurrency=2,
    six_a100s_explicitly_authorized=True,
    worker_minutes=30,
)


def read(path, expected=None):
    if expected is not None and sha(Path(path)) != expected:
        raise ValueError(f"immutable metadata binding changed: {path}")
    return json.loads(Path(path).read_text())


def prepare(root, cpu_gate_root, original_review_root=None):
    revision = clean_source(SOURCE)
    root, cpu_gate_root = Path(root), Path(cpu_gate_root)
    if root.exists():
        raise FileExistsError("new immutable consumed-control root required")
    cpu = cpu_gate(cpu_gate_root, revision)
    parent = read(PARENT / "protocol_lock.json", PARENT_LOCK_SHA)
    cohort = parent["cohorts"]["coco"]["validation"]
    if (
        Path(cohort["path"]).resolve(strict=True) != LEDGER.resolve(strict=True)
        or cohort["sha256"] != LEDGER_SHA
        or tuple(cohort["sample_ids"][:2]) != IDS
        or sha(LEDGER) != LEDGER_SHA
        or parent["arithmetic"] != dict(allow_fp16_reduced_precision_reduction=True)
    ):
        raise ValueError("original consumed ledger, order or native arithmetic changed")
    config = read(admission.CANDIDATE, admission.CANDIDATE_SHA)
    if config != read(admission.REVIEW_TEMPLATE, admission.CANDIDATE_SHA):
        raise ValueError("original complete candidate differs from reviewed template")
    bindings = {
        str(PARENT / "protocol_lock.json"): PARENT_LOCK_SHA,
        str(LEDGER): LEDGER_SHA,
        str(admission.CANDIDATE): admission.CANDIDATE_SHA,
        str(admission.REVIEW_TEMPLATE): admission.CANDIDATE_SHA,
    }
    review_path = None
    if original_review_root is not None:
        review_path = (
            Path(original_review_root).resolve(strict=True) / "original_complete_review.json"
        )
        review, _reader = original_review.load_review(review_path.parent)
        if review["source_sha"] != revision:
            raise ValueError("candidate review and cache controls must share one immutable source")
        bindings[str(review_path)] = sha(review_path)
    for name in ("cpu-tests.json", "cpu-tests.xml", "cpu-tests.log"):
        path = cpu_gate_root / "full-cpu" / name
        bindings[str(path.resolve(strict=True))] = sha(path)
    for method in ("full_kv", "prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact"):
        path = SOURCE / "mosaickv/configs/experiments/evicastkv_fox/coco" / f"{method}.json"
        bindings[str(path)] = sha(path)
    lock = copy.deepcopy(parent)
    lock.update(
        name="V36_original_complete_two_consumed_GQA_packing_regate",
        source_sha=revision,
        source_root=str(SOURCE),
        historical_bindings=bindings,
        inherited_gates={},
        stages=[STAGE],
        cohorts={"coco": {"validation": cohort}},
        environment=environment(),
        authorization=dict(automatic_stage_progression=False, fresh_confirmation_authorized=False),
        resources=dict(
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            array_concurrency=2,
            cpus_per_worker=12,
            memory_gib_per_worker=64,
            wall_minutes=POLICY["worker_minutes"],
        ),
    )
    lock.pop("optimization_protocol", None)
    lock["templates"] = {
        "coco": dict(path=str(admission.CANDIDATE), sha256=admission.CANDIDATE_SHA)
    }
    for method, cap, variant in ARMS:
        resolved_config(lock, "coco", method, cap, variant=variant)
    write_once(root / "protocol_lock.json", lock)
    spec = dict(
        stage=STAGE,
        protocol_sha256=sha(root / "protocol_lock.json"),
        imported_workers={},
        jobs=[
            dict(
                workload="coco", subset="validation", indices=[index], stage=STAGE, arms=list(ARMS)
            )
            for index in range(2)
        ],
    )
    write_once(root / "jobs/correctness.json", spec)
    write_once(
        root / "regate_plan.json",
        dict(
            source_sha=revision,
            policy=POLICY,
            policy_sha256=digest(POLICY),
            protocol_sha256=sha(root / "protocol_lock.json"),
            spec_sha256=sha(root / "jobs/correctness.json"),
            current_CPU_gate=cpu,
            cpu_gate_root=str(cpu_gate_root.resolve(strict=True)),
            original_review_root=str(review_path.parent) if review_path else None,
            original_review_sha256=sha(review_path) if review_path else None,
            inherited_current_config_sha256=admission.CANDIDATE_SHA,
            materialization_performed=False,
            fresh_confirmation_authorized=False,
            jobs_submitted=0,
        ),
    )
    return dict(status="prepared_not_submitted", jobs=2, requests=12, jobs_submitted=0)


def load(root):
    root = Path(root).resolve(strict=True)
    plan = read(root / "regate_plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if (
        plan.get("policy") != POLICY
        or plan.get("policy_sha256") != digest(POLICY)
        or sha(root / "protocol_lock.json") != plan["protocol_sha256"]
        or sha(root / "jobs/correctness.json") != plan["spec_sha256"]
    ):
        raise ValueError("registered consumed-control policy/configuration changed")
    lock = load_lock(root)
    spec = read(root / "jobs/correctness.json")
    expected = [
        dict(
            workload="coco",
            subset="validation",
            indices=[i],
            stage=STAGE,
            arms=[list(a) for a in ARMS],
        )
        for i in range(2)
    ]
    if spec.get("jobs") != expected or spec.get("imported_workers") != {}:
        raise ValueError("only two original consumed six-arm workers are permitted")
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    if plan.get("original_review_root") is not None:
        path = Path(plan["original_review_root"]) / "original_complete_review.json"
        read(path, plan["original_review_sha256"])
        original_review.load_review(path.parent)
    return plan, lock


def check_parity(record):
    names = {"ordinary_generate", "retention_one", "mosaickv_runtime_retention_one"}
    if set(record) != names:
        raise ValueError("all three ordinary/retention-one controls required")
    for item in record.values():
        if (
            item.get("generated_tokens") != 16
            or item.get("token_agreement") != 1
            or not math.isfinite(item.get("maximum_logit_difference", math.inf))
            or not 0 <= item["maximum_logit_difference"] <= 1e-4
            or item.get("reference_token_ids") != item.get("candidate_token_ids")
            or len(item.get("reference_token_ids", [])) != 16
        ):
            raise ValueError("unchanged 16-token ordinary/retention-one parity failed")


def check_representation(record):
    if (
        record.get("status") != "passed"
        or record.get("forced_steps") != 16
        or record.get("compact_absolute_tolerance") != 1e-4
        or record.get("retention_one_absolute_tolerance_unchanged") != 1e-4
        or record.get("original_layout_normwise_tolerance") != 8 * 2**-10
        or any(
            record.get(name) is not True
            for name in (
                "source_entries_bitwise",
                "independent_mask_builder",
                "unpruned_control_passed",
                "independent_compact_passed",
                "original_masked_normwise_passed",
            )
        )
        or len(record.get("per_step", [])) != 16
        or len(record.get("unpruned_padding_control", [])) != 16
    ):
        raise ValueError("complete original independent compressed-cache oracle required")
    for index, item in enumerate(record["per_step"]):
        values = (
            item.get("compact_max_absolute", math.inf),
            item.get("masked_normalized_linf", math.inf),
        )
        if (
            item.get("step") != index
            or item.get("compact_greedy_equal") is not True
            or any(not math.isfinite(v) or v < 0 for v in values)
            or values[0] > 1e-4
            or values[1] > 8 * 2**-10
        ):
            raise ValueError("independent same-selected-source numerical control failed")
    if any(
        not math.isfinite(item.get("normalized_linf", math.inf))
        or not 0 <= item["normalized_linf"] <= 8 * 2**-10
        for item in record["unpruned_padding_control"]
    ):
        raise ValueError("unpruned no-eviction padding control failed")


def analyze(root):
    root = Path(root)
    plan, lock = load(root)
    output = root / "analysis/v36-compressed-regate"
    errors, bindings, paired = [], {}, []
    rows, workers = [], []
    try:
        _spec, workers, rows, errors = collect(root, STAGE)
        expected = {
            (sid, method, cap * 2**20, variant) for sid in IDS for method, cap, variant in ARMS
        }
        actual = [(r["sample_id"], r["method"], r["budget_bytes"], r["variant"]) for r in rows]
        if (
            len(rows) != 12
            or len(set(actual)) != 12
            or set(actual) != expected
            or len(workers) != 2
            or errors
        ):
            raise ValueError("missing, duplicated, failed or extra consumed-control observations")
        for index in range(2):
            paths = list((root / "outputs/correctness" / f"{index:03d}").glob("*/worker.json"))
            if len(paths) != 1:
                raise ValueError("one immutable attempt per worker required")
            worker_path = paths[0]
            worker = read(worker_path)
            if worker["source_sha"] != plan["source_sha"]:
                raise ValueError("worker source provenance changed")
            parity_path = worker_path.parent / "retention_one.json"
            check_parity(read(parity_path))
            for path in (worker_path, parity_path):
                bindings[str(path.resolve())] = sha(path)
            for binding in worker["records"]:
                bindings[str(Path(binding["path"]).resolve())] = binding["sha256"]
        for row in rows:
            cap = row["budget_bytes"] // 2**20
            config = resolved_config(lock, "coco", row["method"], cap)
            mapping = canonical_config(config)
            path = Path(row["resolved_config_path"])
            trace = read(row["trace_path"], row["trace_sha256"])
            trace_tokens = trace.get("generated_token_ids", trace.get("token_ids", []))
            if (
                row["source_sha"] != plan["source_sha"]
                or row["effective_method"] != row["method"]
                or row["runtime_config_sha256"] != config_sha256(config)
                or row["config_sha"]
                != digest(dict(runtime_config=mapping, arithmetic=lock["arithmetic"]))
                or digest(read(path, row["resolved_config_file_sha256"])) != digest(mapping)
                or row["emitted_tokens"] != 64
                or row["forward_steps"] != 63
                or row["cache_append_steps"] != 63
                or len(trace_tokens) != 64
                or len(row.get("timings", {}).get("per_token_forward_wall", [])) != 63
                or row["initial_kv_bytes"] != row["budget_bytes"]
                or row.get("boundary") != "A"
                or any(
                    not math.isfinite(row[name]) or row[name] <= 0
                    for name in (
                        "initial_kv_bytes",
                        "final_kv_bytes",
                        "peak_kv_bytes",
                        "e2e_wall_seconds",
                    )
                )
                or not row["initial_kv_bytes"] <= row["final_kv_bytes"] <= row["peak_kv_bytes"]
            ):
                raise ValueError("executed method/configuration/storage schedule changed")
            for row_field, metric_field in (
                ("initial_kv_bytes", "initial_active_kv_storage_bytes"),
                ("final_kv_bytes", "final_active_kv_storage_bytes"),
                ("peak_kv_bytes", "peak_active_kv_storage_bytes"),
            ):
                if row.get("metrics", {}).get(metric_field) != row[row_field]:
                    raise ValueError("actual generation storage metrics differ from recorded bytes")
            representation = (
                Path(row["manifest_path"]).parent / "correctness" / f"{row['run_id']}.json"
            )
            check_representation(read(representation))
            for artifact in (
                path,
                representation,
                Path(row["manifest_path"]),
                Path(row["trace_path"]),
            ):
                bindings[str(artifact.resolve())] = sha(artifact)
        paired = pairing(rows)
        if len(paired) != 4:
            raise ValueError("all four input/budget pairs must match measured bytes")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
    gate = dict(
        source_sha=plan["source_sha"],
        complete=not errors,
        failures=errors,
        consumed_ids=list(IDS),
        expected_requests=12,
        actual_requests=len(rows),
        actual_workers=len(workers),
        representation_forced_steps=16,
        ordinary_and_retention_one_predictions=16,
        compressed_GQA_same_selected_source_pass=not errors,
        per_sample_storage_pairing=paired,
        controls=admission.CONTROLS,
        candidate_config_sha256=admission.CANDIDATE_SHA,
        evidence_files=[dict(path=path, sha256=value) for path, value in sorted(bindings.items())],
        whole_request_allocator_or_source_reader_claim=False,
        benchmark_quality_tested=False,
        fresh_confirmation_authorized=False,
        paper_claim_allowed=False,
        jobs_submitted=0,
    )
    gate_path = output / "GATE_coco_compressed_regate_v36.json"
    write_once(gate_path, gate)
    if not errors:
        common = dict(
            schema="evicastkv_original_complete_qualified_proof_v36",
            source_sha=plan["source_sha"],
            candidate_config_sha256=admission.CANDIDATE_SHA,
            controls=admission.CONTROLS,
            evidence_files=[dict(path=str(gate_path.resolve()), sha256=sha(gate_path))],
        )
        write_once(
            output / "qualified_correctness_proof.json",
            common
            | dict(
                kind="correctness",
                complete=True,
                failures=[],
                retention_one_1e4_pass=True,
                independent_compact_same_selected_source_1e4_pass=True,
                unpruned_padding_control_normwise_pass=True,
                original_KV_positions_mandatory_masks_preserved=True,
                reference_scope="equivalent_selected_original_source_not_unrestricted_FullKV",
                model_scope="FP16_GQA_COCO_exact_selection",
                retention_one_max_absolute_logit_error=1e-4,
                independent_compact_max_absolute_logit_error=1e-4,
                masked_unpruned_normwise_FP16_bound=8 * 2**-10,
            ),
        )
        write_once(
            output / "qualified_byte_matching_proof.json",
            common
            | dict(
                kind="byte_matching",
                complete=True,
                failures=[],
                compared_methods=["prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact"],
                actual_compared_methods_pass=True,
                per_sample_initial_physical_byte_match_pass=True,
                per_sample_final_physical_byte_match_pass=True,
                per_sample_peak_physical_byte_match_pass=True,
                padding_preallocation_unique_backing_charged=True,
                matched_64_predictions_63_appends=True,
                baseline_fidelity_not_inferred=True,
            ),
        )
    if not errors:
        qualify_shared_engineering(root)
    return gate


def shared_engineering_evidence(root, plan):
    """Validate old measured reducers/local formulas without rereading answers."""
    import hashlib

    import pyarrow.parquet as pq

    old = read(PARENT / "protocol_lock.json", PARENT_LOCK_SHA)
    current = environment()
    if any(
        old["environment"].get(name) != current.get(name)
        for name in ("python", "executable", "packages")
    ):
        raise ValueError("historical/current common Python or dependency environment differs")
    records, bindings = {}, {}
    for relative, expected in SHARED_BINDINGS.items():
        path = PARENT / "analysis/scoring" / relative
        if sha(path) != expected:
            raise ValueError("pinned historical scoring evidence changed")
        bindings[str(path.resolve())] = expected
        if path.suffix == ".json":
            records[relative] = read(path, expected)
    score = records["GATE_scoring.json"]
    if (
        score.get("source_sha") != HISTORICAL_PRODUCER
        or score.get("protocol_sha256") != PARENT_LOCK_SHA
        or score.get("status") != "passed"
        or score.get("blockers") != []
        or score.get("tests_actually_executed")
        != dict(
            actual=224,
            completed=224,
            expected=224,
            pairing_count=64,
            workers_completed=4,
            workers_expected=4,
        )
    ):
        raise ValueError("actual historical 224/224 native scoring gate is not qualified")
    if records["scoring_fixtures.json"].get("status") != "passed":
        raise ValueError("historical official/formula scorer fixtures failed")
    native = records["native/native_metrics.json"]
    if not isinstance(native, list) or len(native) != 1:
        raise ValueError("exact one-workload historical native corpus report required")
    report = native[0]
    verification = report.get("reducer_verification", {})
    error = verification.get("maximum_absolute_CIDEr_difference", math.inf)
    if (
        report.get("workload") != "coco"
        or report.get("metric") != "cider"
        or report.get("scope") != "corpus"
        or report.get("unique_inputs") != 32
        or report.get("media_clusters") != 32
        or report.get("resamples") != 10000
        or report.get("seed") != 20271014
        or verification.get("actual_cohort_draws") != 9
        or verification.get("absolute_tolerance") != 1e-10
        or not math.isfinite(error)
        or not 0 <= error <= 1e-10
    ):
        raise ValueError("historical actual-cohort official CIDEr reducer validation failed")
    manifest = records["native/analysis_manifest.json"]
    if (
        manifest.get("runtime_source_sha") != HISTORICAL_PRODUCER
        or manifest.get("analysis_source_sha") != HISTORICAL_PRODUCER
        or manifest.get("protocol_sha256") != PARENT_LOCK_SHA
        or manifest.get("development_only") is not True
        or manifest.get("resamples") != 10000
        or manifest.get("primary_contrasts") != 6
    ):
        raise ValueError("historical native metrics producer/protocol identity changed")
    fidelity = records["baseline_fidelity.json"]
    if (
        fidelity.get("strong_official_baseline_superiority_claim_allowed") is not False
        or fidelity.get("vl_cache_reimpl")
        != "no_audited_executable_official_path_in_this_repository"
    ):
        raise ValueError("historical local-only baseline qualification changed")
    closure = {}
    for relative in SOURCE_CLOSURE:
        original = subprocess.check_output(
            ["git", "show", f"{HISTORICAL_PRODUCER}:{relative}"], cwd=SOURCE
        )
        original_sha = hashlib.sha256(original).hexdigest()
        if sha(SOURCE / relative) != original_sha:
            raise ValueError(f"historical/current metric or baseline source differs: {relative}")
        closure[relative] = original_sha
    scorer = Path(
        importlib.metadata.distribution("lmms-eval").locate_file(
            "lmms_eval/tasks/coco_cap/utils.py"
        )
    )
    # True column projection: no prediction/reference/score values are opened.
    table = pq.read_table(PARENT / "analysis/scoring/observations.parquet", columns=["scorer_sha"])
    if table.num_rows != 224 or set(table.column("scorer_sha").to_pylist()) != {sha(scorer)}:
        raise ValueError("current official scorer source does not match all historical requests")
    cpu_root = Path(plan["cpu_gate_root"])
    cpu = cpu_gate(cpu_root, plan["source_sha"])
    xml = ET.parse(cpu_root / "full-cpu/cpu-tests.xml").getroot()
    passed_names = {
        case.get("name", "").split("[")[0]
        for case in xml.iter("testcase")
        if not any(case.find(name) is not None for name in ("failure", "error", "skipped"))
    }
    if not REQUIRED_FIXTURES.issubset(passed_names):
        raise ValueError("current full CPU gate omitted required official/formula fixtures")
    for name in ("cpu-tests.json", "cpu-tests.xml", "cpu-tests.log"):
        path = cpu_root / "full-cpu" / name
        bindings[str(path.resolve())] = sha(path)
    for index in range(2):
        paths = list((root / "outputs/correctness" / f"{index:03d}").glob("*/environment.json"))
        if len(paths) != 1:
            raise ValueError("one actual current-cache worker environment required")
        observed = read(paths[0])
        if (
            observed.get("source_sha") != plan["source_sha"]
            or observed.get("model", {}).get("id") != admission.CONTROLS["model_id"]
            or observed["model"].get("revision") != admission.CONTROLS["model_revision"]
            or observed["model"].get("precision") != "fp16"
            or observed.get("backend") != "huggingface"
            or observed.get("attention") != "eager"
            or any(
                observed.get("environment", {}).get(name) != current.get(name)
                for name in ("python", "executable", "packages")
            )
        ):
            raise ValueError("actual current GQA model/dependency controls do not match")
        bindings[str(paths[0].resolve())] = sha(paths[0])
    return dict(
        validation_source_sha=plan["source_sha"],
        actual_historical_GPU_producer=HISTORICAL_PRODUCER,
        actual_current_cache_regate_producer=plan["source_sha"],
        old_metric_evidence_not_new_GPU_execution=True,
        current_full_CPU_gate=cpu,
        source_byte_closure=closure,
        common_environment=current,
        official_scorer_source=dict(path=str(scorer.resolve()), sha256=sha(scorer)),
        historical_actual_reducer_verification=verification,
        historical_artifact_bindings=bindings,
        official_reproduction_validated=False,
        published_point_reproduced=False,
        strong_official_baseline_superiority_claim_allowed=False,
    )


def qualify_shared_engineering(root):
    """Derived metric/local-port qualification only after the actual 12-arm pass."""
    root = Path(root)
    output = root / "analysis/v36-compressed-regate"
    if (output / "shared_engineering_qualification.json").exists():
        raise FileExistsError("shared engineering qualification is write-once")
    plan, _lock = load(root)
    failures = []
    frozen = None
    gate_path = output / "GATE_coco_compressed_regate_v36.json"
    try:
        gate = read(gate_path)
        if (
            gate.get("complete") is not True
            or gate.get("failures") != []
            or gate.get("source_sha") != plan["source_sha"]
            or gate.get("actual_requests") != 12
            or gate.get("actual_workers") != 2
            or gate.get("consumed_ids") != list(IDS)
        ):
            raise ValueError("complete actual current-source 12-request GQA regate required")
        _spec, workers, rows, collection_errors = collect(root, STAGE)
        expected = {(sid, method, cap * 2**20) for sid in IDS for method, cap, _variant in ARMS}
        if (
            collection_errors
            or len(workers) != 2
            or len(rows) != 12
            or {(r["sample_id"], r["method"], r["budget_bytes"]) for r in rows} != expected
            or any(w.get("source_sha") != plan["source_sha"] for w in workers)
            or any(
                r.get("source_sha") != plan["source_sha"]
                or r.get("emitted_tokens") != 64
                or r.get("forward_steps") != 63
                for r in rows
            )
            or len(pairing(rows)) != 4
            or not gate.get("evidence_files")
        ):
            raise ValueError(
                "a bare positive gate cannot replace actual completed cache observations"
            )
        for binding in gate["evidence_files"]:
            if sha(Path(binding["path"])) != binding["sha256"]:
                raise ValueError("actual current-regate evidence changed")
        evidence = shared_engineering_evidence(root, plan)
        evidence_path = output / "shared_engineering_evidence.json"
        write_once(evidence_path, evidence)
        common = dict(
            schema="evicastkv_original_complete_qualified_proof_v36",
            complete=True,
            failures=[],
            source_sha=plan["source_sha"],
            candidate_config_sha256=admission.CANDIDATE_SHA,
            controls=admission.CONTROLS,
            proof_origin="derived_validation_not_new_historical_GPU_execution",
            actual_historical_GPU_producer=HISTORICAL_PRODUCER,
            actual_current_cache_regate_producer=plan["source_sha"],
            evidence_files=[
                dict(path=str(path.resolve()), sha256=sha(path))
                for path in (gate_path, evidence_path)
            ],
            official_reproduction_validated=False,
            published_point_reproduced=False,
            strong_official_baseline_superiority_claim_allowed=False,
        )
        write_once(
            output / "qualified_metric_proof.json",
            common
            | dict(
                kind="metric",
                pinned_official_CIDEr_fixed_fixture_pass=True,
                actual_cohort_sparse_official_reducer_equivalence_pass=True,
                corpus_reducer_recomputed_each_bootstrap_draw=True,
            ),
        )
        write_once(
            output / "qualified_baseline_fidelity_proof.json",
            common
            | dict(
                kind="baseline_fidelity",
                vl_cache_reimpl_qualified_as_local_reimplementation=True,
                registered_formula_and_native_metric_validation_pass=True,
                qualification_scope="local_paper_equations_and_native_CIDEr_only_not_published_point_reproduction",
            ),
        )
        if plan.get("original_review_root") is not None:
            proof_paths = {
                "correctness": output / "qualified_correctness_proof.json",
                "metric": output / "qualified_metric_proof.json",
                "byte_matching": output / "qualified_byte_matching_proof.json",
                "baseline_fidelity": output / "qualified_baseline_fidelity_proof.json",
            }
            frozen = original_review.freeze_candidate(plan["original_review_root"], proof_paths)
            if frozen.get("candidate_parameters_frozen") is not True:
                raise ValueError("validated engineering proofs did not freeze original parameters")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ET.ParseError,
        subprocess.CalledProcessError,
    ) as error:
        failures.append(dict(error=f"{type(error).__name__}: {error}"))
    result = dict(
        source_sha=plan["source_sha"],
        complete=not failures,
        failures=failures,
        actual_historical_GPU_producer=HISTORICAL_PRODUCER,
        fresh_confirmation_authorized=False,
        paper_claim_allowed=False,
        official_reproduction_validated=False,
        published_point_reproduced=False,
        jobs_submitted=0,
        candidate_parameters_frozen=not failures and frozen is not None,
    )
    write_once(output / "shared_engineering_qualification.json", result)
    return result


def submit(root, *, cpu_partition="interactive"):
    """Explicit future invocation only; one finite GPU array and afterany analyzer."""
    root = Path(root).resolve(strict=True)
    plan, _lock = load(root)
    if not re.fullmatch(r"[a-zA-Z0-9_,.-]+", cpu_partition):
        raise ValueError("explicit CPU partition list required")
    active = subprocess.check_output(
        ["squeue", "-h", "-u", os.environ["USER"], "-t", "RUNNING,COMPLETING", "-o", "%b"],
        text=True,
    )
    used = sum(
        int(match.group(1))
        for line in active.splitlines()
        if (match := re.search(r"gpu(?::[a-zA-Z0-9_-]+)?:(\d+)", line))
    )
    limit = admitted_array_concurrency(POLICY, used)
    if limit == 0:
        raise ValueError("aggregate six-GPU ceiling has no free admission slot")
    (root / "slurm").mkdir(exist_ok=True)
    gpu = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        f"--array=0-1%{limit}",
        f"--time=00:{POLICY['worker_minutes']:02d}:00",
        f"--output={root}/slurm/compressed-%A_%a.out",
        f"--error={root}/slurm/compressed-%A_%a.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_worker.sbatch"),
        str(SOURCE),
        str(root),
        STAGE,
        plan["spec_sha256"],
    ]
    write_once(root / "jobs/regate-gpu-request.json", dict(command=gpu))
    job = subprocess.check_output(gpu, text=True).strip().split(";")[0]
    if not re.fullmatch(r"\d+", job):
        raise ValueError("invalid GPU scheduler acknowledgment")
    write_once(root / "jobs/regate-gpu-submitted.json", dict(job=job, command=gpu))
    python_path = f"{SOURCE / 'mosaickv/src'}:{SOURCE / 'mosaickv/scripts'}"
    recipe = " && ".join(
        (
            f"cd {shlex.quote(str(SOURCE))}",
            "source mosaickv/scripts/cache_env.sh",
            f"export PYTHONPATH={shlex.quote(python_path)}",
            f"{shlex.quote(PYTHON)} "
            "mosaickv/scripts/evicastkv_coco_compressed_regate_v36.py analyze "
            f"--root {shlex.quote(str(root))}",
        )
    )
    wrap = f"bash -c {shlex.quote(recipe)}"
    cpu = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        f"--partition={cpu_partition}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=8G",
        "--time=00:10:00",
        f"--dependency=afterany:{job}",
        f"--output={root}/slurm/analysis-%j.out",
        f"--error={root}/slurm/analysis-%j.err",
        "--wrap",
        wrap,
    ]
    write_once(root / "jobs/regate-analysis-request.json", dict(command=cpu))
    analyzer = subprocess.check_output(cpu, text=True).strip().split(";")[0]
    if not re.fullmatch(r"\d+", analyzer):
        raise ValueError("invalid CPU scheduler acknowledgment")
    write_once(root / "jobs/regate-analysis-submitted.json", dict(job=analyzer, command=cpu))
    return dict(gpu_array=job, analysis=analyzer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "analyze", "qualify-engineering"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--original-review-root", type=Path)
    parser.add_argument("--cpu-partition", default="interactive")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.cpu_gate_root is None:
            parser.error("complete current-source CPU gate root required")
        result = prepare(args.root, args.cpu_gate_root, args.original_review_root)
    elif args.mode == "submit":
        result = submit(args.root, cpu_partition=args.cpu_partition)
    elif args.mode == "analyze":
        result = analyze(args.root)
    else:
        result = qualify_shared_engineering(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
