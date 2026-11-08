#!/usr/bin/env python3
"""Finite CPU -> CUDA operators -> consumed model ablations -> analysis.

No new identities, statistical quality claim, baseline campaign or automatic
promotion. Every failed stage receives an afterany analysis. Resources are one
A100/12 CPUs per worker and at most two workers, below the aggregate six cap.
"""

import argparse
import gc
import gzip
import json
import os
import subprocess
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/shared-prototype-ledger-20261108-v6")
TESTS = (
    "test_packed_prototypes.py",
    "test_tiled_softmax.py",
    "test_history_source_precision.py",
    "test_shared_head_prototypes.py",
    "test_shared_head_scope.py",
    "test_kv_timeline.py",
    "test_request_ownership.py",
    "test_ownership_review.py",
    "test_request_peak_coverage.py",
    "test_packed_history_orchestrator.py",
    "test_history_precision_dispatch.py",
    "test_sequential_kv_lease.py",
    "test_sequential_source_history.py",
    "test_bounded_transport_checks.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    p = read(root / "plan.json")
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable plan changed")
    clean_source(SOURCE, p["source_sha"])
    for name, digest in p["bound_files"].items():
        if sha(Path(name)) != digest:
            raise ValueError("immutable consumed control changed: " + name)
    return p


def submit(root, action, *, after=None, stage=None):
    gpu = action in ("operator", "model")
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--job-name=mk-packed-" + action,
    ]
    if gpu:
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=01:00:00",
            "--array=0-1%2",
            "--hold",
        ]
    else:
        args += [
            "--partition=bii",
            "--cpus-per-task=2",
            "--mem=24G",
            "--time=01:10:00" if action == "controller" else "--time=00:15:00",
        ]
    if after:
        args += ["--dependency=afterany:" + after]
    args += [
        f"--output={root}/logs/{action}-{stage or 'none'}-%A_%a.out",
        f"--error={root}/logs/{action}-{stage or 'none'}-%A_%a.err",
        str(SOURCE / "mosaickv/slurm/packed_history.sbatch"),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
        stage or "none",
    ]
    return subprocess.check_output(args, text=True).strip()


def reused_controls(p, index):
    if not p.get("reuse_model_controls"):
        return []
    origin = Path(p["reuse_model_controls"])
    producer = read(origin / "plan.json")["source_sha"]
    rows = [
        json.loads(line) for line in (origin / f"model/{index}/raw.jsonl").read_text().splitlines()
    ]
    rows = [
        r for r in rows if r["arm"] in ("control_original", "control_tiled", "packed_prompt_tiled")
    ]
    if len(rows) != 3 or not all(
        r["status"] == "completed"
        and r.get("numerically_admissible")
        and r.get("whole_request_certificate_valid")
        for r in rows
    ):
        raise ValueError("three complete valid controls per input required")
    return [dict(r, reused=True, producer_source_sha=producer) for r in rows]


def prepare(
    root, reuse=None, reuse_model=None, *, history_feature_group=128, predicate_span_cap=None
):
    if type(history_feature_group) is not int or history_feature_group not in (32, 128):
        raise ValueError("registered history feature group must be 32 or 128")
    if predicate_span_cap is not None and predicate_span_cap != 262144:
        raise ValueError("only the preregistered 262144-scalar predicate cap is admitted")
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new write-once study root required")
    p = read(PARENT / "plan.json")
    p.update(
        source_sha=source,
        guard_tokens=32,
        predictions=128,
        maximum_KL=0.02,
        maximum_logit_difference=1.0,
        require_all_parent_tokens=True,
        model_warmups=0,
        model_repetitions=1,
        operator_warmups=5,
        operator_repetitions=20,
        tiled_softmax_rows=2048,
        model_history_graph=False,
        history_feature_group=history_feature_group,
        predicate_span_cap=predicate_span_cap,
        fresh_or_paper_campaign_authorized=False,
        measurement_type="consumed_packed_constructor_softmax_history_ablation",
        isolated_experimental_variant=True,
    )
    p["bound_files"].update({str(PARENT / "plan.json"): sha(PARENT / "plan.json")})
    for i in (0, 1):
        for name in ("source_codes.pt", "capture.json", "manifest.json"):
            path = Path(p["archives"]) / f"workers/{i}" / name
            p["bound_files"][str(path)] = sha(path)
        for name in ("raw.jsonl", "manifest.json", "shared_source_prototypes-audit.logits.pt"):
            path = PARENT / f"workers/{i}" / name
            p["bound_files"][str(path)] = sha(path)
    if reuse is not None:
        old = read(reuse / "plan.json")
        if read(reuse / "cpu.json")["status"] != "completed":
            raise ValueError("complete CPU feasibility evidence required for reuse")
        for filename in (
            "mosaickv/scripts/packed_source_feasibility.py",
            "mosaickv/src/mosaickv/evaluation/compact_rare_source.py",
            "mosaickv/src/mosaickv/evaluation/all_head_source_codec.py",
        ):
            blob = subprocess.check_output(
                ["git", "show", f"{old['source_sha']}:{filename}"], cwd=SOURCE
            )
            if blob != (SOURCE / filename).read_bytes():
                raise ValueError("CPU cost algorithm changed; cannot reuse its results")
        path = reuse / "cpu-feasibility.json"
        p["reuse_cpu_feasibility"] = str(path)
        p["bound_files"][str(path)] = sha(path)
        p["bound_files"][str(reuse / "plan.json")] = sha(reuse / "plan.json")
    if reuse_model is not None:
        from mosaickv.evaluation.legacy_dispatch_fingerprint import verify

        previous = read(reuse_model / "plan.json")
        filename = "mosaickv/src/mosaickv/evaluation/sequential_kv_lease.py"
        old = subprocess.check_output(
            ["git", "show", f"{previous['source_sha']}:{filename}"], cwd=SOURCE, text=True
        )
        verify(old, (SOURCE / filename).read_text())
        changed = set(
            subprocess.check_output(
                ["git", "diff", "--name-only", previous["source_sha"], source],
                cwd=SOURCE,
                text=True,
            ).splitlines()
        )
        allowed = {
            filename,
            "mosaickv/scripts/packed_history_study.py",
            "mosaickv/src/mosaickv/evaluation/legacy_dispatch_fingerprint.py",
            "mosaickv/tests/unit/test_history_precision_dispatch.py",
            "mosaickv/docs/packed_history_diagnostic.md",
            "mosaickv/src/mosaickv/evaluation/history_source_precision.py",
            "mosaickv/src/mosaickv/evaluation/history_source_precision_kernel.py",
            "mosaickv/tests/unit/test_history_source_precision.py",
            "mosaickv/tests/unit/test_packed_history_orchestrator.py",
            "mosaickv/src/mosaickv/evaluation/bounded_transport_checks.py",
            "mosaickv/tests/unit/test_bounded_transport_checks.py",
        }
        if not changed <= allowed or not read(reuse_model / "operator-analysis.json")["passed"]:
            raise ValueError(
                "retry must preserve all numerical/operator paths except explicit new dispatch"
            )
        p["reuse_model_controls"] = str(reuse_model)
        p["legacy_dispatch_AST_identity_verified"] = True
        for i in (0, 1):
            controls = reused_controls(p, i)
            for path in (
                reuse_model / f"operator/{i}/result.json",
                reuse_model / f"operator/{i}/manifest.json",
                reuse_model / f"model/{i}/raw.jsonl",
                reuse_model / f"model/{i}/manifest.json",
            ):
                p["bound_files"][str(path)] = sha(path)
            for row in controls:
                for path in (
                    Path(row["ledger_path"]),
                    reuse_model / f"model/{i}/{row['arm']}.logits.pt",
                ):
                    p["bound_files"][str(path)] = sha(path)
        path = reuse_model / "operator-analysis.json"
        p["bound_files"][str(path)] = sha(path)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = submit(root, "cpu")
    write_once(root / "submission.json", dict(cpu=cpu, follow=submit(root, "after_cpu", after=cpu)))
    print(json.dumps(read(root / "submission.json")), flush=True)


def launch(root, stage):
    array = submit(root, stage)
    write_once(
        root / f"admission-{stage}.json",
        dict(
            timezone="America/New_York",
            window="all_times",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[array],
            adopt_existing_holds=[array],
        ),
    )
    write_once(
        root / f"{stage}-submission.json",
        dict(
            array=array,
            controller=submit(root, "controller", stage=stage),
            analyzer=submit(
                root, "after_operator" if stage == "operator" else "analyze", after=array
            ),
            maximum_A100s=2,
            CPUs_per_worker=12,
        ),
    )


def cpu(root):
    from packed_source_feasibility import run

    from mosaickv.measurements.allocator_segments import load_reader

    result = dict(status="failed")
    try:
        with (root / "cpu-tests.txt").open("x") as stream:
            done = subprocess.run(
                [
                    PYTHON,
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "addopts=",
                    *["mosaickv/tests/unit/" + f for f in TESTS],
                ],
                cwd=SOURCE,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if done.returncode:
            raise ValueError("CPU regression gate failed")
        native = load_reader()
        result.update(native_binary=native.__file__, native_sha=sha(Path(native.__file__)))
        p = read(root / "plan.json")
        if p.get("reuse_cpu_feasibility"):
            write_once(root / "cpu-feasibility.json", read(p["reuse_cpu_feasibility"]))
            result["feasibility_reused_from"] = p["reuse_cpu_feasibility"]
        else:
            run(root / "cpu-feasibility.json")
        if p.get("reuse_model_controls"):
            from mosaickv.measurements.kv_timeline import valid_whole_request_certificate

            for i in (0, 1):
                for row in reused_controls(p, i):
                    with gzip.open(row["ledger_path"], "rt") as stream:
                        if not valid_whole_request_certificate(json.load(stream)):
                            raise ValueError(
                                "reused interval certificate failed independent verification"
                            )
            write_once(
                root / "operator-analysis.json",
                dict(
                    read(Path(p["reuse_model_controls"]) / "operator-analysis.json"),
                    reused_from=p["reuse_model_controls"],
                ),
            )
        result["status"] = "completed"
    except Exception:
        result["error"] = traceback.format_exc()
    write_once(root / "cpu.json", result)


def trials(function, p):
    import torch

    for _ in range(p["operator_warmups"]):
        value = function()
        del value
    torch.cuda.synchronize()
    rows = []
    for _ in range(p["operator_repetitions"]):
        torch.cuda.synchronize()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start.record()
        value = function()
        end.record()
        torch.cuda.synchronize()
        rows.append(
            dict(
                cuda_ms=start.elapsed_time(end),
                incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated() - baseline,
            )
        )
        del value
    return rows


def operators(root, p, folder, index):
    import torch
    from compact_hybrid_study import manifest
    from packed_source_feasibility import ARCHIVES

    from mosaickv.evaluation.history_source_precision import unpack_into
    from mosaickv.evaluation.packed_prototypes import construct
    from mosaickv.evaluation.precision_tier import QuantizedBody, pack, unpack_cpu
    from mosaickv.evaluation.shared_head_prototypes import construct as reference
    from mosaickv.evaluation.tiled_softmax import probabilities

    manifest(p, sha(root / "plan.json"), folder, index, gpu=True)
    codes = torch.load(
        ARCHIVES / f"workers/{index}/source_codes.pt", map_location="cpu", weights_only=True
    )[0].cuda()
    n = codes.numel() * 8 // (4096 * 3)
    # Real captured codes; artificial unit headers isolate INTEGER equivalence.
    body = QuantizedBody(
        (1, 1, n, 4096),
        3,
        2,
        128,
        codes,
        torch.zeros(1, 1, 4096, (n + 127) // 128, device="cuda"),
        torch.ones(1, 1, 4096, (n + 127) // 128, device="cuda"),
    )
    checks, timings = [], {}
    for count in (1, 2, 4):
        groups = tuple(tuple(range(8 + 16 * i, 16 + 16 * i)) for i in range(count))
        a, b = reference(body, groups, heads=32), construct(body, groups, heads=32)
        checks.append(
            dict(
                kind="packed_constructor",
                groups=count,
                codes_bitwise=torch.equal(a.body.codes, b.body.codes),
                mapping_bitwise=torch.equal(a.body.row_map, b.body.row_map),
                residual_bitwise=torch.equal(a.residual_codes, b.residual_codes),
                changed_symbols_equal=a.changed_code_symbols == b.changed_code_symbols,
                repair_bitwise=torch.equal(b.promote().codes, body.codes),
            )
        )
        del a, b
    groups = (tuple(range(8, 16)),)
    timings["reference_constructor"] = trials(lambda: reference(body, groups, heads=32), p)
    timings["packed_constructor"] = trials(lambda: construct(body, groups, heads=32), p)
    torch.manual_seed(p["seed"] + index)
    for query_length, key_length in ((603, 603), (16, 603), (1, 731)):
        scores = torch.randn(1, 32, query_length, key_length, device="cuda", dtype=torch.bfloat16)
        expected = torch.nn.functional.softmax(scores, -1, dtype=torch.float32).bfloat16()
        actual = probabilities(scores, rows=p["tiled_softmax_rows"])
        checks.append(
            dict(
                kind="tiled_softmax",
                q=query_length,
                k=key_length,
                output_bitwise=torch.equal(actual, expected),
                finite=bool(torch.isfinite(actual).all()),
            )
        )
        if query_length == 603:
            timings["full_softmax"] = trials(
                lambda scores=scores: torch.nn.functional.softmax(
                    scores, -1, dtype=torch.float32
                ).bfloat16(),
                p,
            )
            timings["tiled_softmax"] = trials(
                lambda scores=scores: probabilities(scores, rows=p["tiled_softmax_rows"]), p
            )
        del actual, expected, scores
    write_once(
        folder / "constructor-softmax-progress.json",
        dict(
            checks=checks,
            trials=timings,
            status="completed_partial_operator_scope",
            INT8_unpack_not_yet_checked=True,
        ),
    )
    for n in (16, 32, 96):
        x = torch.randn(1, 1, n, 4096).bfloat16()
        host = pack(x, bits=8, axis=3)
        device = replace(
            host, codes=host.codes.cuda(), scale=host.scale.cuda(), minimum=host.minimum.cuda()
        )
        target = torch.full((128, 4096), -71, device="cuda", dtype=torch.bfloat16)
        unpack_into(device, target)
        checks.append(
            dict(
                kind="history_INT8_unpack",
                n=n,
                output_bitwise=torch.equal(target[:n].cpu().flatten(), unpack_cpu(host).flatten()),
                sentinel_intact=bool((target[n:] == -71).all()),
            )
        )
    result = dict(
        status="completed",
        checks=checks,
        trials=timings,
        model_weights_loaded=False,
        whole_request_memory_NOT_measured=True,
        passed=all(all(v for k, v in row.items() if type(v) is bool) for row in checks),
    )
    write_once(folder / "result.json", result)


@contextmanager
def prompt_scope(groups=None):
    from mosaickv.evaluation import shared_head_scope
    from mosaickv.evaluation.packed_prototypes import construct

    def build(body, original, **kwargs):
        return construct(body, groups or original, **kwargs)

    with (
        patch.object(shared_head_scope, "construct", build),
        shared_head_scope.use(group_size=8, map_policy="first_layer_shared") as report,
    ):
        report["explicit_CPU_feasibility_groups"] = groups
        report["legacy_prompt_group_choice_overridden"] = groups is not None
        yield report


def model(root, p, folder, index):
    import evicastkv_paper_precision as campaign
    import torch
    from compact_hybrid_study import fidelity, manifest

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.history_source_precision import use as precision
    from mosaickv.evaluation.predictive_source_gpu import use as codec
    from mosaickv.evaluation.request_ownership import audit_request
    from mosaickv.evaluation.tiled_softmax import use as tile
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.kv_timeline import valid_whole_request_certificate

    manifest(p, sha(root / "plan.json"), folder, index, gpu=True)
    cfg = RunConfig.from_mapping(p["controls"]["parent"]["config"])
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    transport = dict(
        p["controls"]["parent"],
        options={
            method: dict(options, native_history_graph=False)
            for method, options in p["controls"]["options"].items()
        },
    )  # identical original native history execution control for every arm
    features = read(p["controls"]["inputs"])[index]
    reference = torch.load(
        p["controls"]["references"][str(index)]["storage"], map_location="cpu", weights_only=True
    )
    prototype_reference = torch.load(
        PARENT / f"workers/{index}/shared_source_prototypes-audit.logits.pt",
        map_location="cpu",
        weights_only=True,
    )
    feature_group = p.get("history_feature_group", 128)
    if p.get("predicate_span_cap"):
        from mosaickv.evaluation.bounded_transport_checks import use as bounded
        from mosaickv.evaluation.deferred_transport_checks import use as validation
        from mosaickv.evaluation.shared_source_format import transport_corrections

        fixture = torch.randn(1, 32, 603, 128, device="cuda").bfloat16()
        perturbed = fixture.clone()
        perturbed.flatten()[::127] += 0.25
        empty = torch.empty(0, device="cuda", dtype=torch.int32)

        def correction(capped):
            with ExitStack() as stack:
                if capped:
                    stack.enter_context(bounded(scalars=p["predicate_span_cap"]))
                stack.enter_context(
                    validation(True, coalesce_nonzero=True, full_predicate_span=True)
                )
                return transport_corrections((fixture,), (perturbed,), empty)[0]

        original_patch, bounded_patch = correction(False), correction(True)
        same = all(
            torch.equal(a, b)
            for a, b in zip(original_patch.tensors, bounded_patch.tensors, strict=True)
        )
        result = dict(patch_words_bitwise=same, trials={})
        if same:
            result["trials"] = {
                name: trials(lambda value=flag: correction(value), p)
                for name, flag in (("full_span", False), ("bounded_span", True))
            }
        write_once(folder / "bounded-correction-operator.json", result)
        if not same:
            raise RuntimeError("bounded correction changed sparse patch words")
        del fixture, perturbed, empty, original_patch, bounded_patch
        torch.cuda.synchronize()
    history_arm = (
        "history_INT8_tiled" if feature_group == 128 else f"history_INT8_g{feature_group}_tiled"
    )
    if feature_group != 128:
        # This operator check precedes any model request using the new format.
        # Reused operator evidence certifies group 128 only, not this branch.
        from mosaickv.evaluation.history_source_precision import pack_source, unpack_into

        oracle_checks = []
        for n in (16, 32, 96):
            source = torch.randn(1, 1, n, 4096, device="cuda").bfloat16()
            body = pack_source(source, group=feature_group)
            shaped = body.codes.float().view(1, 1, n, 4096 // feature_group, feature_group)
            oracle = (shaped * body.scale[..., None] + body.minimum[..., None]).bfloat16()
            target = torch.full((128, 4096), -99, dtype=torch.bfloat16, device="cuda")
            unpack_into(body, target)
            oracle_checks.append(
                dict(
                    tokens=n,
                    original_order_oracle_bitwise=bool(
                        torch.equal(target[:n].flatten(), oracle.flatten())
                    ),
                    untouched_workspace_sentinel=bool((target[n:] == -99).all()),
                )
            )
        write_once(
            folder / "new-history-operator.json",
            dict(feature_group=feature_group, checks=oracle_checks),
        )
        if not all(
            r["original_order_oracle_bitwise"] and r["untouched_workspace_sentinel"]
            for r in oracle_checks
        ):
            raise RuntimeError("new history-group GPU operator gate failed")
        del source, body, shaped, oracle, target
        torch.cuda.synchronize()
    cases = [
        ("control_original", False, False, False),
        ("control_tiled", True, False, False),
        ("packed_prompt_tiled", True, True, False),
        (history_arm, True, False, True),
    ]
    if p.get("reuse_model_controls"):
        cases = [(history_arm, True, False, True)]
    if p.get("predicate_span_cap"):
        # Target only the measured peak operation. Both outputs must be bitwise
        # identical to their already bound original/prompt references.
        cases = [
            ("control_bounded_tiled", True, False, False),
            ("packed_prompt_bounded_tiled", True, True, False),
        ]
    feasibility = read(root / "cpu-feasibility.json")
    eligible = [
        r
        for r in feasibility["cases"]
        if r["input"] == index and r["eligible_for_bounded_model_screen"]
    ]
    if eligible:
        chosen = max(eligible, key=lambda r: r["GPU_plus_CPU_saving_at_frozen_inventory_bytes"])
        cases.append(("expanded_prompt_tiled", True, True, False))
    rows = []
    with (folder / "raw.jsonl").open("x") as output:
        for arm, tiled, prototype, history in cases:
            row = dict(
                arm=arm,
                method="evicastkv_experimental_" + arm,
                sample_id=features["sample_id"],
                status="failed",
                manifest_path=str(folder / "manifest.json"),
                model_fidelity_only_not_benchmark_quality=True,
                fresh_or_paper_campaign_authorized=False,
            )
            started = perf_counter()
            ledger = None
            try:
                gc.collect()
                torch.cuda.synchronize()
                with ExitStack() as stack:
                    bounded_report = {}
                    if p.get("predicate_span_cap"):
                        from mosaickv.evaluation.bounded_transport_checks import use as bounded

                        bounded_report = stack.enter_context(
                            bounded(scalars=p["predicate_span_cap"])
                        )
                    stack.enter_context(
                        codec(
                            decoder="packed_words", encoding_policy="direct_additive_compact_rare"
                        )
                    )
                    tile_report = (
                        stack.enter_context(tile(rows=p["tiled_softmax_rows"])) if tiled else {}
                    )
                    explicit_groups = (
                        tuple(tuple(g) for g in chosen["groups"])
                        if arm == "expanded_prompt_tiled"
                        else None
                    )
                    prompt_report = (
                        stack.enter_context(prompt_scope(explicit_groups)) if prototype else {}
                    )
                    history_report = (
                        stack.enter_context(precision(feature_group=feature_group))
                        if history
                        else {}
                    )
                    ledger = stack.enter_context(
                        audit_request(adapter, max_entries=p["maximum_allocator_events"])
                    )
                    logits, detail = campaign.execute(
                        adapter, features, transport, "mosaickv_exact", audit=True
                    )
                torch.cuda.synchronize()
                actual = logits.cpu()
                numeric = fidelity(actual, reference, strict=not (prototype or history), p=p)
                a, b = actual.double(), reference.double()
                full_kl = float(
                    (b.softmax(-1) * (b.log_softmax(-1) - a.log_softmax(-1))).sum(-1).max()
                )
                full_difference = float((a - b).abs().max())
                numeric.update(
                    maximum_128_token_KL_fp64=full_kl,
                    maximum_128_logit_difference=full_difference,
                    all_128_KL_is_same_prefix_evidence=numeric["all_128_tokens_match"],
                )
                numeric["passed"] &= numeric["all_128_tokens_match"]
                if history:
                    numeric["passed"] &= (
                        full_kl <= p["maximum_KL"]
                        and full_difference <= p["maximum_logit_difference"]
                    )
                if arm.startswith("packed_prompt"):
                    numeric["packed_parent_prototype_logits_bitwise"] = bool(
                        torch.equal(actual, prototype_reference)
                    )
                    numeric["passed"] &= numeric["packed_parent_prototype_logits_bitwise"]
                torch.save(actual, folder / f"{arm}.logits.pt")
                certificate = folder / f"{arm}.ledger.json.gz"
                with gzip.open(certificate, "xt", compresslevel=1) as stream:
                    json.dump(ledger, stream, allow_nan=False, separators=(",", ":"))
                row.update(
                    status="completed",
                    fidelity=numeric,
                    numerically_admissible=bool(numeric["passed"]),
                    initial_GPU_KV_bytes=detail["initial_encoded_bytes"],
                    final_GPU_KV_bytes=detail["final_encoded_bytes"],
                    whole_request_KV_plus_workspace_peak_bytes=ledger["exact_KV_peak_bytes"],
                    whole_request_certificate_valid=valid_whole_request_certificate(ledger),
                    ownership_blockers=ledger["blocks_exact_certification"],
                    ledger_path=str(certificate),
                    ledger_sha=sha(certificate),
                    CPU_residual_bytes=prompt_report.get("CPU_residual_bytes", 0),
                    prompt=prompt_report,
                    history=history_report,
                    tiled_softmax=tile_report,
                    bounded_transport_checks=bounded_report,
                    all_head_source_retirement=ledger["all_head_source_retirement_receipts"],
                    timing=dict(
                        decode_cuda_ms=detail["decode_cuda_ms"],
                        instrumented_request_seconds=perf_counter() - started,
                        NOT_clean_latency_measurement=True,
                    ),
                    detail=detail,
                )
                # Legacy per-phase weakref counters are NOT this certificate.
                row["detail"].update(
                    tensor_result_auditing=False,
                    decode_owned_peak_bytes=None,
                    peak_measurement_admissible=False,
                    legacy_tensor_result_audit_superseded=True,
                )
                rows.append(row)
                if arm in ("control_tiled", "control_bounded_tiled") and not numeric["passed"]:
                    raise RuntimeError("tiled full-model parity failed; subsequent arms blocked")
            except Exception:
                row.update(
                    error=traceback.format_exc(), worker_wall_seconds=perf_counter() - started
                )
                if ledger and "ledger_path" not in row:
                    partial = folder / f"{arm}.partial-ledger.json.gz"
                    with gzip.open(partial, "xt", compresslevel=1) as stream:
                        json.dump(ledger, stream, allow_nan=False, separators=(",", ":"))
                    row.update(partial_ledger_path=str(partial), partial_ledger_sha=sha(partial))
                if row not in rows:
                    rows.append(row)
            output.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            output.flush()
            if arm in ("control_tiled", "control_bounded_tiled") and not row.get(
                "numerically_admissible", False
            ):
                break
            if (
                arm == history_arm
                and row.get("numerically_admissible", False)
                and any(
                    r["arm"] == "packed_prompt_tiled" and r.get("numerically_admissible", False)
                    for r in [*rows, *reused_controls(p, index)]
                )
            ):
                combined_arm = (
                    "combined_tiled" if feature_group == 128 else f"combined_g{feature_group}_tiled"
                )
                cases.append((combined_arm, True, True, True))
        # Combined policy is NOT run automatically after a component failure.
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            rows=len(rows),
            failures=sum(r["status"] != "completed" for r in rows),
        ),
    )


def analyze(root):
    p = read(root / "plan.json")
    rows, failures = [], []
    for i in (0, 1):
        rows.extend(dict(input=i, **row) for row in reused_controls(p, i))
        path = root / f"model/{i}/raw.jsonl"
        if not path.exists():
            failures.append(dict(input=i, error="missing worker rows"))
            continue
        worker_rows = [json.loads(line) for line in path.read_text().splitlines()]
        summary_path = root / f"model/{i}/result.json"
        summary = read(summary_path) if summary_path.exists() else {}
        required = (
            {"control_bounded_tiled", "packed_prompt_bounded_tiled"}
            if p.get("predicate_span_cap")
            else {
                "history_INT8_tiled"
                if p.get("history_feature_group", 128) == 128
                else f"history_INT8_g{p['history_feature_group']}_tiled"
            }
        )
        if not p.get("reuse_model_controls") and not p.get("predicate_span_cap"):
            required |= {"control_original", "control_tiled", "packed_prompt_tiled"}
        if (
            not p.get("predicate_span_cap")
            and any(
                r["arm"].startswith("history_INT8") and r.get("numerically_admissible")
                for r in worker_rows
            )
            and any(
                r["arm"] == "packed_prompt_tiled" and r.get("numerically_admissible")
                for r in [*worker_rows, *reused_controls(p, i)]
            )
        ):
            required.add(
                "combined_tiled"
                if p.get("history_feature_group", 128) == 128
                else f"combined_g{p['history_feature_group']}_tiled"
            )
        if (
            summary.get("status") != "completed"
            or summary.get("rows") != len(worker_rows)
            or not required <= {r["arm"] for r in worker_rows}
            or len({r["arm"] for r in worker_rows}) != len(worker_rows)
        ):
            failures.append(
                dict(input=i, error="missing, duplicate or incomplete registered worker arms")
            )
        rows.extend(dict(input=i, **row) for row in worker_rows)
    for row in rows:
        if row["status"] != "completed":
            failures.append(dict(input=row["input"], arm=row["arm"], error=row.get("error")))
    compact = [{k: v for k, v in row.items() if k != "detail"} for row in rows]
    complete = len(rows) >= 8 and not failures
    admitted = complete and all(
        r.get("numerically_admissible", False) and r.get("whole_request_certificate_valid", False)
        for r in rows
    )
    decision = dict(
        complete=complete,
        failures=failures,
        all_registered_component_gates_passed=admitted,
        rows=compact,
        larger_campaign_authorized=False,
        independent_quality_established=False,
        strict_latency_advantage_established=False,
        six_component_superiority_established=False,
        prior_failed_transport_attempts_preserved_at=p.get("reuse_model_controls"),
    )
    folder = root / "analysis"
    folder.mkdir(exist_ok=False)
    write_once(folder / "decision.json", decision)
    lines = [
        "# Packed source / tiled softmax / older-history diagnostic",
        "",
        "Two consumed COCO inputs, not independent quality or paper evidence. "
        "All negative rows retained.",
        "",
        f"Complete: {complete}; all component/ownership gates: {admitted}.",
        "",
        "| Input | Arm | Initial GPU MiB | Final GPU MiB | "
        "Whole-request KV + workspace peak MiB | CPU residual MiB | Model guard |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        sizes = [
            f"{row[k] / 2**20:.6f}" if row.get(k) is not None else "uncertified"
            for k in (
                "initial_GPU_KV_bytes",
                "final_GPU_KV_bytes",
                "whole_request_KV_plus_workspace_peak_bytes",
                "CPU_residual_bytes",
            )
        ]
        lines.append(
            f"| {row['input']} | {row['arm']} | "
            + " | ".join(sizes)
            + f" | {row.get('numerically_admissible', False)} |"
        )
    lines += [
        "",
        "Whole-request peak includes classified shared attention/workspace "
        "requested PyTorch storage, not allocator rounding/reservation, "
        "non-PyTorch allocations, CPU residuals, or process HBM.",
        "Instrumented model times are not clean latency evidence. "
        "INT8 older history is deliberately approximate; original-bit claims do not apply. "
        "The latest sealed 16-token chunk and mutable "
        "tail stay exact. First-32 thresholds are unchanged; history must also pass all-128-token "
        "agreement and the same KL/logit thresholds over the longer trajectory.",
        "CPU source costs include actual codec offsets/payload/replay/maps/residuals "
        "plus the measured frozen noncode remainder. "
        "New transport-fit patch costs require model measurement; CPU rows "
        "are feasibility estimates, never whole-model measured results.",
        "Tiled softmax is common transport offered identically to every method, not an exclusive "
        "EviCastKV algorithmic advantage. No fresh cohort or large campaign is authorized.",
    ]
    operator_path = root / "operator-analysis.json"
    if operator_path.exists():
        lines += [
            "",
            "## CUDA operators (20 trials; not end-to-end model latency)",
            "",
            "| Input | Operator | p50 CUDA ms | p95 CUDA ms | Max incremental allocated MiB |",
            "|---|---|---:|---:|---:|",
        ]
        for r in read(operator_path)["timings"]:
            lines.append(
                f"| {r['input']} | {r['operator']} | {r['p50_cuda_ms']:.6f} | "
                f"{r['p95_cuda_ms']:.6f} | {r['max_incremental_peak_bytes'] / 2**20:.6f} |"
            )
    with (folder / "report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in decision.items() if k != "rows"}), flush=True)


def operator_analysis(root):
    rows, passed = [], True
    for i in (0, 1):
        path = root / f"operator/{i}/result.json"
        result = read(path) if path.exists() else {}
        passed &= result.get("passed") is True
        for name, trials in result.get("trials", {}).items():
            times = [r["cuda_ms"] for r in trials]
            rows.append(
                dict(
                    input=i,
                    operator=name,
                    repetitions=len(times),
                    p50_cuda_ms=median(times),
                    p95_cuda_ms=quantiles(times, n=20, method="inclusive")[-1],
                    max_incremental_peak_bytes=max(
                        r["incremental_peak_allocated_bytes"] for r in trials
                    ),
                )
            )
    write_once(
        root / "operator-analysis.json",
        dict(passed=bool(passed), timings=rows, whole_request_latency_claim=False),
    )
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "cpu",
            "after_cpu",
            "operator",
            "after_operator",
            "model",
            "analyze",
            "controller",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--stage", default="none")
    parser.add_argument("--reuse-cpu", type=Path)
    parser.add_argument("--reuse-model", type=Path)
    parser.add_argument("--history-feature-group", type=int, choices=(32, 128), default=128)
    parser.add_argument("--predicate-span-cap", type=int, choices=(262144,))
    args = parser.parse_args()
    root = args.root
    if args.action == "prepare":
        prepare(
            root,
            args.reuse_cpu,
            args.reuse_model,
            history_feature_group=args.history_feature_group,
            predicate_span_cap=args.predicate_span_cap,
        )
        return
    p = checked(root, args.plan_sha)
    if args.action == "cpu":
        cpu(root)
    elif args.action == "after_cpu":
        if (root / "cpu.json").exists() and read(root / "cpu.json")["status"] == "completed":
            launch(root, "model" if p.get("reuse_model_controls") else "operator")
        else:
            write_once(root / "CPU-negative.json", dict(no_GPU_submission=True))
    elif args.action == "after_operator":
        evidence = [root / f"operator/{i}/result.json" for i in (0, 1)]
        if operator_analysis(root):
            launch(root, "model")
        else:
            write_once(
                root / "operator-negative.json",
                dict(no_model_submission=True, evidence=[str(f) for f in evidence]),
            )
    elif args.action in ("operator", "model"):
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(p["seed"])
        i = int(os.environ["SLURM_ARRAY_TASK_ID"])
        folder = root / f"{args.action}/{i}"
        folder.mkdir(parents=True, exist_ok=False)
        try:
            operators(root, p, folder, i) if args.action == "operator" else model(
                root, p, folder, i
            )
        except Exception:
            write_once(folder / "result.json", dict(status="failed", error=traceback.format_exc()))
            raise
    elif args.action == "controller":
        subprocess.run(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(root / f"admission-{args.stage}.json"),
                "--state",
                str(root / f"admission-{args.stage}-state.json"),
                "--apply",
                "--watch",
            ],
            check=True,
        )
    else:
        analyze(root)


if __name__ == "__main__":
    main()
