#!/usr/bin/env python3
"""Finite lossless-codec screen -> native controls -> consumed hybrid ablation.

No fresh cohort, benchmark-label selection, default-policy promotion or paper
claim is authorized by this diagnostic. Every producer receives an afterany
analysis, including when it fails. Raw failures are retained.
"""

import argparse
import gc
import json
import os
import random
import subprocess
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
CAPTURES = Path("/scratch/djy8hg/runs/evicastkv/complete-owner-operator-20261107-v2")
CONTROLS = Path("/scratch/djy8hg/runs/evicastkv/component-admission-paired-20261107-v3")
ARMS = (
    "storage_parent",
    "storage_compact",
    "hybrid_compact",
    "hybrid_shortlist",
    "hybrid_score_risk",
)
COMPARISONS = (
    (ARMS[0], ARMS[1]),
    (ARMS[2], ARMS[3]),
    (ARMS[3], ARMS[4]),
    (ARMS[1], ARMS[3]),
    (ARMS[1], ARMS[4]),
)
BOUNDARIES = ("initial_encoded_bytes", "final_encoded_bytes", "reported_counter_peak_bytes")


def read(path):
    return json.loads(Path(path).read_text())


def policies(retained):
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy

    parent = OverlayPolicy(**dict(retained, layers=tuple(retained["layers"])))
    short = replace(
        parent, planner_host_transfer="deferred_layer_vectorized", planner_similarity_limit=128
    )
    risk = replace(
        short,
        repair_execution="score_risk",
        repair_policy="prototype_risk",
        inline_repair=False,
        live_anchor_guard=False,
        reuse_anchor_readout=False,
        anchor_suffix="reference",
        anchor_tail="reference",
        device_guard_records=False,
        export_parent_logits=True,
    )
    storage = OverlayPolicy(enabled=False, quality_objective="bitwise_storage_only")
    return dict(zip(ARMS, (storage, storage, parent, short, risk), strict=True))


def checked(root, expected):
    import mosaickv

    if not Path(mosaickv.__file__).resolve().is_relative_to(SOURCE / "mosaickv/src"):
        raise ValueError("immutable package import boundary required")
    if sha(root / "plan.json") != expected:
        raise ValueError("registered immutable diagnostic plan changed")
    p = read(root / "plan.json")
    clean_source(SOURCE, p["source_sha"])
    if any(sha(Path(k)) != v for k, v in p["bound_files"].items()):
        raise ValueError("consumed fixture or frozen control changed")
    return p


def submit(root, action, after=None):
    gpu = action in ("native", "model")
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=mk-compact-{action}",
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
            "--mem=12G",
            "--time=01:10:00" if action == "controller" else "--time=00:15:00",
        ]
        if action == "cpu":
            args += ["--array=0-1%2"]
    if after:
        args += [f"--dependency=afterany:{after}"]
    args += [
        f"--output={root}/logs/{action}-%A_%a.out",
        f"--error={root}/logs/{action}-%A_%a.err",
        str(SOURCE / "mosaickv/slurm/compact_hybrid_study.sbatch"),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    return subprocess.check_output(args, text=True).strip()


def launch_gpu(root, action):
    array = submit(root, action)
    admission = root / f"{action}-admission.json"
    write_once(
        admission,
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
    receipt = dict(
        array=array,
        controller=submit(root, f"controller-{action}"),
        analyzer=submit(root, f"follow-{action}", array),
        maximum_A100s=2,
        CPUs_per_worker=12,
    )
    write_once(root / f"{action}-submission.json", receipt)
    return receipt


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable diagnostic root required")
    controls = read(CONTROLS / "plan.json")
    if read(CONTROLS / "analysis/decision.json")["complete"] is not True:
        raise ValueError("complete retained-hybrid controls required")
    resolved = policies(controls["retained_policy"])
    paths = [
        CONTROLS / "plan.json",
        CONTROLS / "analysis/decision.json",
        CAPTURES / "plan.json",
        SOURCE / "mosaickv/docs/compact_hybrid_study.md",
    ]
    paths += [Path(k) for k in controls["bound_files"]]
    for i in (0, 1):
        folder = CAPTURES / f"workers/{i}"
        c = read(folder / "capture.json")
        if (
            c["all_128_logits_bitwise"] is not True
            or c["layers"] != 32
            or c["code_archive_sha"] != sha(folder / "source_codes.pt")
        ):
            raise ValueError("complete original COCO source archive required")
        paths += [folder / n for n in ("capture.json", "manifest.json", "source_codes.pt")]
        paths += [Path(v) for v in controls["references"][str(i)].values()]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu-tests.txt").open("x") as out:
        tests = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                *[
                    f"mosaickv/tests/unit/{name}"
                    for name in (
                        "test_compact_rare_source.py",
                        "test_compact_hybrid_study.py",
                        "test_predictive_source.py",
                        "test_all_head_source_gpu.py",
                        "test_hybrid_shortlist.py",
                        "test_score_risk_repair.py",
                        "test_request_peak_coverage.py",
                        "test_storage_six_overlay.py",
                        "test_eager_core_operator.py",
                        "test_paper_request_certification.py",
                    )
                ],
            ],
            cwd=SOURCE,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if tests.returncode:
        raise ValueError("CPU suite failed; no Slurm submission")
    paths.append(root / "cpu-tests.txt")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(x): sha(x) for x in paths},
            controls=controls,
            archives=str(CAPTURES),
            arms=list(ARMS),
            policies={k: asdict(v) for k, v in resolved.items()},
            seed=20261108,
            warmups=5,
            repetitions=20,
            predictions=128,
            guard_tokens=32,
            maximum_logit_difference=1.0,
            maximum_KL=0.02,
            minimum_CPU_source_saving=0.01,
            native_maximum_median_decoder_ratio_for_model_admission=1.05,
            native_operator_5pct_screen_is_NOT_request_latency_acceptance=True,
            strict_request_upper_latency_ratio=1.0,
            bootstrap_replicates=10000,
            comparisons=list(COMPARISONS),
            multiplicity_comparisons=2 * len(COMPARISONS),
            source_codec_applied_to_storage_and_hybrids=True,
            whole_request_KV_only_peak_certified=False,
            independent_quality_N=0,
            fresh_or_paper_campaign_authorized=False,
            benchmark_labels_accessed=False,
            measurement_type="consumed_codec_and_six_mechanism_engineering_not_paper_quality",
        ),
    )
    array = submit(root, "cpu")
    receipt = dict(cpu_array=array, analyzer=submit(root, "follow-cpu", array))
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def prepare_audit(root, reuse):
    """Reuse complete timings; retry only the excluded memory instrumentation."""
    source = clean_source(SOURCE)
    if root.exists() or reuse is None:
        raise ValueError("new audit root and immutable completed parent required")
    p = read(reuse / "plan.json")
    allowed = {
        "mosaickv/scripts/compact_hybrid_study.py",
        "mosaickv/src/mosaickv/evaluation/prompt_graph_reader.py",
        "mosaickv/tests/unit/test_prompt_graph_reader.py",
        "mosaickv/tests/unit/test_compact_hybrid_study.py",
        "mosaickv/src/mosaickv/evaluation/projection_cache.py",
        "mosaickv/tests/unit/test_scratch_audit_lifetime.py",
    }
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", p["source_sha"], source], cwd=SOURCE, text=True
        ).splitlines()
    )
    if not changed <= allowed:
        raise ValueError("audit retry changed algorithm, codec or policy; timings cannot be reused")
    paths = [reuse / "plan.json", reuse / "model-analysis/decision.json"]
    for i in (0, 1):
        folder = reuse / f"model/{i}"
        records = [json.loads(s) for s in (folder / "raw.jsonl").read_text().splitlines()]
        keyed = {(r["phase"], r["arm"], r["trial"]): r for r in records}
        if len(keyed) != len(records):
            raise ValueError("duplicate parent rows cannot be reused")
        wanted = [
            (phase, arm, t)
            for phase, n in (("canary", 1), ("warmup", p["warmups"]), ("timed", p["repetitions"]))
            for arm in ARMS
            for t in range(n)
        ]
        if any(
            k not in keyed
            or keyed[k]["status"] != "completed"
            or not keyed[k]["numerically_admissible"]
            for k in wanted
        ):
            raise ValueError("all parent canaries, warmups and timed rows must be valid")
        paths += [folder / n for n in ("raw.jsonl", "manifest.json", "result.json")]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    timing_producer = p["source_sha"]
    p.update(
        source_sha=source,
        bound_files=dict(p["bound_files"], **{str(f): sha(f) for f in paths}),
        memory_only=True,
        timing_reuse_root=str(reuse),
        timing_rows_producer_SHA=timing_producer,
        audit_rows_producer_SHA=source,
        timing_rerun_authorized=False,
        measurement_type="excluded_memory_audit_only_consumed_inputs_no_new_timing_or_quality",
    )
    write_once(root / "plan.json", p)
    print(json.dumps(launch_gpu(root, "model")), flush=True)


def prepare_fill(root, reuse):
    """One targeted operation change; new paired timing, same consumed inputs."""
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy

    source = clean_source(SOURCE)
    if root.exists() or reuse is None:
        raise ValueError("new fused-fill root and immutable parent required")
    p = read(reuse / "plan.json")
    if p.get("memory_only") or p["fresh_or_paper_campaign_authorized"]:
        raise ValueError("original consumed engineering plan required")
    base = OverlayPolicy(**dict(p["policies"]["hybrid_score_risk"], layers=(31,)))
    arms = ("storage_compact", "hybrid_score_risk", "hybrid_fused")
    registered, paths = (
        {},
        [
            reuse / "plan.json",
            reuse / "model-analysis/decision.json",
            SOURCE / "mosaickv/docs/fused_overlay_fill.md",
        ],
    )
    for i in (0, 1):
        folder = reuse / f"model/{i}"
        records = [json.loads(s) for s in (folder / "raw.jsonl").read_text().splitlines()]
        keyed = {(r["phase"], r["arm"], r["trial"]): r for r in records}
        wanted = [
            (phase, a, t)
            for phase, n in (("canary", 1), ("timed", p["repetitions"]))
            for a in arms[:2]
            for t in range(n)
        ]
        if len(keyed) != len(records) or any(
            k not in keyed
            or keyed[k]["status"] != "completed"
            or not keyed[k]["numerically_admissible"]
            for k in wanted
        ):
            raise ValueError("complete positive frozen controls required; no failed row reuse")
        ref = folder / "hybrid_score_risk.logits.pt"
        paths += [folder / n for n in ("raw.jsonl", "manifest.json", "result.json")]
        paths.append(ref)
        registered[str(i)] = {"hybrid_score_risk": str(ref), "hybrid_fused": str(ref)}
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu-tests.txt").open("x") as out:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_fused_overlay_fill.py",
                "mosaickv/tests/unit/test_storage_six_overlay.py",
                "mosaickv/tests/unit/test_score_risk_repair.py",
                "mosaickv/tests/unit/test_compact_hybrid_study.py",
                "mosaickv/tests/unit/test_scratch_audit_lifetime.py",
            ],
            cwd=SOURCE,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise ValueError("fused policy CPU gate failed; no GPU submission")
    paths.append(root / "cpu-tests.txt")
    p.update(
        source_sha=source,
        bound_files=dict(p["bound_files"], **{str(f): sha(f) for f in paths}),
        arms=list(arms),
        model_arms=list(arms),
        required_canary_arms=list(arms),
        policies={
            "storage_compact": p["policies"]["storage_compact"],
            "hybrid_score_risk": asdict(base),
            "hybrid_fused": asdict(replace(base, overlay_fill_backend="fused")),
        },
        registered_references=registered,
        require_frozen_bitwise_all_arms=True,
        require_all_parent_tokens=True,
        force_control=False,
        fused_fill_native_control=True,
        comparisons=[["hybrid_score_risk", "hybrid_fused"], ["storage_compact", "hybrid_fused"]],
        multiplicity_comparisons=4,
        timing_rerun_reason="new_fused_operation_requires_new_paired_controls",
        measurement_type="consumed_fused_exact_prototype_repair_fill_engineering_not_quality",
    )
    write_once(root / "plan.json", p)
    print(json.dumps(launch_gpu(root, "model")), flush=True)


def archive(p, i):
    import torch

    return torch.load(
        Path(p["archives"]) / f"workers/{i}/source_codes.pt", weights_only=True, map_location="cpu"
    )


def manifest(p, expected, folder, i, *, gpu):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.measurements.telemetry import capture_gpu_environment

    if gpu and (torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name()):
        raise ValueError("one isolated A100 required")
    if gpu and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic CUDA launch required")
    old = read(Path(p["archives"]) / f"workers/{i}/manifest.json")
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
            config_sha=expected,
            source_dirty=False,
            model=old["model"],
            dataset=dict(
                id="COCO2014 captions",
                revision="c3213b54e173ae86517d481de17f31439d6f8c72",
                split="consumed_diagnostic",
            ),
            sample_id=read(p["controls"]["inputs"])[i]["sample_id"],
            environment=environment(),
            CUDA=torch.version.cuda,
            gpu=asdict(capture_gpu_environment(torch)) if gpu else "not_used",
            gpu_count=int(gpu),
            backend="HF_eager" if gpu else "CPU_codec",
            attention_implementation="eager" if gpu else "not_used",
            precision="bf16" if gpu else "uint8_exact_fixture",
            vllm="not_used",
            sglang="not_used",
            seed=p["seed"],
            measurement_type=p["measurement_type"],
            **scheduler_identity({}),
        ),
    )


def cpu(p, folder, i):
    import numpy as np

    from mosaickv.evaluation.compact_rare_source import decode, encode
    from mosaickv.evaluation.predictive_source import encode as parent_encode

    codes = archive(p, i)
    previous, rows = None, []
    for layer in range(32):
        current = codes[layer].numpy()
        packet = encode(previous, current)
        if not np.array_equal(decode(previous, packet), current):
            raise ValueError("lossless compact source round-trip failed")
        before = (
            current.nbytes
            if previous is None
            else min(
                current.nbytes, parent_encode(previous, current, "delta_add").body.serialized_bytes
            )
        )
        if packet.serialized_bytes > before:
            raise ValueError("new actual backing exceeds the parent packet")
        rows.append(
            dict(
                layer=layer,
                parent_bytes=before + 1,
                compact_bytes=packet.serialized_bytes + 1,
                raw=packet.raw,
                bitwise_original_source=True,
            )
        )
        previous = current
    replay = max(c.numel() for c in codes.values())
    before, after = (sum(r[k] for r in rows) + replay for k in ("parent_bytes", "compact_bytes"))
    saving = 1 - after / before
    return dict(
        rows=rows,
        parent_source_plus_replay_bytes=before,
        compact_source_plus_replay_bytes=after,
        saving_fraction=saving,
        passed=saving >= p["minimum_CPU_source_saving"],
        code_bits_exact=True,
    )


def native(p, folder, i):
    import numpy as np
    import torch

    from mosaickv.evaluation import all_head_source_gpu as gpu
    from mosaickv.evaluation.compact_rare_source import encode as oracle

    codes = archive(p, i)
    previous, rows = None, []
    for layer in range(32):
        current = codes[layer].cuda()
        if previous is None:
            rows.append(dict(layer=layer, raw_original=True))
            previous = current
            continue
        reference = oracle(codes[layer - 1].numpy(), codes[layer].numpy())
        packet = gpu.encode(previous, current, additive=True, compact_rare=True)
        parent = gpu.encode(previous, current, additive=True)
        if packet is None:
            if not reference.raw:
                raise ValueError("native/CPU raw selection differs")
            rows.append(dict(layer=layer, raw_original=True))
            previous = current
            continue
        if (
            reference.raw
            or not np.array_equal(packet.payload.cpu().numpy(), reference.payload)
            or not np.array_equal(packet.offsets.cpu().numpy().view(np.uint32), reference.offsets)
        ):
            raise ValueError("native payload/offset bytes differ from CPU oracle")
        restored = gpu.decode(previous, packet, implementation="packed_words")
        if not torch.equal(restored, current):
            raise ValueError("native compact replay changed original source bits")
        scratch, snapshot = previous.clone(), previous.clone()
        gpu.decode(scratch, packet, scratch, implementation="packed_words")
        if not torch.equal(scratch, current) or not torch.equal(previous, snapshot):
            raise ValueError("in-place replay/state isolation failed")
        timings = {}
        for name, body in (("parent", parent), ("compact", packet)):
            if body is None:
                raise ValueError("registered parent packet unexpectedly raw")
            for _ in range(3):
                gpu.decode(previous, body, scratch, implementation="packed_words")
            samples = []
            for _ in range(20):
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                gpu.decode(previous, body, scratch, implementation="packed_words")
                b.record()
                torch.cuda.synchronize()
                samples.append(a.elapsed_time(b))
            timings[name] = samples
        rows.append(
            dict(
                layer=layer,
                parent_bytes=parent.serialized_bytes,
                compact_bytes=packet.serialized_bytes,
                bitwise_original_source=True,
                in_place_isolation=True,
                decoder_ms=timings,
            )
        )
        previous = current
    ratio = float(
        np.median(
            [
                np.median(r["decoder_ms"]["compact"]) / np.median(r["decoder_ms"]["parent"])
                for r in rows
                if "decoder_ms" in r
            ]
        )
    )
    return dict(
        rows=rows,
        median_decoder_ratio=ratio,
        passed=ratio <= p["native_maximum_median_decoder_ratio_for_model_admission"],
        original_source_bits_exact=True,
        native_screen_not_request_latency_claim=True,
    )


def fidelity(actual, reference, *, strict, p):
    import torch

    if (
        actual.shape != reference.shape
        or actual.shape[0] != p["predictions"]
        or not bool(torch.isfinite(actual).all())
    ):
        raise ValueError("complete finite vocabulary vectors required")
    a, b = actual[: p["guard_tokens"]].double(), reference[: p["guard_tokens"]].double()
    identical = bool(torch.equal(actual, reference))
    tokens_match = bool(torch.equal(a.argmax(-1), b.argmax(-1)))
    kl = float((b.softmax(-1) * (b.log_softmax(-1) - a.log_softmax(-1))).sum(-1).max())
    difference = float((a - b).abs().max())
    return dict(
        passed=identical
        if strict
        else tokens_match and kl <= p["maximum_KL"] and difference <= p["maximum_logit_difference"],
        all_128_logit_words_bitwise=identical,
        all_128_tokens_match=bool(torch.equal(actual.argmax(-1), reference.argmax(-1))),
        first_32_tokens_match=tokens_match,
        maximum_32_token_KL_fp64=kl,
        maximum_32_logit_difference=difference,
        KL_is_same_prefix_evidence=tokens_match,
        benchmark_labels_accessed=False,
    )


def model(p, folder, i):
    import evicastkv_paper_precision as campaign
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.component_admission import component_scope
    from mosaickv.evaluation.paper_request_certification import WholeRequestEnvelope
    from mosaickv.evaluation.predictive_source_gpu import use as codec_scope
    from mosaickv.evaluation.request_peak_coverage import coverage
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy
    from mosaickv.graph.structural_templates import engine_scope
    from mosaickv.hf_cli import _model_kwargs

    controls = p["controls"]
    active_arms = tuple(p.get("model_arms", ARMS))
    features = read(controls["inputs"])[i]
    cfg = RunConfig.from_mapping(controls["parent"]["config"])
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    transport = dict(controls["parent"], options=controls["options"])
    options = campaign.request_options(
        controls["options"]["mosaickv_exact"], transport, "mosaickv_exact", False
    )
    reference = torch.load(
        controls["references"][str(i)]["storage"], weights_only=True, map_location="cpu"
    )
    hybrid_reference = torch.load(
        controls["references"][str(i)]["hybrid"], weights_only=True, map_location="cpu"
    )
    registered_references = {
        a: torch.load(path, weights_only=True, map_location="cpu")
        for a, path in p.get("registered_references", {}).get(str(i), {}).items()
    }
    resolved = {
        a: OverlayPolicy(**dict(v, layers=tuple(v["layers"]))) for a, v in p["policies"].items()
    }
    eligible, errors, row_count = [], [], 0
    rng = random.Random(p["seed"] + i)
    with engine_scope(), (folder / "raw.jsonl").open("x") as out:

        def run(arm, phase, trial, order, *, forced=False):
            nonlocal row_count
            row = dict(
                arm=arm,
                phase=phase,
                trial=trial,
                sample_id=features["sample_id"],
                method_order=order,
                status="failed",
                manifest_path=str(folder / "manifest.json"),
                forced_repair_control=forced,
            )
            try:
                policy = replace(resolved[arm], force_repair=True) if forced else resolved[arm]
                audit = phase == "memory_audit"
                gc.collect()
                torch.cuda.synchronize()
                start = perf_counter()
                from contextlib import nullcontext

                envelope = WholeRequestEnvelope(adapter.model) if audit else None
                with (
                    codec_scope(
                        decoder="packed_words",
                        encoding_policy="direct_additive"
                        if arm == ARMS[0]
                        else "direct_additive_compact_rare",
                        audit_encoding=audit,
                    ) as codec,
                    component_scope(
                        adapter,
                        policy,
                        mode="storage_preservation"
                        if not policy.enabled
                        else "experimental_overlay",
                        reader_options=options,
                    ) as mechanisms,
                    envelope if audit else nullcontext(),
                ):
                    logits, detail = campaign.execute(
                        adapter, features, transport, "mosaickv_exact", audit=audit
                    )
                torch.cuda.synchronize()
                elapsed = perf_counter() - start
                actual = logits.cpu()
                detail.update(
                    storage_transport_six_mechanisms_disabled=True,
                    active_overlay_six_mechanisms=policy.enabled,
                    original_six_mechanisms_disabled=not policy.enabled,
                    original_compaction_design_reproduction=False,
                )
                numerical = fidelity(
                    actual,
                    registered_references.get(
                        arm, hybrid_reference if arm == ARMS[2] else reference
                    ),
                    strict=p.get("require_frozen_bitwise_all_arms", arm in ARMS[:3]),
                    p=p,
                )
                numerical["frozen_reference_kind"] = (
                    "registered_unchanged_Torch_arm"
                    if arm in registered_references
                    else "retained_hybrid"
                    if arm == ARMS[2]
                    else "storage_parent"
                )
                numerical["parent_answer_tokens_match"] = bool(
                    torch.equal(actual.argmax(-1), reference.argmax(-1))
                )
                if arm in (ARMS[2], ARMS[3]) or p.get("require_all_parent_tokens"):
                    numerical["passed"] = (
                        numerical["passed"] and numerical["parent_answer_tokens_match"]
                    )
                proof = codec.report()
                if not proof["all_shared_source_consumers_covered"] or len(proof["records"]) != 32:
                    raise ValueError("complete shared-source consumer coverage required")
                if proof["additive_original_owners_retired"] != proof["additive_replacements"]:
                    raise ValueError("old source owners were retained")
                if detail.get("used_fullkv_fallback") is not False or any(
                    type(detail[k]) is not int or detail[k] <= 0 for k in BOUNDARIES
                ):
                    raise ValueError(
                        "no fallback and complete physical storage accounting required"
                    )
                row.update(
                    request_wall_seconds=elapsed,
                    fidelity=numerical,
                    detail=detail,
                    codec=proof,
                    mechanisms=mechanisms,
                    numerically_admissible=numerical["passed"],
                    status="completed",
                )
                if audit:
                    row["allocator_envelope"] = envelope.certificate
                    row["KV_peak_coverage"] = coverage(
                        detail,
                        mechanisms,
                        source_encoding_audit=proof["source_encoding_overlap_audit"],
                    )
                if phase == "canary" or forced:
                    torch.save(actual, folder / f"{arm}{'-forced' if forced else ''}.logits.pt")
                if policy.enabled and (
                    not mechanisms["layers"]
                    or mechanisms["extra_gpu_bytes"] <= 0
                    or mechanisms["residual_cpu_bytes"] <= 0
                ):
                    raise ValueError("six-component arm did not construct real tiers")
                if forced and not mechanisms.get("score_risk_repair", {}).get(
                    "actual_repair_events"
                ):
                    raise ValueError("forced actual repair did not engage")
                del actual, logits
            except Exception:
                row.update(
                    status="failed", error=traceback.format_exc(), numerically_admissible=False
                )
                errors.append(dict(arm=arm, phase=phase, error=row["error"]))
            out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            out.flush()
            row_count += 1
            return row

        if p.get("memory_only"):
            for arm in active_arms:
                row = run(arm, "memory_audit", 0, list(active_arms))
                if row["status"] != "completed" or not row["numerically_admissible"]:
                    break
            return dict(
                passed=not errors and row_count == len(active_arms),
                errors=errors,
                raw_rows=row_count,
                memory_only=True,
                original_timings_reused=True,
            )
        canaries = [run(a, "canary", 0, list(active_arms)) for a in active_arms]
        eligible = [
            r["arm"]
            for r in canaries
            if r.get("numerically_admissible") and r["status"] == "completed"
        ]
        # The trigger control proves mechanics, NOT natural repair or quality recovery.
        if ARMS[4] in eligible and p.get("force_control", True):
            run(ARMS[4], "repair_control", 0, [ARMS[4]], forced=True)
        if not all(a in eligible for a in p.get("required_canary_arms", ARMS[:2])):
            return dict(
                passed=False,
                eligible=eligible,
                errors=errors,
                raw_rows=row_count,
                reason="lossless codec canary failed; no timing",
            )
        for phase, n in (
            ("warmup", p["warmups"]),
            ("timed", p["repetitions"]),
            ("memory_audit", 1),
        ):
            for trial in range(n):
                order = list(eligible)
                rng.shuffle(order)
                for a in order:
                    row = run(a, phase, trial, order)
                    if row["status"] != "completed" or not row["numerically_admissible"]:
                        return dict(
                            passed=False,
                            eligible=eligible,
                            errors=errors,
                            raw_rows=row_count,
                            reason="stop at failed repetition",
                        )
    return dict(
        passed=not errors,
        eligible=eligible,
        errors=errors,
        raw_rows=row_count,
        canary_gate_passed_all_arms=len(eligible) == len(active_arms),
        whole_request_KV_only_peak_certified=False,
    )


def worker(root, expected, action):
    import torch

    p = checked(root, expected)
    torch.set_num_threads(1)
    torch.manual_seed(p["seed"])
    torch.use_deterministic_algorithms(True)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"{action}/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    try:
        manifest(p, expected, folder, i, gpu=action != "cpu")
        if action == "model" and p.get("fused_fill_native_control"):
            from mosaickv.evaluation.fused_overlay_fill import native_controls

            write_once(folder / "native_fill_control.json", native_controls())
        result = globals()[action](p, folder, i)
        result["status"] = "completed"
    except Exception:
        result = dict(status="failed", passed=False, error=traceback.format_exc())
    write_once(folder / "result.json", result)
    print(json.dumps(result), flush=True)
    return int(not result["passed"])


def follow(root, expected, action):
    p = checked(root, expected)
    rows, errors = [], []
    for i in (0, 1):
        try:
            rows.append(read(root / f"{action}/{i}/result.json"))
        except Exception:
            errors.append(traceback.format_exc())
    passed = (
        not errors
        and len(rows) == 2
        and all(r.get("passed") is True and r.get("status") == "completed" for r in rows)
    )
    result = dict(
        complete=not errors and len(rows) == 2,
        gate_passed=passed,
        errors=errors,
        workers=rows,
        fresh_or_paper_campaign_authorized=False,
        exact_whole_request_KV_peak_certified=False,
    )
    if action == "model":
        from mosaickv.evaluation.source_hybrid_validation import paired_summary

        comparisons = []
        for i in (0, 1):
            file = root / f"model/{i}/raw.jsonl"
            if not file.is_file():
                continue
            records = [json.loads(s) for s in file.read_text().splitlines()]
            if p.get("memory_only"):
                parent_file = Path(p["timing_reuse_root"]) / f"model/{i}/raw.jsonl"
                parent_rows = [json.loads(s) for s in parent_file.read_text().splitlines()]
                records += [r for r in parent_rows if r["phase"] != "memory_audit"]
            for old, new in p["comparisons"]:
                pair = {
                    a: sorted(
                        [r for r in records if r["phase"] == "timed" and r["arm"] == a],
                        key=lambda r: r["trial"],
                    )
                    for a in (old, new)
                }
                complete = all(
                    len(pair[a]) == p["repetitions"]
                    and all(
                        r["status"] == "completed" and r["numerically_admissible"] for r in pair[a]
                    )
                    for a in (old, new)
                )
                entry = dict(
                    input_index=i, reference=old, candidate=new, complete_paired_timing=complete
                )
                if complete:
                    entry["latency"] = paired_summary(
                        [r["request_wall_seconds"] for r in pair[old]],
                        [r["request_wall_seconds"] for r in pair[new]],
                        seed=p["seed"] + i,
                        replicates=p["bootstrap_replicates"],
                        comparisons=p["multiplicity_comparisons"],
                    )
                    entry["measured_initial_final_reductions"] = {
                        k: 1 - pair[new][0]["detail"][k] / pair[old][0]["detail"][k]
                        for k in BOUNDARIES[:2]
                    }
                    audits = {
                        a: [
                            r
                            for r in records
                            if r["phase"] == "memory_audit"
                            and r["arm"] == a
                            and r["status"] == "completed"
                        ]
                        for a in (old, new)
                    }
                    entry["independent_audit_present"] = all(
                        len(audits[a]) == 1 for a in (old, new)
                    )
                    if entry["independent_audit_present"]:
                        entry["audited_decode_owned_peak_reduction"] = (
                            1
                            - audits[new][0]["detail"]["decode_owned_peak_bytes"]
                            / audits[old][0]["detail"]["decode_owned_peak_bytes"]
                        )
                        entry["whole_request_KV_peak_certified"] = False
                comparisons.append(entry)
        result["comparisons"] = comparisons
        result["superiority_established"] = (
            False  # consumed engineering inputs; no independent quality test
        )
        result["execution_gate_passed"] = result["gate_passed"]
        result["strict_latency_all_comparisons_passed"] = len(comparisons) == len(
            p["comparisons"]
        ) * 2 and all(
            r.get("latency", {}).get("paired_latency_engineering_gate") is True for r in comparisons
        )
    write_once(root / f"{action}-analysis/decision.json", result)
    if passed and action in ("cpu", "native"):
        result["submitted"] = launch_gpu(root, "native" if action == "cpu" else "model")
        write_once(root / f"{action}-analysis/launch.json", result["submitted"])
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "prepare-audit",
            "prepare-fill",
            "cpu",
            "native",
            "model",
            "follow-cpu",
            "follow-native",
            "follow-model",
            "controller-native",
            "controller-model",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "prepare-audit":
        prepare_audit(args.root, args.reuse)
    elif args.action == "prepare-fill":
        prepare_fill(args.root, args.reuse)
    elif args.action.startswith("controller-"):
        checked(args.root, args.plan_sha)
        suffix = args.action.removeprefix("controller-")
        subprocess.run(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(args.root / f"{suffix}-admission.json"),
                "--state",
                str(args.root / f"{suffix}-controller.jsonl"),
                "--apply",
                "--watch",
            ],
            check=True,
        )
    elif args.action.startswith("follow-"):
        follow(args.root, args.plan_sha, args.action.removeprefix("follow-"))
    else:
        raise SystemExit(worker(args.root, args.plan_sha, args.action))


if __name__ == "__main__":
    main()
