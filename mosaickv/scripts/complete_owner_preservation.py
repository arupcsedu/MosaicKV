#!/usr/bin/env python3
"""Finite 16-input frozen-codec preservation and whole-request envelope check."""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.owner_preservation import assess_pair, select_inputs

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/complete-owner-direct-packed-model-20261107-v3")
POOL = Path("/scratch/djy8hg/runs/evicastkv/coco-256-all-methods-20261106-v3/ledger.json")
FROZEN_CODEC = "27cf65a3bf0f4cce7320e1bf5a6e13e9cb9258a7"
ARMS = ("storage_only_common", "storage_only_complete_source_replacement")
SCRIPT = SOURCE / "mosaickv/slurm/complete_owner_preservation.sbatch"


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable preservation plan changed")
    p = read(root / "plan.json")
    clean_source(SOURCE, p["source_sha"])
    for path, value in p["bound_files"].items():
        if sha(Path(path)) != value:
            raise ValueError("hash-bound preservation evidence changed: " + path)
    return p


def submit(root, action, after=None):
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    if action == "worker":
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--array=0-15%6",
            "--hold",
            "--time=00:30:00",
        ]
    else:
        args += ["--partition=bii", "--cpus-per-task=4", "--mem=12G", "--time=00:45:00"]
    args += [
        f"--job-name=mk-owner-{action}",
        f"--output={root}/logs/{action}-%A_%a.out",
        f"--error={root}/logs/{action}-%A_%a.err",
    ]
    if after:
        args += ["--dependency=afterany:" + after]
    args += [str(SCRIPT), str(SOURCE), str(root), sha(root / "plan.json"), action]
    return subprocess.check_output(args, text=True).strip()


def prepare(root):
    source = clean_source(SOURCE)
    parent, decision = read(PARENT / "plan.json"), read(PARENT / "analysis/decision.json")
    if (
        parent["source_sha"] != FROZEN_CODEC
        or not decision["complete"]
        or decision["errors"]
        or not all(r["bitwise_parent_preservation"] for r in decision["results"].values())
        or parent["encoding_policy"] != "direct_additive"
        or parent["decoder_policy"] != "packed_words"
    ):
        raise ValueError(
            "complete frozen numerical/source-retirement engineering evidence required"
        )
    # The failed strict latency gate is retained, never upgraded by this check.
    import complete_owner_study as prior

    if source == FROZEN_CODEC:
        raise ValueError("versioned preservation tooling required")
    for filename in (
        "all_head_source_kernel.py",
        "all_head_source_gpu.py",
        "predictive_source_gpu.py",
    ):
        path = Path("mosaickv/src/mosaickv/evaluation") / filename
        if (
            subprocess.check_output(["git", "show", f"{FROZEN_CODEC}:{path}"], cwd=SOURCE)
            != (SOURCE / path).read_bytes()
        ):
            raise ValueError("frozen source algorithm changed")
    controls = parent["model_controls"]
    excluded = read(controls["inputs"])
    pool = read(POOL)
    inputs = select_inputs(pool["samples"], excluded, count=16, seed=20261108)
    if root.exists():
        raise FileExistsError("write-once preservation root required")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    write_once(root / "inputs.json", inputs)
    selected_ids = {r["sample_id"] for r in inputs}
    references = {
        r["sample_id"]: r["references"] for r in pool["samples"] if r["sample_id"] in selected_ids
    }
    write_once(root / "references.json", references)
    bound = dict(parent["bound_files"])
    for path in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        POOL,
        root / "inputs.json",
        root / "references.json",
        SOURCE / "mosaickv/docs/complete_owner_preservation.md",
    ):
        bound[str(path)] = sha(path)
    for row in inputs:
        for m in row["media"]:
            if sha(Path(m["path"])) != m["sha256"]:
                raise ValueError("candidate media changed")
            bound[m["path"]] = m["sha256"]
    with (root / "cpu-tests.txt").open("x") as out:
        test = subprocess.run(
            [
                prior.PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_owner_preservation.py",
                "mosaickv/tests/unit/test_paper_request_certification.py",
                "mosaickv/tests/unit/test_complete_owner_launch.py",
                "mosaickv/tests/unit/test_predictive_source.py",
            ],
            cwd=SOURCE,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if test.returncode:
        raise ValueError("CPU checks failed; no GPU submission")
    bound[str(root / "cpu-tests.txt")] = sha(root / "cpu-tests.txt")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            frozen_codec_sha=FROZEN_CODEC,
            bound_files=bound,
            model_controls=controls,
            samples=16,
            inputs=str(root / "inputs.json"),
            excluded_codec_development_inputs=controls["inputs"],
            arms=ARMS,
            encoding_policy="direct_additive",
            decoder_policy="packed_words",
            seed=20261108,
            output_predictions=128,
            maximum_a100s=6,
            cpus_per_worker=12,
            scope="frozen_lossless_codec_preservation_and_conservative_whole_request_envelopes",
            original_256_cohort_prior_use_disclosed=True,
            global_untouched_claim=False,
            independent_of_codec_tuning_inputs=True,
            benchmark_labels_used_for_selection=False,
            benchmark_labels_available_to_worker=False,
            performance_acceptance_authorized=False,
            parent_strict_latency_gate_passed=decision[
                "engineering_memory_and_latency_gate_passed"
            ],
            mechanism_scope="storage_only_common_coding_not_six_component_design",
            reference_scope="frozen_quantized_storage_parent_not_unquantized_FullKV",
            exact_whole_request_KV_peak_claim=False,
            fresh_or_paper_campaign_authorized=False,
            measurement_type="independent_input_codec_preservation_not_benchmark_superiority",
        ),
    )
    array = submit(root, "worker")
    write_once(
        root / "admission.json",
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
        root / "submission.json",
        dict(
            array=array,
            controller=submit(root, "controller"),
            analyzer=submit(root, "analyze", array),
        ),
    )
    print(json.dumps(read(root / "submission.json")), flush=True)


def worker(root, expected):
    import evicastkv_paper_precision as campaign
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.all_head_source_gpu import use as common
    from mosaickv.evaluation.component_admission import component_scope
    from mosaickv.evaluation.paper_request_certification import WholeRequestEnvelope
    from mosaickv.evaluation.predictive_source_gpu import use as replacement
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy
    from mosaickv.graph.structural_templates import engine_scope
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    p = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise ValueError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic CUDA configuration required before Python")
    torch.set_num_threads(1)
    torch.manual_seed(p["seed"])
    torch.use_deterministic_algorithms(True)
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    features = read(p["inputs"])[index]
    folder = root / f"workers/{index}"
    folder.mkdir(parents=True, exist_ok=False)
    controls = p["model_controls"]
    cfg = RunConfig.from_mapping(controls["parent"]["config"])
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
            frozen_codec_sha=p["frozen_codec_sha"],
            config_sha=expected,
            source_dirty=False,
            sample_id=features["sample_id"],
            model=asdict(cfg.model),
            dataset=dict(
                id="lmms-lab/COCO-Caption",
                revision="c3213b54e173ae86517d481de17f31439d6f8c72",
                split="historically_used_val_codec_tuning_disjoint",
            ),
            environment=environment(),
            cuda=torch.version.cuda,
            gpu=asdict(capture_gpu_environment(torch)),
            gpu_count=1,
            backend="HF_eager",
            attention_implementation="eager",
            precision="bf16",
            seed=p["seed"],
            vllm="not_used",
            sglang="not_used",
            measurement_type=p["measurement_type"],
            reference_artifact_sha=sha(root / "references.json"),
            **scheduler_identity({}),
        ),
    )
    rows, errors = [], []
    try:
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        transport = dict(controls["parent"], options=controls["options"])
        options = campaign.request_options(
            controls["options"]["mosaickv_exact"], transport, "mosaickv_exact", True
        )
        policy = OverlayPolicy(enabled=False, quality_objective="bitwise_storage_only")

        def scope(arm):
            return (
                common(True, construction="pack_selected", decoder="packed_words")
                if arm == ARMS[0]
                else replacement(decoder="packed_words", encoding_policy="direct_additive")
            )

        warm = read(controls["inputs"])[0]
        order = ARMS if index % 2 == 0 else ARMS[::-1]
        with engine_scope(), (folder / "raw.jsonl").open("x") as out:
            # Same consumed warmup for both arms; not a measurement or a new ID.
            for arm in ARMS:
                with (
                    scope(arm),
                    component_scope(
                        adapter, policy, mode="storage_preservation", reader_options=options
                    ),
                ):
                    values, _ = campaign.execute(
                        adapter, warm, transport, "mosaickv_exact", audit=False
                    )
                del values
            captures = {}
            for arm in order:
                row = dict(
                    sample_id=features["sample_id"],
                    arm=arm,
                    status="failed",
                    manifest_path=str(folder / "manifest.json"),
                )
                audit = None
                try:
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    with (
                        scope(arm) as codec,
                        component_scope(
                            adapter, policy, mode="storage_preservation", reader_options=options
                        ) as mechanisms,
                        WholeRequestEnvelope(adapter.model) as audit,
                    ):
                        values, detail = campaign.execute(
                            adapter, features, transport, "mosaickv_exact", audit=True
                        )
                    tensor = values.cpu().contiguous()
                    del values
                    if tuple(tensor.shape) != (
                        128,
                        adapter.model.get_output_embeddings().weight.shape[0],
                    ):
                        raise ValueError("128 complete pretrained vocabulary vectors required")
                    captures[arm] = tensor
                    row.update(
                        status="completed",
                        detail=detail,
                        codec=codec.report(),
                        mechanism_evidence=mechanisms,
                        whole_request_envelope=audit.certificate,
                        logits_shape=list(tensor.shape),
                        logits_dtype=str(tensor.dtype),
                        logits_sha256=hashlib.sha256(
                            tensor.view(torch.uint8).numpy().tobytes()
                        ).hexdigest(),
                    )
                except Exception:
                    row.update(
                        error=traceback.format_exc(),
                        whole_request_envelope=None if audit is None else audit.certificate,
                    )
                    errors.append(row["error"])
                rows.append(row)
            if len(captures) == 2:
                a, b = (captures[x] for x in ARMS)
                for row in rows:
                    row.update(
                        all_128_logits_bitwise=bool(torch.equal(a, b)),
                        maximum_logit_difference=float((a.float() - b.float()).abs().max()),
                        token_agreement=float((a.argmax(-1) == b.argmax(-1)).float().mean()),
                    )
            for row in rows:
                out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            pair = assess_pair(rows)
            if not pair["valid"]:
                errors.extend(pair["errors"])
            write_once(folder / "pair.json", pair)
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(status="failed" if errors else "completed", errors=errors, rows=len(rows)),
    )
    return int(bool(errors))


def analyze(root, expected):
    import numpy as np

    from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
    from mosaickv.evaluation.metrics.types import MetricSample

    p = checked(root, expected)
    references = read(root / "references.json")
    pairs, rows, errors = [], [], []
    for i in range(p["samples"]):
        folder = root / f"workers/{i}"
        try:
            result = read(folder / "result.json")
            if result["status"] != "completed":
                raise ValueError(result)
            observed = [
                json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
            ]
            pair = assess_pair(observed)
            if not pair["valid"]:
                raise ValueError(pair)
            rows.extend(observed)
            pairs.append(dict(sample_id=observed[0]["sample_id"], **pair))
        except Exception:
            errors.append(dict(index=i, error=traceback.format_exc()))
    quality = None
    if not errors:
        groups = {
            arm: [
                MetricSample(
                    r["sample_id"],
                    r["detail"]["answer"]["answer"],
                    tuple(references[r["sample_id"]]),
                )
                for r in rows
                if r["arm"] == arm
            ]
            for arm in ARMS
        }
        reducer = ResampledCider(groups)
        points = reducer.score(np.ones((1, p["samples"]), dtype=int))
        reducer.validate_official(np.ones((1, p["samples"]), dtype=int), points)
        draws = paired_draw_counts(p["samples"], 1000, p["seed"])
        scores = reducer.score(draws)
        diff = scores[:, reducer.points.index(ARMS[1])] - scores[:, reducer.points.index(ARMS[0])]
        quality = dict(
            corpus_CIDEr={arm: float(points[0, reducer.points.index(arm)]) for arm in ARMS},
            paired_preservation_difference_CI95=np.quantile(diff, [0.025, 0.975]).tolist(),
            evaluator="pycocoevalcap==1.2/PTBTokenizer+Cider_x100",
            bootstrap_replicates=1000,
            identical_answers_verified=True,
            new_quality_improvement_claim=False,
        )
    decision = dict(
        complete=not errors,
        errors=errors,
        completed_pairs=len(pairs),
        expected_pairs=p["samples"],
        all_parent_logits_preserved=not errors,
        all_declared_KV_boundaries_smaller=not errors
        and all(x["smaller_declared_KV_boundaries"] for x in pairs),
        pairs=pairs,
        quality=quality,
        whole_request_envelopes={
            r["sample_id"] + "/" + r["arm"]: r["whole_request_envelope"] for r in rows
        },
        exact_whole_request_KV_peak_certified=False,
        parent_strict_latency_gate_passed=p["parent_strict_latency_gate_passed"],
        new_latency_acceptance_established=False,
        six_component_claim_supported=False,
        global_untouched_claim=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "analysis/decision.json", decision)
    lines = [
        "# Frozen complete-owner codec preservation",
        "",
        "Sixteen preselected inputs from an historically used COCO cohort; IDs/media disjoint "
        "from the codec's two development inputs. Not globally untouched confirmation.",
        "The source codec is frozen at 27cf65a. No new six-component or latency claim is tested. "
        "Every output vocabulary vector must equal the storage parent, not unquantized FullKV.",
        "",
        f"Completed pairs: {len(pairs)}/{p['samples']}; numerical preservation: {not errors}.",
        "",
        "| Sample | Initial saving MiB | Final saving MiB | Audited decode saving MiB |",
        "| --- | ---: | ---: | ---: |",
    ]
    for x in pairs:
        vals = [
            x["memory"][k]["saved_bytes"] / 2**20
            for k in ("initial_encoded_bytes", "final_encoded_bytes", "decode_owned_peak_bytes")
        ]
        lines.append(f"| {x['sample_id']} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines += [
        "",
        f"Corpus scoring: {json.dumps(quality, sort_keys=True)}",
        "",
        "Whole-request envelopes include all PyTorch CUDA allocations across threads/streams "
        "minus only continuously resident immutable model backing. They include non-KV "
        "activations and are conservative upper bounds, NOT exact KV peaks. Vendor workspace "
        "and CPU/Python objects are outside that CUDA tensor boundary. Do not infer peak "
        "reduction from smaller endpoint tensors.",
        "The prior strict latency gate remains negative; this preservation check cannot "
        "upgrade it. Larger/paper/six-mechanism campaigns remain blocked.",
    ]
    with (root / "analysis/report.md").open("x") as out:
        out.write("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(complete=not errors, completed_pairs=len(pairs), quality=quality, errors=errors)
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "controller", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "worker":
        raise SystemExit(worker(args.root, args.plan_sha))
    elif args.action == "analyze":
        analyze(args.root, args.plan_sha)
    else:
        checked(args.root, args.plan_sha)
        subprocess.run(
            [
                "/scratch/djy8hg/env/mosaickv/bin/python",
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(args.root / "admission.json"),
                "--state",
                str(args.root / "admission_state.json"),
                "--apply",
                "--watch",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
