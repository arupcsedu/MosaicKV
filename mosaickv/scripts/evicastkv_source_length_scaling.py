#!/usr/bin/env python3
"""R09: one consumed image, synthetic text extensions, fixed source decode horizon."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import evicastkv_review_source_controls as source

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = source.RUNS / "fox-review-v6-parallel-controls-20261110/source"
EXTENSIONS = (0, 128, 256)
FILLER = " neutral"
ARMS = (
    "fullkv_native_reference",
    "source3_attention_optional",
    "direct_projected_kv3_attention_optional_local",
)
FILL_STEPS = (1, 32, 64, 96, 127)
LAYERS = 32
PREDICTIONS = 128
MEASURED = 2
WARMUPS = 1
SOFT_SECONDS = 1100
POLICY = dict(
    source_policy=source.POLICY,
    synthetic_text_extension_units=EXTENSIONS,
    filler=FILLER,
    arms=ARMS,
    fixed_predictions=PREDICTIONS,
    source_replay=False,
    measured_prompt_fill_steps=FILL_STEPS,
    warmups_per_cell=WARMUPS,
    measured_repetitions_per_cell=MEASURED,
    worker_soft_seconds=SOFT_SECONDS,
    worker_hard_seconds=1200,
    independent_quality_N=0,
    realistic_benchmark_scores_allowed=False,
    exact_byte_matching_claim_allowed=False,
    whole_request_KV_only_peak_certified=False,
    headline_latency_allowed=False,
    fresh_confirmation_authorized=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def fixture(payload, units):
    """Never use references, alter visual input, or call filler units model tokens."""
    if (
        type(units) is not int
        or units not in EXTENSIONS
        or set(payload) != {"task", "sample_id", "prompt", "media"}
        or not isinstance(payload["prompt"], str)
        or not payload["prompt"].strip()
        or len(payload["media"]) != 1
        or payload["media"][0]["kind"] != "image"
    ):
        raise ValueError(
            "exactly one consumed image and a registered label-free extension required"
        )
    prompt = payload["prompt"]
    if units:
        prompt += "\nSynthetic length probe:" + FILLER * units
    return dict(payload, prompt=prompt)


def prepare(root):
    root = Path(root)
    revision = clean_source(SOURCE)
    parent_path = PARENT / "plan.json"
    gate_path = PARENT / "analysis/GATE_source_controls.json"
    plan, gate = read(parent_path), read(gate_path)
    if (
        gate.get("complete") is not True
        or gate.get("failures") != []
        or gate.get("actual_rows") != 14
        or gate.get("plan_sha256") != sha(parent_path)
    ):
        raise ValueError("complete historical consumed-source execution evidence required")
    job = plan["jobs"][0]  # fixed before seeing any model outputs
    (input_path,) = map(Path, job["inputs"])
    payload = fixture(read(input_path), 0)
    media = payload["media"][0]
    media_path = Path(media["path"])
    if sha(input_path) != job["input_sha256"] or sha(media_path) != media["sha256"]:
        raise ValueError("historical input/media bytes changed")
    files = [
        parent_path,
        gate_path,
        input_path,
        media_path,
        Path(__file__),
        Path(source.__file__),
        *(
            SOURCE / f"mosaickv/src/mosaickv/evaluation/{name}.py"
            for name in (
                "storage_runs",
                "graphed_storage",
                "shared_source_format",
                "source_history",
            )
        ),
    ]
    write_once(
        root / "plan.json",
        dict(
            source_sha=revision,
            policy_sha256=digest(POLICY),
            policy=POLICY,
            parent_input=str(input_path),
            config=job.get("config", plan["config"]),
            bindings={str(p): sha(p) for p in files},
            fixture_sha256={str(u): digest(fixture(payload, u)) for u in EXTENSIONS},
            expected_rows=len(EXTENSIONS) * len(ARMS) * MEASURED,
            workers=1,
        ),
    )
    return dict(workers=1, expected_rows=len(EXTENSIONS) * len(ARMS) * MEASURED)


def load(root):
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if (
        digest(plan["policy"]) != digest(POLICY)
        or plan["policy_sha256"] != digest(POLICY)
        or any(sha(Path(p)) != value for p, value in plan["bindings"].items())
    ):
        raise ValueError("registered scaling protocol or producer artifact changed")
    return plan


def geometry(prepared, context_limit):
    """Observe the actual expanded processor output, not a guessed tokenizer ratio."""
    n = prepared.logical_sequence_length
    if type(context_limit) is not int or n + PREDICTIONS - 1 > context_limit:
        raise ValueError("expanded prompt plus fixed continuation exceeds model context")
    counts = {"text": 0, "image": 0}
    images, cursor = [], 0
    for span in prepared.modality_map:
        kind = span.modality.value
        if kind not in counts or span.start != cursor or span.end > n or span.end <= cursor:
            raise ValueError("complete contiguous text/image modality provenance required")
        counts[kind] += span.end - cursor
        cursor = span.end
        if kind == "image":
            images.append(dict(start=span.start, end=span.end, patch_grid=span.grid_shape))
    if cursor != n or len(images) != 1 or not counts["text"] or not counts["image"]:
        raise ValueError("expected one expanded image with an exact modality partition")
    ids = prepared.model_inputs["input_ids"]
    pixels = prepared.model_inputs["pixel_values"]
    if tuple(ids.shape) != (1, n) or pixels.shape[0] != 1:
        raise ValueError("batch-one processor tensors must match logical geometry")
    return dict(
        decoder_prompt_positions=n,
        text_positions=counts["text"],
        image_positions=counts["image"],
        image_spans=images,
        processed_pixel_shape=list(pixels.shape),
        processed_pixel_sha256=hashlib.sha256(
            pixels.detach().float().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        input_token_ids_sha256=digest(ids.detach().cpu().tolist()),
        context_limit=context_limit,
    )


@contextmanager
def prompt_fills(torch):
    """Measure only actual prompt fills, not sealed-history fills or other attention."""
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.graphed_storage import SharedSourceLayer
    from mosaickv.evaluation.head_local_precision import HeadLocalFormat
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    records = []
    active = None
    original = SharedSourceLayer.fill_immutable

    @wraps(original)
    def fill_immutable(layer):
        nonlocal active
        if active is not None:
            raise RuntimeError("nested shared workspace fill")
        if layer.history_length in FILL_STEPS:
            active = (layer.representation, layer.history_length)
        try:
            return original(layer)
        finally:
            active = None

    def instrument(method):
        @wraps(method)
        def fill(packet, *args, **kwargs):
            if active is None or packet is not active[0]:
                return method(packet, *args, **kwargs)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            begin = time.perf_counter_ns()
            result = method(packet, *args, **kwargs)
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            end.record()
            records.append((active[1], packet.shape[-2], elapsed, start, end))
            return result

        return fill

    with ExitStack() as stack:
        stack.enter_context(patch.object(SharedSourceLayer, "fill_immutable", fill_immutable))
        for cls in (ExactFormat, HeadLocalFormat, SharedSourceFormat):
            stack.enter_context(patch.object(cls, "fill", instrument(cls.fill)))
        yield records


def summarize_fills(records, n):
    result = []
    for step in FILL_STEPS:
        group = [r for r in records if r[0] == step]
        if len(group) != LAYERS or any(length != n for _, length, *_ in group):
            raise ValueError("expected one observed full prompt fill per layer at each fixed step")
        host = [r[2] for r in group]
        stream = [r[3].elapsed_time(r[4]) for r in group]
        if any(not math.isfinite(x) or x < 0 for x in host + stream):
            raise ValueError("nonfinite prompt-fill interval")
        result.append(dict(step=step, layers=len(group), host_ms=sum(host), stream_ms=sum(stream)))
    if len(records) != len(FILL_STEPS) * LAYERS:
        raise ValueError("unregistered observed prompt fill")
    return result


def worker(root, index=0):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = load(root)
    if type(index) is not int or index != 0:
        raise ValueError("only the single registered worker is permitted")
    out = root / "worker"
    out.mkdir(parents=True, exist_ok=False)
    errors, completed = [], 0
    deadline = time.monotonic() + SOFT_SECONDS
    try:
        if (
            not os.environ.get("SLURM_JOB_ID", "").isdigit()
            or torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("isolated deterministic one-A100 source worker required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(source.POLICY["random_seed"])
        cfg = RunConfig.from_mapping(plan["config"])
        if (
            (cfg.model.id, cfg.model.revision) != source.MODEL
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
            or cfg.execution.decode_attention_implementation != "eager"
            or cfg.generation.max_new_tokens != PREDICTIONS
        ):
            raise ValueError(
                "registered BF16 eager LLaVA checkpoint and 128-token horizon required"
            )
        adapter = load_hf_adapter(
            source.MODEL[0],
            revision=source.MODEL[1],
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        text = adapter.model.config.text_config
        model_geometry = dict(
            layers=len(adapter._language_layers()),
            query_heads=text.num_attention_heads,
            kv_heads=text.num_key_value_heads,
            source_width=text.hidden_size,
            head_width=text.hidden_size // text.num_attention_heads,
        )
        if tuple(model_geometry.values()) != (LAYERS, 32, 32, 4096, 128):
            raise ValueError("unsupported source cache geometry: full-MHA is mandatory")
        payload = read(plan["parent_input"])
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                config_sha256=config_sha256(cfg),
                plan_sha256=sha(root / "plan.json"),
                source_dirty=False,
                model=asdict(cfg.model),
                model_geometry=model_geometry,
                dataset=asdict(cfg.dataset),
                processor_revision=source.MODEL[1],
                origin_sample_id=payload["sample_id"],
                media_sha256=payload["media"][0]["sha256"],
                environment=environment(),
                gpu=asdict(capture_gpu_environment(torch)),
                cuda_runtime=torch.version.cuda,
                seed=source.POLICY["random_seed"],
                backend="huggingface",
                attention="eager",
                vllm="not_used",
                sglang="not_used",
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                measurement_type="synthetic_prompt_geometry_diagnostic",
                result_category="unified_comparison",
                result_origin="locally_measured",
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                independent_quality_N=0,
            ),
        )
        native_prepare = adapter.prepare_inputs
        with (out / "raw.jsonl").open("x") as stream, torch.inference_mode():
            for units in EXTENSIONS:
                item = fixture(payload, units)
                if digest(item) != plan["fixture_sha256"][str(units)]:
                    raise ValueError("synthetic fixture content changed")
                for repetition in range(-WARMUPS, MEASURED):
                    order = ARMS if repetition != 1 else tuple(reversed(ARMS))
                    for arm in order:
                        row = dict(
                            extension_units=units,
                            arm=arm,
                            repetition=repetition,
                            warmup=repetition < 0,
                            status="failed",
                            fixture_sha256=digest(item),
                            source_sha=plan["source_sha"],
                            manifest_path=str(out / "manifest.json"),
                        )
                        try:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("bounded worker soft deadline reached")
                            observed = []

                            def prepare_checked(*args, _observed=observed, **kwargs):
                                prepared = native_prepare(*args, **kwargs)
                                _observed.append(geometry(prepared, text.max_position_embeddings))
                                return prepared

                            with (
                                patch.object(adapter, "prepare_inputs", prepare_checked),
                                prompt_fills(torch) as fills,
                            ):
                                logits, record = source._request(adapter, item, cfg, arm)
                            (geo,) = observed
                            n = geo["decoder_prompt_positions"]
                            storage = record["storage"]
                            if (
                                logits.ndim != 2
                                or logits.shape[0] != PREDICTIONS
                                or record["original_logical_length"] != n
                                or record["final_next_position"] != n + PREDICTIONS - 1
                                or record["used_fullkv_fallback"]
                                or storage["graph_replays"]
                                or storage["outstanding_layer_leases"]
                                or storage["active_workspace_bytes"]
                                or storage["history_updates_per_layer"]
                                != [PREDICTIONS - 1] * LAYERS
                            ):
                                raise ValueError("fixed-horizon source execution contract failed")
                            row.update(
                                geometry=geo,
                                first_prefill_logit_sha256=digest(logits[0].tolist()),
                                prompt_fill_by_step=summarize_fills(fills, n),
                                phase_wall_seconds=record["phase_wall_seconds"],
                                decode_cuda_stream_ms=record["decode_cuda_ms"],
                                storage={
                                    k: storage[k]
                                    for k in (
                                        "initial_active_kv_storage_bytes",
                                        "final_active_kv_storage_bytes",
                                        "peak_active_kv_storage_bytes",
                                        "maximum_workspace_capacity",
                                        "active_workspace_bytes",
                                    )
                                },
                                representation_sha256=record["representation_sha256"],
                                whole_request_allocator_peak_bytes=record[
                                    "whole_request_allocator_peak_bytes"
                                ],
                                whole_request_allocator_reserved_peak_bytes=record[
                                    "whole_request_allocator_reserved_peak_bytes"
                                ],
                                status="completed",
                            )
                            completed += not row["warmup"]
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                            errors.append({k: row[k] for k in ("extension_units", "arm", "error")})
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
    except Exception as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        write_once(
            out / "result.json",
            dict(
                status="completed" if not errors else "failed",
                errors=errors,
                measured_rows=completed,
                source_sha=plan["source_sha"],
                plan_sha256=sha(root / "plan.json"),
                outputs={str(p.name): sha(p) for p in out.iterdir() if p.is_file()},
            ),
        )


def summarize(rows):
    expected = {
        (units, arm, repetition)
        for units in EXTENSIONS
        for arm in ARMS
        for repetition in range(-WARMUPS, MEASURED)
    }
    if (
        len(rows) != len(expected)
        or {(x["extension_units"], x["arm"], x["repetition"]) for x in rows} != expected
        or any(x["status"] != "completed" for x in rows)
    ):
        raise ValueError("incomplete, failed or duplicate scaling grid")
    lengths, visual, summary = [], [], []
    for units in EXTENSIONS:
        group = [x for x in rows if x["extension_units"] == units]
        if any(
            len({digest(x[field]) for x in group}) != 1
            for field in ("geometry", "fixture_sha256", "first_prefill_logit_sha256")
        ):
            raise ValueError("same-length arm processor input or native prefill drifted")
        geo = group[0]["geometry"]
        lengths.append(geo["decoder_prompt_positions"])
        visual.append(
            digest({k: v for k, v in geo.items() if k.startswith(("image_", "processed_"))})
        )
        for arm in ARMS:
            measured = [x for x in group if x["arm"] == arm and not x["warmup"]]
            if len(measured) != MEASURED or any(
                len({digest(x[field]) for x in measured}) != 1
                for field in ("storage", "representation_sha256")
            ):
                raise ValueError("repeated source representation or storage changed")
            summary.append(
                dict(
                    extension_units=units,
                    geometry=geo,
                    arm=arm,
                    measured_repetitions=len(measured),
                    median_prompt_fill_stream_ms_per_sampled_step=statistics.median(
                        statistics.mean(v["stream_ms"] for v in x["prompt_fill_by_step"])
                        for x in measured
                    ),
                    median_prompt_fill_host_ms_per_sampled_step=statistics.median(
                        statistics.mean(v["host_ms"] for v in x["prompt_fill_by_step"])
                        for x in measured
                    ),
                    storage=measured[0]["storage"],
                )
            )
    if lengths != sorted(set(lengths)) or len(set(visual)) != 1:
        raise ValueError("actual prompt lengths must grow while processed image stays fixed")
    return summary


def analyze(root):
    root = Path(root)
    plan = load(root)
    rows, failures, summary = [], [], []
    try:
        result = read(root / "worker/result.json")
        if (
            result["source_sha"] != plan["source_sha"]
            or result["plan_sha256"] != sha(root / "plan.json")
            or any(sha(root / "worker" / p) != s for p, s in result["outputs"].items())
        ):
            raise ValueError("scaling worker evidence changed")
        failures.extend(result["errors"])
        rows = [json.loads(line) for line in (root / "worker/raw.jsonl").read_text().splitlines()]
        summary = summarize(rows)
        if result["measured_rows"] != plan["expected_rows"] or result["status"] != "completed":
            raise ValueError("worker did not pass the bounded completion gate")
    except Exception as error:
        failures.append(dict(error=f"{type(error).__name__}: {error}"))
    write_once(root / "scaling_summary.json", dict(rows=summary, policy=POLICY))
    gate = dict(
        status="passed_synthetic_scaling" if not failures else "blocked",
        rows=len(rows),
        cells=len(summary),
        failures=failures,
        source_sha=plan["source_sha"],
        summary_sha256=sha(root / "scaling_summary.json"),
        raw_prompt_control="full_BF16_prompt_KV_common_lossless_source_history_not_native_E2E",
        direct_control="local_direct_3bit_projected_KV_not_KIVI",
        sampled_prompt_fill_stream_intervals_include_host_gaps=True,
        synthetic_scaling_not_representative_workload_quality=True,
        full_request_KV_only_peak_certified=False,
        shared_realized_bytes=False,
        headline_latency_eligible=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "GATE_source_length_scaling.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.root), indent=2))


if __name__ == "__main__":
    main()
