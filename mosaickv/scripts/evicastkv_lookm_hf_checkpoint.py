"""Prepared bounded real-checkpoint LOOK-M compatibility control, not reproduction.

Only existing pinned local weights and the two already-consumed media inputs
are admitted. No download, new identity, metric evaluation or paper campaign.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import audit_lookm_hf_generation_compat as transport
import audit_lookm_patched_import as patching
import lookm_checkpoint_schema as checkpoint_schema
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
UPSTREAM = Path("/scratch/djy8hg/workdir/fox-review-v11-code/third_party/LOOK-M")
MODEL = Path("/scratch/djy8hg/models/mosaickv/prefixkv/Zuyan-ElasticCache/llava-v1.5-7b")
VISION = Path(
    "/scratch/djy8hg/cache/mosaickv/huggingface/hub/"
    "models--openai--clip-vit-large-patch14-336/snapshots/"
    "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
)
PARENT = Path("/scratch/djy8hg/audits/fox-review-v7-closure-20261110/source-prefix/plan.json")
PARENT_SHA = "17526fa46703e5473e83e1d6c4703bc29fcd16789652dd1f4424ae3ab0b0fe31"
MODEL_INDEX_SHA = "e843c542d65da614b748791823c2af0aa97168b969fcb3e56235b663a9fb5d19"
MODEL_CONFIG_SHA = "481fbb797321878eeae101e0c8b0b935709df2a68f7926c9835bf64f2b508561"
TOKENS = 16
ATOL = 1e-4
POLICY = {
    "label": "patched_official_hf_compat",
    "purpose": "real_checkpoint_execution_and_transport_only_not_paper_reproduction",
    "model": {
        "repo": "Zuyan/ElasticCache",
        "subfolder": "llava-v1.5-7b",
        "revision": "833edbdc7512240f2a3aa49feeb9468e2297bdbc",
    },
    "not_same_checkpoint_as_historical_HF_source_variant": True,
    "same_checkpoint_as_LOOKM_paper_not_verified": True,
    "vision_revision": "ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
    "precision": "float16",
    "attention": "upstream_eager",
    "seed": 0,
    "conversation_template": "llava_llama_2",
    "compressed_mode": "text_prior_pivot_merge",
    "hh_ratio": 0.1,
    "recent_ratio": 0.1,
    "generated_tokens": TOKENS,
    "fullkv_logit_atol": ATOL,
    "deterministic": True,
    "eos_disabled": True,
    "use_legacy_tuple_cache": True,
    "vision_weight_provider": "combined_legacy_checkpoint_all_391_verified",
    "obsolete_rotary_criterion": "source_native_ignored_serialized_buffers",
    "preserve_negative_rotary_value_equality": True,
    "no_rotary_rewrite": True,
    "prebuilt_rotary_position_limit": 2048,
    "a100s": 1,
    "cpus": 12,
    "wall_seconds": 1200,
    "soft_seconds": 1080,
    "expected_native_rows": 6,
    "no_benchmark_labels": True,
    "no_downloads": True,
    "no_fresh_cohort": True,
    "no_latency_or_quality_claim": True,
    "no_shared_budget_or_whole_request_peak_claim": True,
}
GATE = "GATE_lookm_hf_checkpoint_compat.json"


def read(path):
    return json.loads(Path(path).read_text())


def schema_control():
    """Separate CPU process: meta-only graph and mmap-only checkpoint inspection."""
    with tempfile.TemporaryDirectory(
        prefix="lookm-checkpoint-schema-",
        dir=patching.scratch_directory(os.environ["TMPDIR"]),
    ) as raw:
        isolated = Path(raw) / "LOOK-M"
        shutil.copytree(
            UPSTREAM, isolated, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
        )
        patching.verify_patch_scope(patching.PATCH, isolated)
        subprocess.run(
            ["git", "apply", str(patching.PATCH)], cwd=isolated, check=True, capture_output=True
        )
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(isolated / "LLaVA-mix_merge_v1"),
                        str(Path(__file__).parent),
                        str(SOURCE / "mosaickv/src"),
                    ]
                ),
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        try:
            child = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "schema-child"],
                cwd=isolated,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            return {"schema": {"passed": False, "exception_type": "TimeoutExpired"}}
        records = []
        for line in child.stdout.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "schema" in value:
                records.append(value)
        return (
            records[-1]
            if records
            else {
                "schema": {
                    "passed": False,
                    "exception_type": "UnstructuredChildFailure",
                    "returncode": child.returncode,
                    "stderr": child.stderr[-8000:],
                },
            }
        )


def input_geometry():
    """Prompt-only preprocessing bound; never open benchmark annotations."""
    from transformers import AutoTokenizer

    conversation = importlib.import_module("llava.conversation")
    mm = importlib.import_module("llava.mm_utils")
    constants = importlib.import_module("llava.constants")
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("Consumed parent changed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    result = []
    for job in read(PARENT)["jobs"]:
        if sha(Path(job["inputs"][0])) != job["input_sha256"]:
            raise ValueError("Consumed input binding changed")
        payload = blind_payload(job["inputs"][0])
        conv = conversation.conv_templates[POLICY["conversation_template"]].copy()
        conv.append_message(conv.roles[0], constants.DEFAULT_IMAGE_TOKEN + "\n" + payload["prompt"])
        conv.append_message(conv.roles[1], None)
        ids = mm.tokenizer_image_token(
            conv.get_prompt(), tokenizer, constants.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        if int((ids == constants.IMAGE_TOKEN_INDEX).sum()) != 1:
            raise ValueError("Exactly one original image marker required")
        vision_cfg = read(VISION / "config.json")
        vision_cfg = vision_cfg.get("vision_config", vision_cfg)
        if (vision_cfg["image_size"], vision_cfg["patch_size"]) != (336, 14):
            raise ValueError("Unreviewed vision token expansion")
        expanded = ids.numel() + 575
        result.append(
            {
                "input_sha256": job["input_sha256"],
                "raw_token_count": ids.numel(),
                "expanded_prompt_length": expanded,
                "required_position_length": expanded + TOKENS,
            }
        )
    if len(result) != 2:
        raise ValueError("Only two source-bound consumed inputs allowed")
    return result


def blind_payload(path):
    payload = read(path)
    if set(payload) != {"task", "sample_id", "prompt", "media"}:
        raise ValueError("Only consumed prompt/media/identity fields are permitted")
    if not isinstance(payload["prompt"], str) or "<image>" in payload["prompt"]:
        raise ValueError("Raw single-image prompt required, not pretemplated input")
    if (
        len(payload["media"]) != 1
        or set(payload["media"][0]) != {"kind", "path", "sha256"}
        or payload["media"][0]["kind"] != "image"
    ):
        raise ValueError("Exactly one hash-bound consumed image required")
    return payload


def weight_files(model, vision):
    index = read(model / "pytorch_model.bin.index.json")
    names = sorted(set(index["weight_map"].values()))
    if not names or any(Path(name).name != name for name in names):
        raise ValueError("Unregistered weight-index path")
    files = [model / name for name in names]
    files += [
        model / name
        for name in (
            "config.json",
            "pytorch_model.bin.index.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        )
    ]
    vision_files = [p for p in vision.iterdir() if p.is_file()]
    if not any(p.name in {"pytorch_model.bin", "model.safetensors"} for p in vision_files):
        raise ValueError("Vision weights are unavailable locally")
    return files + vision_files


def prepare(root, *, cpu_gate_root=None):
    root = Path(root).resolve()
    if (
        not root.is_relative_to(Path("/scratch/djy8hg").resolve())
        or root.is_relative_to(SOURCE.resolve())
        or root.exists()
    ):
        raise ValueError("New write-once scratch diagnostic root required")
    revision = clean_source(SOURCE)
    if cpu_gate_root is None:
        raise ValueError("Current immutable-source CPU gate root is required")
    cpu_gate_root = Path(cpu_gate_root)
    cpu_gate(cpu_gate_root, revision)
    registration = read(cpu_gate_root / "registration.json")
    if registration.get("source_sha") != revision or registration.get("execute") is not True:
        raise ValueError("Executing current-source parent registration required")
    cpu_compat = transport.run(UPSTREAM, patching.PATCH, Path(os.environ["TMPDIR"]))
    write_once(root / "cpu_compatibility.json", cpu_compat)
    if cpu_compat["status"] != "synthetic_compatibility_passed_only":
        raise ValueError("Current-source synthetic compatibility control did not pass")
    schema = schema_control()
    write_once(root / "checkpoint_schema.json", schema)
    if schema["schema"].get("passed") is not True:
        raise ValueError("Actual checkpoint schema/rotary validation did not pass")
    if sha(MODEL / "pytorch_model.bin.index.json") != MODEL_INDEX_SHA:
        raise ValueError("Cached legacy checkpoint index changed; no HF-format conversion allowed")
    if sha(MODEL / "config.json") != MODEL_CONFIG_SHA:
        raise ValueError("Cached legacy checkpoint configuration changed")
    config = read(MODEL / "config.json")
    if config.get("model_type") != "llava" or config.get("hidden_size") != 4096:
        raise ValueError("Unsupported cached original-format checkpoint")
    if config.get("mm_use_im_patch_token", True) or config.get("mm_use_im_start_end", True):
        raise ValueError("Token-resizing path is not registered")
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("Consumed parent protocol changed")
    parent = read(PARENT)
    jobs, files = (
        [],
        [
            PARENT,
            patching.PATCH,
            Path(__file__),
            Path(transport.__file__),
            Path(patching.__file__),
            Path(checkpoint_schema.__file__),
            SOURCE / "mosaickv/env/common/requirements.lock",
            root / "cpu_compatibility.json",
            root / "checkpoint_schema.json",
        ],
    )
    files += [
        cpu_gate_root / "registration.json",
        cpu_gate_root / "full-cpu/cpu-tests.json",
        cpu_gate_root / "full-cpu/cpu-tests.xml",
        cpu_gate_root / "full-cpu/cpu-tests.log",
    ]
    if len(parent["jobs"]) != 2:
        raise ValueError("Exactly the existing two consumed inputs are allowed")
    for job in parent["jobs"]:
        path = Path(job["inputs"][0])
        if sha(path) != job["input_sha256"]:
            raise ValueError("Consumed identity binding changed")
        payload = blind_payload(path)
        for item in payload["media"]:
            media = Path(item["path"])
            if sha(media) != item["sha256"]:
                raise ValueError("Consumed media changed")
            files.append(media)
        jobs.append({"input": str(path), "input_sha256": sha(path)})
        files.append(path)
    if (
        subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
        != patching.PIN
    ):
        raise ValueError("Wrong upstream pin")
    if subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True):
        raise ValueError("Pinned upstream checkout changed")
    for name in (
        "llava/model/builder.py",
        "llava/model/language_model/llava_llama.py",
        "llava/model/kv_token_merge/modify_llama.py",
        "llava/model/kv_token_merge/v433_modeling_llama.py",
        "llava/model/llava_arch.py",
        "llava/model/multimodal_encoder/clip_encoder.py",
        "llava/mm_utils.py",
        "llava/conversation.py",
        "llava/constants.py",
    ):
        files.append(UPSTREAM / "LLaVA-mix_merge_v1" / name)
    files += weight_files(MODEL, VISION)
    bindings = {str(path): sha(path) for path in files}
    plan = {
        "source_sha": revision,
        "source_root": str(SOURCE),
        "policy": POLICY,
        "policy_sha256": digest(POLICY),
        "bindings": bindings,
        "jobs": jobs,
        "upstream": str(UPSTREAM),
        "model_directory": str(MODEL),
        "vision_directory": str(VISION),
        "checkpoint_schema": str(root / "checkpoint_schema.json"),
        "gpu_execution_authorized": True,
        "paper_reproduction_authorized": False,
        "fresh_confirmation_authorized": False,
    }
    write_once(root / "protocol.json", plan)
    return {
        "workers": 1,
        "expected_native_rows": 6,
        "protocol_sha256": sha(root / "protocol.json"),
        "resources": {"gpus": 1, "cpus": 12, "time_limit_seconds": 1200},
        "gpu_execution_authorized": True,
    }


def validate(root):
    plan = read(Path(root) / "protocol.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy"] != POLICY or plan["policy_sha256"] != digest(POLICY):
        raise ValueError("Registered compatibility policy changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Immutable diagnostic dependency changed: {name}")
    return plan


def live_cache_bytes(cache):
    owners = {}
    for pair in cache:
        for tensor in pair:
            storage = tensor.untyped_storage()
            owners[(str(tensor.device), storage.data_ptr(), storage._cdata)] = storage.nbytes()
    return sum(owners.values())


def reset_request(model):
    for layer in model.model.layers:
        attention = layer.self_attn
        if hasattr(attention, "_clean_cache"):
            attention._clean_cache()
            if attention.kv_cache.hh_score is not None:
                raise ValueError("Upstream per-request attention scores were not cleared")


def assigned_isolation(snapshot, assigned_uuid, pid):
    others = [p.pid for p in snapshot.processes if p.gpu_uuid == assigned_uuid and p.pid != pid]
    known = any(device.uuid == assigned_uuid for device in snapshot.devices)
    return {
        "assigned_gpu_uuid": assigned_uuid,
        "assigned_gpu_background_pids": others,
        "assigned_gpu_isolation_verified": known and not others and snapshot.query_error is None,
    }


def _checkpoint_run(plan, out, *, started):
    import torch
    from PIL import Image
    from transformers import AutoTokenizer, GenerationMixin

    from mosaickv.measurements.telemetry import capture_gpu_environment

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one allocated CUDA device is required")
    if "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("Only the registered A100 device family is allowed")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(12)
    torch.use_deterministic_algorithms(True)
    if importlib.metadata.version("transformers") != transport.TESTED_TRANSFORMERS:
        raise ValueError("Unreviewed Transformers interface")
    if sha(Path(inspect.getsourcefile(GenerationMixin))) != transport.GENERATION_SOURCE_SHA:
        raise ValueError("Unreviewed generation driver")
    snapshot = capture_gpu_environment(torch)
    environment = dataclasses.asdict(snapshot)
    assigned_uuid = str(torch.cuda.get_device_properties(0).uuid)
    isolation = assigned_isolation(snapshot, assigned_uuid, os.getpid())
    environment.update(isolation)
    write_once(out / "gpu_environment.json", environment)
    if not isolation["assigned_gpu_isolation_verified"]:
        raise ValueError("Assigned-GPU isolation is not verified")
    module = importlib.import_module(patching.ENTRYPOINT)
    modify = importlib.import_module("llava.model.kv_token_merge.modify_llama")
    mm = importlib.import_module("llava.mm_utils")
    constants = importlib.import_module("llava.constants")
    conversation = importlib.import_module("llava.conversation")
    wrapper = checkpoint_schema.checkpoint_compatibility_class(
        module.LlavaLlamaForCausalLM,
        plan["vision_directory"],
    )
    cfg = module.LlavaConfig.from_pretrained(plan["model_directory"], local_files_only=True)
    original_vision_name = cfg.mm_vision_tower
    cfg.mm_vision_tower = plan["vision_directory"]
    model, loading = wrapper.from_pretrained(
        plan["model_directory"],
        config=cfg,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        output_loading_info=True,
        kv_mode="origin",
        hh_ratio=0.1,
        recent_ratio=0.1,
    )
    schema = read(plan["checkpoint_schema"])["schema"]
    checkpoint_schema.verify_loading_info(loading, schema)
    index = read(MODEL / "pytorch_model.bin.index.json")["weight_map"]
    with checkpoint_schema.MappedCheckpoint(MODEL, index) as mapped:
        vision_values = checkpoint_schema.verify_vision_values(model.state_dict(), mapped, schema)
    write_once(
        out / "checkpoint_loading.json",
        {
            "strict_loading_info": loading,
            "vision_values": vision_values,
            "obsolete_rotary_buffers_verified_before_loading": schema["rotary_checks"],
            "deviation": "public_CLIP_constructor_binds_combined_checkpoint_vision_before_load",
            "unmodified_official": False,
        },
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model_directory"], use_fast=False, local_files_only=True
    )
    if len(tokenizer) != cfg.vocab_size:
        raise ValueError("Unregistered token-embedding resize would be required")
    tower = model.get_vision_tower()
    tower.load_model(device_map={"": "cuda:0"})
    tower.to(device="cuda:0", dtype=torch.float16)
    model.generation_config.eos_token_id = None
    rows, tensors, prepared = [], {}, []
    for job in plan["jobs"]:
        payload = blind_payload(job["input"])
        template = conversation.conv_templates[POLICY["conversation_template"]].copy()
        template.append_message(
            template.roles[0], constants.DEFAULT_IMAGE_TOKEN + "\n" + payload["prompt"]
        )
        template.append_message(template.roles[1], None)
        ids = (
            mm.tokenizer_image_token(
                template.get_prompt(), tokenizer, constants.IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        with Image.open(payload["media"][0]["path"]) as picture:
            pixels = mm.process_images(
                [picture.convert("RGB")], tower.image_processor, model.config
            )
        if not isinstance(pixels, torch.Tensor):
            raise ValueError("Unregistered image-packing layout")
        prepared.append((ids, pixels.to(device="cuda:0", dtype=torch.float16)))

    def native(case, label):
        if time.monotonic() - started > POLICY["soft_seconds"]:
            raise TimeoutError("Bounded compatibility diagnostic exceeded its soft deadline")
        reset_request(model)
        write_once(
            out / "attempts" / f"{label}.json",
            {
                "case_index": case,
                "arm": label,
                "input_sha256": plan["jobs"][case]["input_sha256"],
                "status": "started",
            },
        )
        ids, pixels = prepared[case]
        records = []

        def hook(_model, _args, kwargs, output):
            pos = kwargs.get("position_ids")
            if pos is None:
                raise ValueError("Cannot observe actual logical positions")
            cache = output.past_key_values
            records.append(
                {
                    "position_last": int(pos[0, -1]),
                    "physical_lengths": transport.cache_lengths(cache),
                    "active_bytes": live_cache_bytes(cache),
                    "tensor_payload_bytes": sum(
                        t.numel() * t.element_size() for pair in cache for t in pair
                    ),
                }
            )

        handle = model.register_forward_hook(hook, with_kwargs=True)
        try:
            with (
                torch.inference_mode(),
                checkpoint_schema.rotary_runtime_guard(model, schema) as rotary_guard,
            ):
                result = model.generate(
                    inputs=ids,
                    images=pixels,
                    **dict(transport.generation_kwargs(), max_new_tokens=TOKENS),
                )
        finally:
            handle.remove()
        logits = torch.stack(tuple(t.detach().float().cpu()[0] for t in result.logits))
        tokens = result.sequences[:, -TOKENS:].detach().cpu()
        if logits.shape[0] != TOKENS or len(records) != TOKENS or not torch.isfinite(logits).all():
            raise ValueError("Incomplete/nonfinite actual checkpoint output")
        prompt_length = records[0]["position_last"] + 1
        if prompt_length != schema["registered_input_geometry"][case]["expanded_prompt_length"]:
            raise ValueError("Runtime expanded prompt differs from registered geometry")
        if rotary_guard["calls_per_layer"] != [TOKENS] * len(model.model.layers):
            raise ValueError("Incomplete per-layer rotary runtime guard")
        expected_positions = list(range(prompt_length - 1, prompt_length + TOKENS - 1))
        if [r["position_last"] for r in records] != expected_positions:
            raise ValueError("Original logical positions were renumbered")
        initial = records[0]["physical_lengths"]
        for step, record in enumerate(records):
            if record["physical_lengths"] != [n + step for n in initial]:
                raise ValueError("Legacy cache append transaction changed")
        row = {
            "case_index": case,
            "arm": label,
            "input_sha256": plan["jobs"][case]["input_sha256"],
            "logical_prompt_length": prompt_length,
            "calls": records,
            "initial_kv_bytes": records[0]["active_bytes"],
            "final_kv_bytes": records[-1]["active_bytes"],
            "observed_decode_output_peak_bytes": max(r["active_bytes"] for r in records),
            "whole_request_peak_certified": False,
            "complete": True,
            "attention_class": type(model.model.layers[0].self_attn).__name__,
            "kv_mode": model.kv_mode,
            "source_native_rotary_guard": rotary_guard,
        }
        rows.append(row)
        tensors[label] = {"logits": logits, "tokens": tokens}
        torch.save(tensors[label], out / f"{label}.pt")
        write_once(out / "rows" / f"{label}.json", row)
        del result
        reset_request(model)
        return logits, tokens

    for case in range(2):
        native_logits, native_tokens = native(case, f"fullkv_{case}")
        ids, pixels = prepared[case]
        with torch.inference_mode(), checkpoint_schema.rotary_runtime_guard(model, schema):
            _ids, _pos, _mask, past, embeds, _labels = model.prepare_inputs_labels_for_multimodal(
                ids,
                None,
                None,
                None,
                None,
                pixels,
            )
            if embeds is None or past is not None:
                raise ValueError("Unsupported upstream multimodal preparation")
            length = embeds.shape[1]
            output = model(
                inputs_embeds=embeds,
                position_ids=torch.arange(length, device="cuda").unsqueeze(0),
                use_cache=True,
            )
            manual_logits, manual_tokens = [], []
            for step in range(TOKENS):
                z = output.logits[:, -1, :]
                token = z.argmax(-1, keepdim=True)
                manual_logits.append(z.detach().float().cpu()[0])
                manual_tokens.append(token.cpu())
                if step + 1 < TOKENS:
                    output = model(
                        input_ids=token,
                        position_ids=torch.tensor([[length + step]], device="cuda"),
                        past_key_values=output.past_key_values,
                        use_cache=True,
                    )
        difference = float((torch.stack(manual_logits) - native_logits).abs().max())
        write_once(
            out / f"fullkv_parity_{case}.json",
            {
                "case_index": case,
                "maximum_logit_difference": difference,
                "registered_atol": ATOL,
                "tokens_equal": torch.equal(torch.cat(manual_tokens, 1), native_tokens),
                "token_count": TOKENS,
            },
        )
        tensors[f"fullkv_explicit_{case}"] = {
            "logits": torch.stack(manual_logits),
            "tokens": torch.cat(manual_tokens, 1),
        }
        torch.save(tensors[f"fullkv_explicit_{case}"], out / f"fullkv_explicit_{case}.pt")
        if difference > ATOL or not torch.equal(torch.cat(manual_tokens, 1), native_tokens):
            raise ValueError("FullKV native-vs-explicit 16-token parity failed")
        rows[-1]["explicit_parity"] = {
            "passed": True,
            "max_logit_difference": difference,
            "tokens_equal": True,
            "tokens": TOKENS,
        }
        del output, embeds

    originals = [layer.self_attn for layer in model.model.layers]
    for layer in model.model.layers:
        replacement = modify.TextPivotLlamaAttention_drop(model.config).to(
            device="cuda:0", dtype=torch.float16
        )
        replacement.load_state_dict(layer.self_attn.state_dict(), strict=True)
        replacement.eval()
        layer.self_attn = replacement
    model.kv_mode = POLICY["compressed_mode"]
    for case, label in ((0, "compressed_A"), (1, "compressed_B"), (0, "compressed_A_repeat")):
        native(case, label)
        if not all(
            n < rows[-1]["logical_prompt_length"] for n in rows[-1]["calls"][0]["physical_lengths"]
        ):
            raise ValueError("Actual upstream compression did not execute")
    isolation = {
        "tokens_equal": torch.equal(
            tensors["compressed_A"]["tokens"], tensors["compressed_A_repeat"]["tokens"]
        ),
        "logits_equal": torch.equal(
            tensors["compressed_A"]["logits"], tensors["compressed_A_repeat"]["logits"]
        ),
    }
    write_once(out / "compressed_ABA_isolation.json", isolation)
    if not all(isolation.values()):
        raise ValueError("Compressed A-B-A request isolation failed")
    for layer, original in zip(model.model.layers, originals, strict=True):
        layer.self_attn = original
    model.kv_mode = "origin"
    native(0, "fullkv_A_restored")
    if not torch.equal(
        tensors["fullkv_0"]["tokens"], tensors["fullkv_A_restored"]["tokens"]
    ) or not torch.equal(tensors["fullkv_0"]["logits"], tensors["fullkv_A_restored"]["logits"]):
        raise ValueError("Restored native class changed FullKV output")
    torch.save(tensors, out / "logits_tokens.pt")
    write_once(out / "rows.json", rows)
    write_once(
        out / "runtime.json",
        {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda": torch.version.cuda,
            "seed": 0,
            "model": POLICY["model"],
            "vision_location_original": original_vision_name,
            "vision_location_pinned": plan["vision_directory"],
            "processor_config_sha256": sha(VISION / "preprocessor_config.json"),
            "vision_attention_implementation": tower.vision_tower.config._attn_implementation,
            "policy_sha256": digest(POLICY),
            "vllm": "not_used",
            "sglang": "not_used",
            "synthetic_weights": False,
            "paper_reproduction": False,
        },
    )
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    return {"rows": len(rows), "A_B_A_isolation": True, "restored_fullkv_identity": True}


def worker(root, index=0):
    started = time.monotonic()
    root = Path(root)
    plan = validate(root)
    if index != 0 or plan["gpu_execution_authorized"] is not True:
        raise ValueError("Exactly the registered one-GPU worker must be authorized")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Registered Slurm execution is required; no login-node CUDA run")
    out = root / "worker"
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "complete": False,
        "source_sha": plan["source_sha"],
        "protocol_sha256": sha(root / "protocol.json"),
        "implementation": POLICY["label"],
        "status": "blocked",
        "errors": [],
    }
    try:
        with tempfile.TemporaryDirectory(
            prefix="lookm-checkpoint-", dir=patching.scratch_directory(os.environ["TMPDIR"])
        ) as raw:
            isolated = Path(raw) / "LOOK-M"
            shutil.copytree(
                UPSTREAM, isolated, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
            )
            patching.verify_patch_scope(patching.PATCH, isolated)
            subprocess.run(
                ["git", "apply", str(patching.PATCH)], cwd=isolated, check=True, capture_output=True
            )
            patching.verify_patch_semantics(
                (UPSTREAM / patching.CONSUMER).read_text(),
                (isolated / patching.CONSUMER).read_text(),
            )
            if "llava" in sys.modules:
                raise ValueError("Foreign llava module already imported")
            for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
                os.environ[name] = "1"
            sys.path.insert(0, str(isolated / "LLaVA-mix_merge_v1"))
            try:
                extra = _checkpoint_run(plan, out, started=started)
            finally:
                sys.path.pop(0)
        result.update(extra, complete=True, status="passed_execution_only")
    except Exception as exc:
        result["errors"].append(
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    result["outputs"] = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}
    write_once(out / "result.json", result)
    return result


def analyze(root):
    root = Path(root)
    plan = validate(root)
    failures = []
    result_path = root / "worker/result.json"
    result = read(result_path) if result_path.exists() else {}
    if (
        result.get("complete") is not True
        or result.get("rows") != POLICY["expected_native_rows"]
        or result.get("status") != "passed_execution_only"
        or result.get("errors") != []
    ):
        failures.append("incomplete_real_checkpoint_controls")
    for name, expected in result.get("outputs", {}).items():
        if sha(root / "worker" / name) != expected:
            failures.append("changed_worker_artifact")
    if result.get("protocol_sha256") != sha(root / "protocol.json"):
        failures.append("protocol_binding_mismatch")
    if not failures:
        try:
            import torch

            values = torch.load(
                root / "worker/logits_tokens.pt", map_location="cpu", weights_only=True
            )
            verify_saved_tensors(values, expected_vocab=read(MODEL / "config.json")["vocab_size"])
            schema = read(plan["checkpoint_schema"])["schema"]
            verify_rows(
                read(root / "worker/rows.json"),
                expected_required_length=schema["source_native_irrelevance_proof"][
                    "required_position_length"
                ],
            )
        except Exception as exc:
            failures.append(f"independent_artifact_control: {type(exc).__name__}: {exc}")
    gate = {
        "status": "passed_execution_only" if not failures else "blocked",
        "failures": failures,
        "source_sha": plan["source_sha"],
        "protocol_sha256": sha(root / "protocol.json"),
        "implementation": POLICY["label"],
        "real_checkpoint_control_passed": not failures,
        "published_operating_point_reproduced": False,
        "benchmark_quality_claim": False,
        "superiority_established": False,
        "fresh_confirmation_authorized": False,
    }
    write_once(root / GATE, gate)
    return gate


def verify_saved_tensors(values, *, expected_vocab):
    """Recheck numerical/isolation controls without interpreting generated text."""
    import torch

    names = {
        "fullkv_0",
        "fullkv_1",
        "fullkv_explicit_0",
        "fullkv_explicit_1",
        "compressed_A",
        "compressed_B",
        "compressed_A_repeat",
        "fullkv_A_restored",
    }
    if set(values) != names:
        raise ValueError("Complete native and explicit tensor records required; no intersection")
    for item in values.values():
        if (
            item["logits"].shape != (TOKENS, expected_vocab)
            or item["tokens"].shape != (1, TOKENS)
            or not torch.isfinite(item["logits"]).all()
        ):
            raise ValueError("Incomplete or nonfinite saved checkpoint tensors")
    for case in range(2):
        actual, reference = values[f"fullkv_{case}"], values[f"fullkv_explicit_{case}"]
        if (
            not torch.equal(actual["tokens"], reference["tokens"])
            or float((actual["logits"] - reference["logits"]).abs().max()) > ATOL
        ):
            raise ValueError("Saved FullKV token/logit parity failed")
    for a, b in (("compressed_A", "compressed_A_repeat"), ("fullkv_0", "fullkv_A_restored")):
        if any(not torch.equal(values[a][key], values[b][key]) for key in ("tokens", "logits")):
            raise ValueError("Saved A-B-A isolation or native restoration failed")


def verify_rows(rows, *, expected_required_length=None):
    expected = {
        "fullkv_0",
        "fullkv_1",
        "compressed_A",
        "compressed_B",
        "compressed_A_repeat",
        "fullkv_A_restored",
    }
    if len(rows) != len(expected) or {r["arm"] for r in rows} != expected:
        raise ValueError("Complete unique six-arm row records required")
    for row in rows:
        calls, prompt = row["calls"], row["logical_prompt_length"]
        if row["complete"] is not True or len(calls) != TOKENS:
            raise ValueError("Incomplete native cache/position trace")
        guard = row["source_native_rotary_guard"]
        if (
            guard["calls_per_layer"] != [TOKENS] * 32
            or guard["prebuilt_length"] != POLICY["prebuilt_rotary_position_limit"]
            or not 0 < guard["registered_required_length"] < guard["prebuilt_length"]
            or any(
                not 0 < n <= guard["registered_required_length"]
                for n in guard["maximum_length_per_layer"]
            )
            or (
                expected_required_length is not None
                and guard["registered_required_length"] != expected_required_length
            )
        ):
            raise ValueError("Incomplete or changed saved source-native rotary guard")
        initial = calls[0]["physical_lengths"]
        if len(initial) != 32 or any(type(n) is not int or not 0 < n <= prompt for n in initial):
            raise ValueError("Invalid native cache layout")
        if row["arm"].startswith("compressed") and any(n >= prompt for n in initial):
            raise ValueError("Actual compression is missing")
        if row["arm"].startswith("compressed") and (
            row["attention_class"] != "TextPivotLlamaAttention_drop"
            or row["kv_mode"] != POLICY["compressed_mode"]
        ):
            raise ValueError("Unexpected compressed implementation or silent fallback")
        if row["arm"].startswith("fullkv") and initial != [prompt] * 32:
            raise ValueError("FullKV is not full length")
        if row["arm"].startswith("fullkv") and row["kv_mode"] != "origin":
            raise ValueError("FullKV did not use the original native mode")
        for step, call in enumerate(calls):
            if call["position_last"] != prompt - 1 + step:
                raise ValueError("Saved logical positions were renumbered")
            if call["physical_lengths"] != [n + step for n in initial]:
                raise ValueError("Saved physical cache transactions differ")
            if not 0 < call["tensor_payload_bytes"] <= call["active_bytes"]:
                raise ValueError("Invalid active-backing accounting")
        if (
            row["initial_kv_bytes"] != calls[0]["active_bytes"]
            or row["final_kv_bytes"] != calls[-1]["active_bytes"]
            or row["observed_decode_output_peak_bytes"] != max(c["active_bytes"] for c in calls)
        ):
            raise ValueError("Saved boundary accounting differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze", "schema-child"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--cpu-gate-root", type=Path)
    args = parser.parse_args()
    if args.mode == "schema-child":
        try:
            module = importlib.import_module(patching.ENTRYPOINT)
            geometry = input_geometry()
            value = checkpoint_schema.meta_audit(
                module.LlavaLlamaForCausalLM,
                MODEL,
                VISION,
                read(MODEL / "pytorch_model.bin.index.json")["weight_map"],
                required_length=max(row["required_position_length"] for row in geometry),
            )
            value["registered_input_geometry"] = geometry
        except Exception as exc:
            value = {
                "passed": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        print(json.dumps({"schema": value}, sort_keys=True))
        return
    if args.root is None:
        parser.error("--root is required outside the private CPU schema child")
    if args.mode == "prepare":
        result = prepare(args.root, cpu_gate_root=args.cpu_gate_root)
    else:
        result = worker(args.root) if args.mode == "worker" else analyze(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
