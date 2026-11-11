"""CPU-only availability controls for explicitly patched LOOK-M/HF transport.

This adds GenerationMixin and declares the existing legacy tuple-cache contract
in an isolated dynamic subclass. Upstream source and attention arithmetic are
not edited. Synthetic controls do not establish real-checkpoint reproduction.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import audit_lookm_patched_import as upstream_audit

LABEL = "patched_official_hf_compat"
TESTED_TRANSFORMERS = "4.55.2"
GENERATION_SOURCE_SHA = "6de1a0fbf0b26352cbd93a2f494fc5fca1856e6636d20947e81f7f83ace7dec5"
NEW_TOKENS = 4


def compatibility_class(original, generation_mixin=None):
    """Declare transport only; inherit upstream forward/preparation verbatim."""
    if generation_mixin is None:
        from transformers.generation import GenerationMixin

        generation_mixin = GenerationMixin
    if not hasattr(generation_mixin, "_supports_default_dynamic_cache"):
        raise ValueError("Unreviewed generation-cache interface")
    wrapper = type(
        "PatchedOfficialLookMHfCompat",
        (original, generation_mixin),
        {
            "__module__": __name__,
            "_supports_default_dynamic_cache": classmethod(lambda cls: False),
            "compatibility_label": LABEL,
        },
    )
    if wrapper.forward is not original.forward:
        raise ValueError("Upstream forward changed")
    if wrapper.prepare_inputs_for_generation is not original.prepare_inputs_for_generation:
        raise ValueError("Upstream generation preparation changed")
    return wrapper


def cache_lengths(cache) -> list[int]:
    if not isinstance(cache, tuple) or not cache:
        raise ValueError("Expected a nonempty upstream legacy tuple cache")
    result = []
    for pair in cache:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or len(pair[0].shape) != 4
            or pair[0].shape != pair[1].shape
        ):
            raise ValueError("Unexpected upstream K/V pair")
        result.append(int(pair[0].shape[-2]))
    return result


@contextlib.contextmanager
def call_records(model):
    records = []

    def record(_module, _args, kwargs, output):
        positions = kwargs.get("position_ids")
        if positions is None or positions.ndim != 2 or positions.shape[0] != 1:
            raise ValueError("Cannot verify native logical positions")
        before = kwargs.get("past_key_values")
        records.append(
            {
                "position_first": int(positions[0, 0]),
                "position_last": int(positions[0, -1]),
                "position_count": int(positions.shape[1]),
                "physical_before": cache_lengths(before) if before is not None else None,
                "physical_after": cache_lengths(output.past_key_values),
            }
        )

    handle = model.register_forward_hook(record, with_kwargs=True)
    try:
        yield records
    finally:
        handle.remove()


def explicit_greedy(model, ids, *, tokens: int):
    import torch

    prompt = ids.shape[1]
    rows, emitted = [], []
    with torch.inference_mode(), call_records(model) as records:
        output = model(
            input_ids=ids,
            position_ids=torch.arange(prompt).unsqueeze(0),
            attention_mask=torch.ones_like(ids),
            use_cache=True,
        )
        for step in range(tokens):
            logits = output.logits[:, -1, :].clone()
            rows.append(logits)
            token = logits.argmax(dim=-1, keepdim=True)
            emitted.append(token)
            if step + 1 < tokens:
                output = model(
                    input_ids=token,
                    position_ids=torch.tensor([[prompt + step]]),
                    attention_mask=torch.ones((1, prompt + step + 1), dtype=torch.long),
                    past_key_values=output.past_key_values,
                    use_cache=True,
                )
    return torch.cat(emitted, dim=1), rows, records


def generation_kwargs() -> dict:
    return {
        "max_new_tokens": NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "eos_token_id": None,
        "pad_token_id": 0,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_logits": True,
    }


def verify_positions(records: list[dict], *, prompt: int, initial_physical: int) -> None:
    if len(records) != NEW_TOKENS:
        raise ValueError("Incomplete native forward-call trace")
    for step, row in enumerate(records):
        expected = prompt - 1 if step == 0 else prompt + step - 1
        if row["position_last"] != expected:
            raise ValueError("Native generation renumbered logical positions")
        if row["physical_after"] != [initial_physical + step]:
            raise ValueError("Unexpected physical-cache growth")
        expected_before = None if step == 0 else [initial_physical + step - 1]
        if row.get("physical_before") != expected_before:
            raise ValueError("Native past cache does not match preceding transaction")
        if step == 0 and (row["position_first"] != 0 or row["position_count"] != prompt):
            raise ValueError("Incomplete prefill logical positions")
        if step > 0 and (row["position_count"] != 1 or row["position_first"] != expected):
            raise ValueError("Native decode did not consume exactly one position")


def cpu_controls() -> dict:
    import torch
    from transformers.generation import GenerationMixin

    if importlib.metadata.version("transformers") != TESTED_TRANSFORMERS:
        raise ValueError("Unreviewed Transformers version; no interface guessing permitted")
    generation_source_sha = upstream_audit.digest(Path(inspect.getsourcefile(GenerationMixin)))
    if generation_source_sha != GENERATION_SOURCE_SHA:
        raise ValueError("Generation-cache driver source differs from the reviewed implementation")
    module = importlib.import_module(upstream_audit.ENTRYPOINT)
    wrapper = compatibility_class(module.LlavaLlamaForCausalLM, GenerationMixin)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config_kwargs = {
        "vocab_size": 32,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 1024,
        "pretraining_tp": 1,
        "use_cache": True,
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": None,
    }
    full = wrapper(
        module.LlavaConfig(**config_kwargs), kv_mode="origin", hh_ratio=0.1, recent_ratio=0.1
    ).eval()
    ids = (torch.arange(16) % 32).unsqueeze(0)
    reference_tokens, reference_logits, manual_records = explicit_greedy(
        full, ids, tokens=NEW_TOKENS
    )
    with torch.inference_mode(), call_records(full) as native_records:
        native = full.generate(inputs=ids, **generation_kwargs())
    native_tokens = native.sequences[:, -NEW_TOKENS:]
    if len(native.logits) != NEW_TOKENS:
        raise ValueError("Missing native-generation logits")
    differences = [
        float((a - b).abs().max()) for a, b in zip(reference_logits, native.logits, strict=True)
    ]
    if not torch.equal(native_tokens, reference_tokens) or any(value != 0 for value in differences):
        raise ValueError("Synthetic full-cache generation differs from explicit greedy reference")
    verify_positions(native_records, prompt=16, initial_physical=16)
    if native_records != manual_records:
        raise ValueError("Native and explicit cache/position transactions differ")

    compressed = wrapper(
        module.LlavaConfig(**config_kwargs),
        kv_mode="text_prior_pivot_merge",
        hh_ratio=0.1,
        recent_ratio=0.1,
    ).eval()
    for layer in compressed.model.layers:
        layer.self_attn.kv_cache.image_position = torch.tensor([2])
    synthetic_ids = (torch.arange(584) % 32).unsqueeze(0)
    with torch.inference_mode(), call_records(compressed) as compressed_records:
        embeddings = compressed.get_model().embed_tokens(synthetic_ids)
        # Synthetic embeddings bypass the image encoder, not attention or merging.
        result = GenerationMixin.generate(
            compressed,
            inputs_embeds=embeddings,
            attention_mask=torch.ones_like(synthetic_ids),
            **generation_kwargs(),
        )
    verify_positions(compressed_records, prompt=584, initial_physical=116)
    if not all(torch.isfinite(row).all().item() for row in result.logits):
        raise ValueError("Non-finite compressed synthetic logits")
    return {
        "passed": True,
        "classification": "synthetic_random_weight_cpu_only",
        "implementation_label": LABEL,
        "fullcache_tokens_equal": True,
        "maximum_fullcache_logit_difference": max(differences),
        "fullcache_native_calls": native_records,
        "compressed_native_calls": compressed_records,
        "compressed_attention_class": type(compressed.model.layers[0].self_attn).__name__,
        "compressed_cache_class": type(compressed.model.layers[0].self_attn.kv_cache).__name__,
        "compressed_entrypoint": "HF generation; synthetic embeddings; no image encoder",
        "legacy_tuple_cache_declared": wrapper._supports_default_dynamic_cache() is False,
        "generation_mixin_source_sha256": generation_source_sha,
        "packages": {"torch": torch.__version__, "transformers": TESTED_TRANSFORMERS},
        "backend": "cpu_eager",
        "precision": "float32",
        "seed": 0,
        "model": "synthetic_random_one_layer_two_head_128d_llava",
        "dataset": "not_used_synthetic_inputs",
        "vllm": "not_used",
        "sglang": "not_used",
        "gpu_count": 0,
        "attention_forward_inherited_unchanged": wrapper.forward
        is module.LlavaLlamaForCausalLM.forward,
        "published_operating_point_reproduced": False,
        "real_checkpoint_quality_validated": False,
    }


def run(upstream: Path, patch: Path, temp_root: Path) -> dict:
    upstream, patch = upstream.resolve(strict=True), patch.resolve(strict=True)
    temp_root = upstream_audit.scratch_directory(temp_root)
    head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != upstream_audit.PIN or subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    ):
        raise ValueError("Official checkout is not the clean registered pin")
    if upstream_audit.digest(upstream / upstream_audit.CONSUMER) != upstream_audit.CONSUMER_SHA:
        raise ValueError("Unregistered upstream consumer")
    result = {
        "classification": LABEL,
        "official_pin": head,
        "patch_sha256": upstream_audit.digest(patch),
        "launcher_sha256": upstream_audit.digest(Path(__file__)),
        "source_sha": subprocess.check_output(
            ["git", "-C", str(upstream_audit.REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(upstream_audit.REPO), "status", "--porcelain"], text=True
            )
        ),
        "unmodified_official": False,
        "checkpoint_parity_validated": False,
        "paper_reproduction_passed": False,
        "gpu_used": False,
        "environment_changed": False,
    }
    with tempfile.TemporaryDirectory(prefix="lookm-hf-compat-", dir=temp_root) as raw:
        isolated = Path(raw) / "LOOK-M"
        shutil.copytree(
            upstream, isolated, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
        )
        upstream_audit.verify_patch_scope(patch, isolated)
        subprocess.run(["git", "apply", str(patch)], cwd=isolated, capture_output=True, check=True)
        result["static_patch_control"] = upstream_audit.verify_patch_semantics(
            (upstream / upstream_audit.CONSUMER).read_text(),
            (isolated / upstream_audit.CONSUMER).read_text(),
        )
        if upstream_audit.digest(isolated / upstream_audit.PROVIDER) != upstream_audit.PROVIDER_SHA:
            raise ValueError("Upstream merge arithmetic changed")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    [str(isolated / "LLaVA-mix_merge_v1"), str(Path(__file__).parent)]
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        command = [sys.executable, str(Path(__file__).resolve()), "--child"]
        try:
            child = subprocess.run(
                command,
                cwd=isolated,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            child = subprocess.CompletedProcess(
                command,
                124,
                stdout=json.dumps(
                    {
                        "cpu_controls": {
                            "passed": False,
                            "exception_type": "TimeoutExpired",
                            "timeout_seconds": 90,
                        }
                    }
                ),
                stderr="Bounded offline CPU diagnostic exceeded 90 seconds",
            )
        values = []
        for line in child.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "cpu_controls" in item:
                values.append(item)
        result["child_result"] = (
            values[-1]
            if values
            else {"cpu_controls": {"passed": False, "error": "No structured output"}}
        )
        result["child_returncode"] = child.returncode
        result["stderr_tail"] = child.stderr[-12000:]
    result["official_checkout_unchanged"] = not subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"],
        text=True,
    )
    result["status"] = (
        "synthetic_compatibility_passed_only"
        if result["child_result"]["cpu_controls"].get("passed") and child.returncode == 0
        else "blocked_at_recorded_interface"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=upstream_audit.REPO / "third_party/LOOK-M")
    parser.add_argument("--patch", type=Path, default=upstream_audit.PATCH)
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/scratch/djy8hg/cache/mosaickv/tmp")),
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        try:
            value = cpu_controls()
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "cpu_controls": {
                            "passed": False,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    }
                )
            )
            return 3
        print(json.dumps({"cpu_controls": value}))
        return 0
    print(json.dumps(run(args.upstream, args.patch, args.temp_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
