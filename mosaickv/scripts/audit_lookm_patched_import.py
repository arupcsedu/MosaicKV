"""Bounded CPU import diagnostic; never mutates the pinned official checkout.

Only a stale import, an internally unused dictionary and unused plotting import are removed in an
ephemeral scratch copy. This is a patched official-path *import diagnostic*,
not a numerical reproduction, model execution or faithful-baseline certificate.
No dependency installation, checkpoint access or GPU execution is performed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PIN = "ecf0f51a9c416c2d85e47faf2638502f01a6d748"
CONSUMER = Path("LLaVA-mix_merge_v1/llava/model/language_model/llava_llama.py")
CONSUMER_SHA = "95b18b2f8df25c498bb290d9ab090bba3fd4acfcfb1d1ae8e53f3582d6e9d546"
PROVIDER = Path("LLaVA-mix_merge_v1/llava/model/kv_token_merge/modify_llama.py")
PROVIDER_SHA = "8ae8cabe408aec6f9050f450143ac01c74ab0eda33702c1ce60e16a0dc2eefbe"
MISSING = "MixMerLlamaAttention_drop"
ENTRYPOINT = "llava.model.language_model.llava_llama"
REPO = Path(__file__).resolve().parents[2]
PATCH = REPO / "mosaickv/docs/baselines/lookm_import_cleanup_ecf0f51.patch"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_patch_scope(patch: Path, cwd: Path) -> None:
    """The patch may change only the three registered dead references."""
    stats = subprocess.check_output(
        ["git", "apply", "--numstat", str(patch)],
        cwd=cwd,
        text=True,
    ).splitlines()
    if stats != [f"1\t5\t{CONSUMER.as_posix()}"]:
        raise ValueError("Patch must touch only the registered import and unused map")


def verify_patch_semantics(before: str, after: str) -> dict:
    """Reject every AST change except the three registered dead references."""
    expected, actual = ast.parse(before), ast.parse(after)
    imports = [
        n
        for n in expected.body
        if isinstance(n, ast.ImportFrom) and n.module == "kv_token_merge.modify_llama"
    ]
    if len(imports) != 1 or sum(a.name == MISSING for a in imports[0].names) != 1:
        raise ValueError("Expected exactly one stale import")
    imports[0].names = [a for a in imports[0].names if a.name != MISSING]
    maps = [
        n
        for n in expected.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "KV_DICT" for t in n.targets)
    ]
    if (
        len(maps) != 1
        or ast.unparse(maps[0].value) != "{'weighted_merge': MixMerLlamaAttention_drop}"
    ):
        raise ValueError("Unexpected dictionary shape")
    if any(
        isinstance(n, ast.Name) and n.id == "KV_DICT" and isinstance(n.ctx, ast.Load)
        for n in ast.walk(expected)
    ):
        raise ValueError("Dictionary has a reader; removal is not permitted")
    expected.body.remove(maps[0])
    plots = [
        n
        for n in expected.body
        if isinstance(n, ast.Import)
        and [(a.name, a.asname) for a in n.names] == [("matplotlib.pyplot", "plt")]
    ]
    if len(plots) != 1 or any(
        isinstance(n, ast.Name) and n.id == "plt" for n in ast.walk(expected)
    ):
        raise ValueError("Plotting import is absent or has a reader")
    expected.body.remove(plots[0])
    if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
        raise ValueError("Unregistered code or arithmetic change")
    return {"only_registered_stale_references_removed": True, "arithmetic_unchanged": True}


def scratch_directory(raw: str | Path) -> Path:
    path = Path(raw).resolve(strict=True)
    scratch = Path("/scratch/djy8hg").resolve(strict=True)
    if not path.is_relative_to(scratch) or not path.is_dir():
        raise ValueError("Temporary directory must be under /scratch/djy8hg")
    return path


def synthetic_smoke(module) -> dict:
    """Random tiny CPU model; 128-D heads respect the upstream hard-coded merge.

    The synthetic 584-position prompt includes the documented 576-position
    image span. No image encoder, real media, checkpoint or benchmark metric
    is exercised, so success is availability evidence only.
    """
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = module.LlavaConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        pretraining_tp=1,
        use_cache=True,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = module.LlavaLlamaForCausalLM(
        config,
        kv_mode="text_prior_pivot_merge",
        hh_ratio=0.1,
        recent_ratio=0.1,
    ).eval()
    # This is the same metadata assignment made by upstream generate(), but no
    # generation wrapper or image encoder is used in this synthetic smoke.
    for layer in model.model.layers:
        layer.self_attn.kv_cache.image_position = torch.tensor([2])
    ids = (torch.arange(584) % config.vocab_size).unsqueeze(0)
    with torch.inference_mode():
        prefill = model(input_ids=ids, position_ids=torch.arange(584).unsqueeze(0), use_cache=True)
        before = prefill.past_key_values
        lengths_before = [int(pair[0].shape[-2]) for pair in before]
        decoded = model(
            input_ids=torch.tensor([[3]]),
            position_ids=torch.tensor([[584]]),
            past_key_values=before,
            use_cache=True,
        )
    lengths_after = [int(pair[0].shape[-2]) for pair in decoded.past_key_values]
    if not all(torch.isfinite(t).all().item() for t in (prefill.logits, decoded.logits)):
        raise ValueError("Non-finite synthetic logits")
    if not all(0 < length < 584 for length in lengths_before):
        raise ValueError("Synthetic merge failed to reduce cache length")
    if lengths_after != [length + 1 for length in lengths_before]:
        raise ValueError("Unexpected upstream one-token cache update")
    return {
        "passed": True,
        "classification": "synthetic_random_weight_cpu_only",
        "attention_class": type(model.model.layers[0].self_attn).__name__,
        "cache_class": type(model.model.layers[0].self_attn.kv_cache).__name__,
        "head_dimension": 128,
        "prompt_positions": 584,
        "prefill_lengths": lengths_before,
        "decode_lengths": lengths_after,
        "explicit_next_logical_position": 584,
        "prefill_logit_shape": list(prefill.logits.shape),
        "decode_logit_shape": list(decoded.logits.shape),
        "all_logits_finite": True,
        "native_generate_super_available": callable(
            getattr(super(module.LlavaLlamaForCausalLM, model), "generate", None)
        ),
        "checkpoint_or_paper_quality_validation": False,
    }


def run_import(
    python_path: Path,
    *,
    module: str = ENTRYPOINT,
    timeout: int = 90,
    run_synthetic_smoke: bool = False,
) -> dict:
    """Run a fresh CPU-only child with offline caches and a bounded timeout."""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(python_path),
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
    program = (
        "import importlib,json,traceback,sys\n"
        + inspect.getsource(synthetic_smoke)
        + "\nreport={'import_passed':False,'phase':'import'}\n"
        "\ntry:\n"
        " m=importlib.import_module(sys.argv[1]); "
        "report.update(import_passed=True,module_path=m.__file__)\n"
        " if sys.argv[2]=='1':\n"
        "  report['phase']='synthetic_forward'; report['synthetic_smoke']=synthetic_smoke(m)\n"
        " print(json.dumps(report))\n"
        "except Exception as exc:\n"
        " report.update(exception_type=type(exc).__name__,"
        "exception_message=str(exc),traceback=traceback.format_exc())\n"
        " print(json.dumps(report)); sys.exit(3)\n"
    )
    start = time.monotonic()
    try:
        child = subprocess.run(
            [sys.executable, "-c", program, module, str(int(run_synthetic_smoke))],
            cwd=python_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "import_passed": False,
            "exception_type": "TimeoutExpired",
            "timeout_seconds": timeout,
        }
    records = []
    for line in child.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "import_passed" in value:
            records.append(value)
    record = (
        records[-1]
        if records
        else {
            "import_passed": False,
            "exception_type": "UnstructuredChildFailure",
        }
    )
    record.update(
        {
            "returncode": child.returncode,
            "wall_seconds": time.monotonic() - start,
            "stderr_tail": child.stderr[-16000:],
            "cuda_visible_devices": "",
        }
    )
    if child.returncode != 0 and record.get("phase") != "synthetic_forward":
        record["import_passed"] = False
    return record


def audit(
    upstream: Path, patch: Path, temp_root: Path, *, run_synthetic_smoke: bool = False
) -> dict:
    upstream, patch = upstream.resolve(strict=True), patch.resolve(strict=True)
    temp_root = scratch_directory(temp_root)
    head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    )
    if head != PIN or dirty:
        raise ValueError("Official pin mismatch or modified official checkout")
    if digest(upstream / CONSUMER) != CONSUMER_SHA or digest(upstream / PROVIDER) != PROVIDER_SHA:
        raise ValueError("Unexpected pinned source contents")
    result = {
        "schema_version": 1,
        "classification": "patched_official_path_cpu_import_diagnostic",
        "official_pin": PIN,
        "patch_sha256": digest(patch),
        "unmodified_upstream_import_certified": False,
        "model_parity_certified": False,
        "published_operating_point_reproduced": False,
        "superiority_established": False,
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers")
        },
    }
    with tempfile.TemporaryDirectory(prefix="lookm-import-audit-", dir=temp_root) as raw:
        isolated = Path(raw) / "LOOK-M"
        shutil.copytree(
            upstream, isolated, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
        )
        verify_patch_scope(patch, isolated)
        subprocess.run(
            ["git", "apply", "--check", str(patch)], cwd=isolated, check=True, capture_output=True
        )
        subprocess.run(["git", "apply", str(patch)], cwd=isolated, check=True, capture_output=True)
        result["static_verification"] = verify_patch_semantics(
            (upstream / CONSUMER).read_text(),
            (isolated / CONSUMER).read_text(),
        )
        if digest(isolated / PROVIDER) != PROVIDER_SHA:
            raise ValueError("Patch unexpectedly changes merge implementation")
        result["patched_consumer_sha256"] = digest(isolated / CONSUMER)
        result["cpu_import"] = run_import(
            isolated / "LLaVA-mix_merge_v1",
            run_synthetic_smoke=run_synthetic_smoke,
        )
    after = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    )
    result["official_checkout_unchanged"] = (
        not after
        and digest(upstream / CONSUMER) == CONSUMER_SHA
        and digest(upstream / PROVIDER) == PROVIDER_SHA
    )
    result["status"] = (
        "import_passed_only"
        if result["cpu_import"]["import_passed"]
        else "blocked_after_narrow_import_cleanup"
    )
    if run_synthetic_smoke and result["cpu_import"]["import_passed"]:
        result["status"] = (
            "synthetic_forward_passed_only"
            if result["cpu_import"].get("synthetic_smoke", {}).get("passed")
            else "synthetic_forward_blocked"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=REPO / "third_party/LOOK-M")
    parser.add_argument("--patch", type=Path, default=PATCH)
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/scratch/djy8hg/cache/mosaickv/tmp")),
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Also test a random tiny CPU-only forward/cache path",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.upstream, args.patch, args.temp_root, run_synthetic_smoke=args.synthetic_smoke
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
