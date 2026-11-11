"""Static patch limits and isolated CPU import; no model weights or GPU needed."""

import ast
import inspect
import shutil
import subprocess
from pathlib import Path

import audit_lookm_patched_import as study
import pytest


def test_exact_registered_patch_leaves_attention_and_dispatch_unchanged(tmp_path):
    upstream = study.REPO / "third_party/LOOK-M"
    target = tmp_path / study.CONSUMER
    target.parent.mkdir(parents=True)
    shutil.copyfile(upstream / study.CONSUMER, target)
    study.verify_patch_scope(study.PATCH, tmp_path)
    subprocess.run(["git", "apply", "--check", str(study.PATCH)], cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", str(study.PATCH)], cwd=tmp_path, check=True)
    report = study.verify_patch_semantics(
        (upstream / study.CONSUMER).read_text(), target.read_text()
    )
    assert report == {
        "only_registered_stale_references_removed": True,
        "arithmetic_unchanged": True,
    }
    assert 'if kv_mode == "text_prior_pivot_merge":' in target.read_text()
    assert (
        "self.model.layers[layer_idx].self_attn = TextPivotLlamaAttention_drop(config)"
        in target.read_text()
    )


def fixture_source(extra=""):
    return (
        "from .kv_token_merge.modify_llama import "
        "H2OLlamaAttention_drop, MixMerLlamaAttention_drop\n"
        "import matplotlib.pyplot as plt\n"
        "KV_DICT = {'weighted_merge': MixMerLlamaAttention_drop}\n"
        "def run(value): return value * 2\n" + extra
    )


def fixture_patched():
    return (
        "from .kv_token_merge.modify_llama import H2OLlamaAttention_drop\n"
        "def run(value): return value * 2\n"
    )


def test_correct_minimal_cleanup_is_accepted():
    assert study.verify_patch_semantics(fixture_source(), fixture_patched())["arithmetic_unchanged"]


def test_patch_touching_another_file_is_rejected(tmp_path):
    patch = tmp_path / "extra.patch"
    patch.write_text(
        study.PATCH.read_text()
        + (
            "diff --git a/unrelated.py b/unrelated.py\n"
            "--- a/unrelated.py\n+++ b/unrelated.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        )
    )
    with pytest.raises(ValueError, match="touch only"):
        study.verify_patch_scope(patch, tmp_path)


def test_arithmetic_change_is_rejected():
    with pytest.raises(ValueError, match="Unregistered"):
        study.verify_patch_semantics(
            fixture_source(), fixture_patched().replace("value * 2", "value * 3")
        )


def test_dictionary_reader_blocks_removal():
    with pytest.raises(ValueError, match="has a reader"):
        study.verify_patch_semantics(
            fixture_source("choice = KV_DICT['weighted_merge']\n"), fixture_patched()
        )


def test_used_plot_dependency_cannot_be_removed():
    with pytest.raises(ValueError, match="has a reader"):
        study.verify_patch_semantics(fixture_source("plt.plot([1])\n"), fixture_patched())


def test_unexpected_import_does_not_pass():
    with pytest.raises(ValueError, match="stale import"):
        study.verify_patch_semantics(fixture_patched(), fixture_patched())


def test_import_error_is_recorded_exactly_without_compatibility_shims(tmp_path):
    (tmp_path / "broken_import.py").write_text("raise ImportError('missing_existing_hf_api')\n")
    report = study.run_import(tmp_path, module="broken_import")
    assert report["import_passed"] is False
    assert report["exception_type"] == "ImportError"
    assert report["exception_message"] == "missing_existing_hf_api"
    assert "broken_import.py" in report["traceback"]


def test_isolated_import_is_cpu_only_and_offline(tmp_path):
    (tmp_path / "safety_import.py").write_text(
        "import os\nassert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
        "assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'\n"
    )
    assert study.run_import(tmp_path, module="safety_import")["import_passed"]
    assert not (tmp_path / "__pycache__").exists()


def test_synthetic_smoke_uses_actual_class_without_checkpoint_or_generate():
    tree = ast.parse(inspect.getsource(study.synthetic_smoke))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "LlavaLlamaForCausalLM" in attributes
    assert "from_pretrained" not in attributes
    assert "generate" not in attributes
    assert "cuda" not in attributes


def test_non_scratch_temporary_root_is_rejected():
    with pytest.raises(ValueError, match="under /scratch/djy8hg"):
        study.scratch_directory(Path("/tmp"))
