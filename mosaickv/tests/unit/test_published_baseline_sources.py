"""Immutable source and license registry checks for published baselines."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "third_party" / "baseline_sources.yaml"


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_published_source_registry_has_unique_required_methods() -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    sources = payload["sources"]
    by_method = {source["method"]: source for source in sources}
    assert len(by_method) == len(sources)
    assert set(by_method) == {
        "elastic_cache_reimpl",
        "h2o_reimpl",
        "streamingllm_reimpl",
        "pyramidkv_reimpl",
        "snapkv_reimpl",
        "roco_reimpl",
    }
    assert all(method.endswith("_reimpl") for method in by_method)
    assert all(len(source["audited_revision"]) == 40 for source in sources)


def test_vendored_sources_are_clean_pinned_submodules_with_licenses() -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    for source in payload["sources"]:
        if not source["vendored"]:
            assert source["license"] == "no-license-file-found-at-audited-revision"
            assert "path" not in source
            audit_path = ROOT / source["audit_path"]
            audit_text = (audit_path / "README.md").read_text(encoding="utf-8")
            assert source["audited_revision"] in audit_text
            assert not tuple(audit_path.glob("*.py"))
            continue
        checkout = ROOT / source["path"]
        assert checkout.is_dir(), f"uninitialized submodule: {checkout}"
        assert _git(checkout, "rev-parse", "HEAD") == source["audited_revision"]
        assert _git(checkout, "status", "--porcelain") == ""
        assert (checkout / "LICENSE").is_file()
