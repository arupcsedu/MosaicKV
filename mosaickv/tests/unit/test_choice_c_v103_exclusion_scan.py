from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _scanner(package: Path) -> ModuleType:
    scripts = package / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "choice_c_v103_exclusion_scan.py"
        spec = importlib.util.spec_from_file_location("mosaickv_test_v103_scan", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_v103_scan_request_is_blocked_until_resolution_commit_review(tmp_path: Path) -> None:
    package = Path(__file__).parents[2]
    scanner = _scanner(package)
    output = tmp_path / "exclusion_manifest.json"
    scanner.EXPECTED_OUTPUT = output
    with pytest.raises(PermissionError, match="scan retry"):
        scanner.require_scan_request(
            spec_path=(
                package
                / "configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml"
            ),
            output=output,
            package_root=package,
        )

    with pytest.raises(PermissionError, match="scan output"):
        scanner.require_scan_request(
            spec_path=(
                package
                / "configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml"
            ),
            output=output.with_name("different.json"),
            package_root=package,
        )


def test_v103_scan_snapshot_allowlist_excludes_runtime_and_algorithm_files() -> None:
    package = Path(__file__).parents[2]
    scanner = _scanner(package)
    assert "mosaickv/src/mosaickv/backends/hf_runtime.py" not in (
        scanner.ALLOWED_CHANGES_AFTER_SEAL
    )
    assert "mosaickv/configs/experiments/choice_c_v72_frozen_candidate.yaml" not in (
        scanner.ALLOWED_CHANGES_AFTER_SEAL
    )
    assert {
        "mosaickv/analysis/choice_c_v103_protocol_draft.md",
        "mosaickv/configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml",
        "mosaickv/configs/ledgers/choice_c_v103_protocol_scan_scope_seal.yaml",
        "mosaickv/configs/ledgers/choice_c_v103_legacy_dataset_config_resolutions.yaml",
        "mosaickv/scripts/choice_c_v103_exclusion_scan.py",
        "mosaickv/scripts/validate_choice_c_v103_protocol.py",
        "mosaickv/src/mosaickv/evaluation/disjoint_samples.py",
        "mosaickv/tests/unit/test_choice_c_v103_exclusion_scan.py",
        "mosaickv/tests/unit/test_choice_c_v103_protocol.py",
        "mosaickv/tests/unit/test_disjoint_samples.py",
    } == scanner.ALLOWED_CHANGES_AFTER_SEAL


def test_v103_scan_rejects_manifest_missing_a_mandatory_v102_ledger() -> None:
    package = Path(__file__).parents[2]
    scanner = _scanner(package)
    protocol = scanner._load_protocol(
        package / "configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml"
    )
    sources = protocol["fresh_exclusion_scan"]["mandatory_bound_sources"]
    ledgers = [source for source in sources if source.get("expected_unique_identities") is not None]
    payload = {
        "excluded_unique_sample_count": 64,
        "sources": [
            {
                "path": ledgers[0]["path"],
                "sample_count": ledgers[0]["expected_unique_identities"],
            }
        ],
    }
    with pytest.raises(ValueError, match="every mandatory V102 ledger"):
        scanner.verify_payload_coverage(protocol, payload, package, ())


def test_v103_resolution_ledger_verifies_records_or_rejects_missing_git_archive() -> None:
    package = Path(__file__).parents[2]
    scanner = _scanner(package)
    protocol = scanner._load_protocol(
        package / "configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml"
    )
    commit = "4449e6d673dba5b0f65235f9967f7b6da0b00444"
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(package.parent),
            "cat-file",
            "-e",
            f"{commit}:mosaickv/configs/experiments/choice_c_v102_v100_fresh_cross_dataset_calibration.yaml",
        ],
        capture_output=True,
        check=False,
    )
    if archive.returncode:
        # Never replace unavailable historical evidence with the live file.
        with pytest.raises(subprocess.CalledProcessError) as error:
            scanner.load_provenance_resolutions(protocol, package, require_reviewed=False)
        assert commit in " ".join(error.value.cmd)
        assert error.value.returncode != 0
        with pytest.raises(PermissionError, match="not committed and retry-authorized"):
            scanner.load_provenance_resolutions(protocol, package, require_reviewed=True)
        return
    resolutions = scanner.load_provenance_resolutions(protocol, package, require_reviewed=False)
    assert len(resolutions) == 10
    assert len({record.source_path for record in resolutions}) == 10
    assert {record.dataset_config for record in resolutions} == {"default", "InfographicVQA"}
    with pytest.raises(PermissionError, match="not committed and retry-authorized"):
        scanner.load_provenance_resolutions(protocol, package, require_reviewed=True)


def test_v103_scan_ignores_catalogs_without_reading_them(tmp_path: Path) -> None:
    package = Path(__file__).parents[2]
    scanner = _scanner(package)
    protocol = scanner._load_protocol(
        package / "configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml"
    )
    catalog = tmp_path / "catalog_sensitive.json"
    catalog.write_text("this is deliberately not parsed", encoding="utf-8")
    ledger = tmp_path / "calibration_ledger.json"
    ledger.write_text("{}", encoding="utf-8")
    ignored = scanner.catalog_paths_to_ignore(protocol, (tmp_path,), package)
    assert catalog.resolve() in ignored
    assert ledger.resolve() not in ignored
    assert all(
        Path(record["path"]).resolve() in ignored
        for record in protocol["reviewed_id_catalogs"]["catalogs"]
    )
