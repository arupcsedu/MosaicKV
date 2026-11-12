"""Synthetic orchestration only; no historical bodies or scheduler access."""

from copy import deepcopy
from types import SimpleNamespace

import evicastkv_parquet_recovery_v39 as driver
import pytest


def row(path="/fake/mechanism-terms/key/edges.parquet"):
    return dict(
        path=path, kind="closed_parquet", signature=[1, 2, 10, 4], original_error_sha256s=["a" * 64]
    )


def test_only_failed_or_deferred_parquet_is_selected():
    a, b = row(), row("/fake/mechanism-terms/other/edges.parquet")
    plan = dict(
        queue=[a, dict(path="/fake/good.yaml", kind="yaml_events")],
        registration_blocked=[
            dict(row=b, resource_deferred=True),
            dict(row=row("/fake/unknown.parquet"), reason="unsupported"),
        ],
    )
    assert driver.plan_rows(plan) == [a, b]


def test_duplicate_retry_candidate_rejected():
    a = row()
    with pytest.raises(ValueError, match="duplicate"):
        driver.plan_rows(
            dict(queue=[a], registration_blocked=[dict(row=a, resource_deferred=True)])
        )


def test_proof_keeps_original_error_and_registration(monkeypatch):
    candidate = row()
    proof = dict(path=candidate["path"], original_error_sha256s=["a" * 64])
    monkeypatch.setattr(driver.producers, "prove", lambda *args: deepcopy(proof))
    monkeypatch.setattr(driver.producers, "validate_proof", lambda *args: True)
    result = driver.prove_row(candidate, ["/fake"], 123, {})
    assert result["registration_sha256"] == driver.digest(candidate)
    assert result["kind"] == "closed_parquet"
    assert result["original_error_sha256s"] == ["a" * 64]


@pytest.mark.parametrize("corrupt", [False, True])
def test_wave_rejects_tampered_source_or_registration(tmp_path, monkeypatch, corrupt):
    monkeypatch.setattr(driver, "clean_source", lambda *args: "b" * 40)
    monkeypatch.setattr(driver.producers, "validate_proof", lambda proof: True)
    candidate = row()

    def run(command, **kwargs):
        index = int(command[command.index("--index") + 1])
        folder = tmp_path / "wave-0"
        spec = driver.history.read(folder / "plan.json")
        rows = [r for r in spec["queue"] if r["index"] == index]
        proofs = [
            dict(
                path=r["path"],
                signature=r["signature"],
                original_error_sha256s=r["original_error_sha256s"],
                registration_sha256=("c" * 64 if corrupt else driver.digest(r)),
            )
            for r in rows
        ]
        driver.write_once(
            folder / f"worker-{index}.json",
            dict(
                source_sha="b" * 40,
                wave=0,
                index=index,
                plan_sha256=driver.sha(folder / "plan.json"),
                proofs=proofs,
                pending=[],
                blocked=[],
                two_pass_bytes_budgeted=20 * len(rows),
            ),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    if corrupt:
        with pytest.raises(ValueError, match="original error/signature"):
            driver.run_wave(tmp_path, 0, [candidate])
    else:
        proved, remaining = driver.run_wave(tmp_path, 0, [candidate])
        assert len(proved) == 1 and not remaining


def test_canonical_cpu_failure_blocks_before_proof_canaries(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "load", lambda root: dict(source_sha="b" * 40))

    def fail(*args):
        raise ValueError("CPU did not pass")

    monkeypatch.setattr(driver, "cpu_gate", fail)
    with pytest.raises(ValueError, match="CPU did not pass"):
        driver.continue_recovery(tmp_path)
    assert not (tmp_path / "canaries.json").exists()
