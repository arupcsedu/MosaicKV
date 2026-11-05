import evicastkv_joint_key_latency as timing
import evicastkv_joint_kv_latency as wrapper
import pytest


def test_kv_wrapper_only_selects_the_frozen_correctness_variant(monkeypatch):
    names = (
        "CANDIDATE_ARM",
        "POLICY",
        "GATE_FIELD",
        "WAIT",
        "SCRIPT",
        "VARIANT",
        "JOB_NAME",
        "DOC",
        "ARMS",
        "AUXILIARY_PROFILE_ARMS",
        "APPROVED_CONTROLLER_PATHS",
    )
    for name in names:
        monkeypatch.setattr(timing, name, getattr(timing, name))
    for name in ("POLICY", "CANDIDATE_ARM", "VARIANT", "SEQUENCE", "EXPECTED_TRANSACTION"):
        monkeypatch.setattr(
            wrapper.correctness.control, name, getattr(wrapper.correctness.control, name)
        )
    wrapper.configure()
    assert timing.POLICY == "kv" and timing.CANDIDATE_ARM == "joint-kv"
    assert ("mosaickv_exact", "joint-kv") in timing.ARMS
    assert timing.GATE_FIELD == "joint_KV_128_vector_parity_and_isolation"
    assert {
        ("mosaickv_exact", "joint-kv"),
        ("prefixkv_reimpl", "graph"),
    } == timing.AUXILIARY_PROFILE_ARMS
    assert not any("src/mosaickv" in path for path in timing.APPROVED_CONTROLLER_PATHS)


def test_wrapper_never_authorizes_a_missing_kv_parity_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(timing, "GATE_FIELD", "joint_KV_128_vector_parity_and_isolation")
    monkeypatch.setattr(
        timing.driver,
        "read",
        lambda *args: dict(
            complete=True, failures=[], joint_KV_128_vector_parity_and_isolation=False
        ),
    )
    with pytest.raises(ValueError, match="positive"):
        timing.prepare(tmp_path, "source")
