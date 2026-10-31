from types import SimpleNamespace

import evicastkv_text_row_study as driver
import pytest


def test_compact_derivation_preserves_measurements_failures_and_original_catalogs():
    original = dict(
        sample_id="id",
        task="coco",
        method="mosaickv_exact",
        precision="p",
        status="completed",
        warmup=False,
        source_sha="sha",
        plan_sha256="config",
        manifest_path="manifest",
        logits_sha256="logits",
        scored_answer={"answer": "unchanged"},
        end_to_end_seconds=2.5,
        storage={
            "initial_active_kv_storage_bytes": 100,
            "final_active_kv_storage_bytes": 200,
            "peak_active_kv_storage_bytes": 300,
            "catalog": [1, 2],
        },
        online_variant={"source_memberships": [1, 2, 3]},
    )
    compact = driver.compact_row(original)
    assert compact["scored_answer"] == original["scored_answer"]
    assert compact["storage"] == {
        "initial_active_kv_storage_bytes": 100,
        "final_active_kv_storage_bytes": 200,
        "peak_active_kv_storage_bytes": 300,
    }
    assert compact["end_to_end_seconds"] == 2.5 and compact["source_sha"] == "sha"
    assert "online_variant" not in compact and original["online_variant"]["source_memberships"] == [
        1,
        2,
        3,
    ]
    failure = dict(status="failed", error="original exception", sample_id="id")
    assert driver.compact_row(failure) == failure


def test_canary_requires_complete_rows_and_unchanged_control():
    result = dict(status="completed", failures=[])
    rows = [
        dict(
            sample_id=str(i),
            method="mosaickv_exact",
            precision=driver.POLICY,
            warmup=False,
            status="completed",
        )
        for i in range(4)
    ]
    assert driver.canary_valid(result, rows, dict(bitwise_logit_hash_match=True))
    assert not driver.canary_valid(result, rows[:-1], dict(bitwise_logit_hash_match=True))
    assert not driver.canary_valid(result, rows, dict(bitwise_logit_hash_match=False))
    assert not driver.canary_valid(
        dict(status="failed", failures=["error"]), rows, dict(bitwise_logit_hash_match=True)
    )


def test_multiarm_canary_rejects_duplicate_rows_or_unregistered_fallback():
    result = dict(status="completed", failures=[])
    policies = driver.STUDY_POLICIES
    rows = [
        dict(
            sample_id=str(i), method="mosaickv_exact", precision=p, warmup=False, status="completed"
        )
        for i in range(4)
        for p in policies
    ]
    assert driver.canary_valid(result, rows, dict(bitwise_logit_hash_match=True), policies=policies)
    rows[-1] = dict(rows[-2])
    assert not driver.canary_valid(
        result, rows, dict(bitwise_logit_hash_match=True), policies=policies
    )
    rows[-1] = dict(rows[-1], precision="unregistered_fallback")
    assert not driver.canary_valid(
        result, rows, dict(bitwise_logit_hash_match=True), policies=policies
    )


def test_constructor_labels_policy_and_rejects_unregistered_precision(monkeypatch):
    from mosaickv.evaluation import head_local_precision

    calls = []

    def bundle(*args, **kwargs):
        calls.append((args, kwargs))
        return {driver.POLICY: "packet"}, dict(source_memberships="unchanged")

    monkeypatch.setattr(head_local_precision, "bundle", bundle)
    packet, proof = driver.construct(*range(8), "query3", policy=driver.POLICY)
    assert packet == "packet" and proof["precision"] == driver.POLICY
    assert calls == [(tuple(range(8)), dict(policies=(driver.POLICY,)))]
    with pytest.raises(ValueError, match="registered"):
        driver.construct(*range(8), "ordinary", policy=driver.POLICY)


def test_text_mask_uses_original_prompt_positions_not_compressed_indices(monkeypatch):
    import torch

    from mosaickv.evaluation import head_local_precision as heads
    from mosaickv.evaluation import value_row_precision

    calls = []
    # Avoid model-dependent scoring; inspect the actual mask crossing the
    # constructor boundary. Two exact boundary layers precede the test layer.
    pair = (torch.zeros(1, 2, 20, 128, dtype=torch.bfloat16),) * 2
    memberships = ((tuple(range(10)),) * 2,) * 5
    monkeypatch.setattr(heads.request_precision, "online_frozen", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(heads, "baseline_payload", lambda *a, **k: (memberships, None, {}))
    monkeypatch.setattr(heads.Path, "read_text", lambda *a, **k: "[]")
    packet = SimpleNamespace(component_bytes=0)
    monkeypatch.setattr(heads, "construct", lambda *a, **k: packet)
    monkeypatch.setattr(heads, "prompt_weights", lambda *a, **k: (None, None))
    monkeypatch.setattr(heads, "fund_corrections", lambda *a, **k: (packet, {}))
    monkeypatch.setattr(heads, "localize", lambda *a, **k: packet)
    monkeypatch.setattr(heads, "refit_corrections", lambda *a, **k: (packet, {}))
    packet.exact_indices = torch.tensor([], dtype=torch.int32)

    def upgrade(*args, **kwargs):
        calls.append(kwargs["query_mask"].tolist())
        return packet, {}

    monkeypatch.setattr(value_row_precision, "upgrade_rows", upgrade)
    attention = SimpleNamespace(self_attn=None)
    adapter = SimpleNamespace(
        _legacy_layers=lambda _: ((pair,) * 5, None), _language_layers=lambda: [attention] * 5
    )
    prepared = SimpleNamespace(modality_map=[SimpleNamespace(modality="text", start=17, end=20)])
    prefill = SimpleNamespace(
        state=SimpleNamespace(past_key_values=None),
        attention_weights={2: torch.zeros(1, 2, 16, 20)},
    )
    heads.bundle(
        adapter,
        prepared,
        prefill,
        SimpleNamespace(summaries=False, records={2: None}),
        None,
        "mosaickv_exact",
        "fixture",
        0.5,
        policies=(driver.POLICY,),
    )
    assert calls == [[False] * 13 + [True] * 3]
