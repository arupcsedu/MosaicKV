from pathlib import Path


def test_rebased_and_synchronized_histories_agree_on_real_tiny_hf(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify_hf_adapters import tiny_adapters

    from mosaickv.evaluation.fox_history_control import diagnose_history
    from mosaickv.evaluation.messages import build_multimodal_messages

    _, adapter = tiny_adapters()[0]
    prefill = adapter.prefill(
        adapter.prepare_inputs(build_multimodal_messages("tiny source")),
        capture_queries=False,
        capture_attentions=False,
    )
    result = diagnose_history(adapter, prefill.state, int(prefill.next_token_id.item()), steps=4)
    assert result["benchmark_quality_evidence"] is False
    assert all(r["identical_source_history_before_forward"] for r in result["records"])
    assert max(r["rebased_synchronized_max_absolute"] for r in result["records"]) <= 1e-4
