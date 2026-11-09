import json

import common_baseline_headers as study


def test_registered_pairs_keep_baselines_distinct():
    arms = study.arms()
    assert len(arms) == len(set(arms)) == 8
    assert study.METHODS == (
        "prompt_attention_topk",
        "prefixkv_reimpl",
        "vl_cache_reimpl",
        "lookm_reimpl",
    )
    for before, after in zip(arms[::2], arms[1::2], strict=True):
        assert before.split(":")[0] == after.split(":")[0]
        assert before.endswith(":control") and after.endswith(":headers")


def test_incomplete_baselines_cannot_pass(tmp_path):
    study.analyze(tmp_path, {})
    proof = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert not proof["complete"]
    assert not proof["comparative_quality_or_latency_superiority_established"]
    assert not proof["cross_method_byte_matching_established"]
    assert len(proof["errors"]) == 10
