import pytest

from mosaickv.evaluation.precision_quality import answer_prefix


def decode(ids):
    return " ".join(map(str, ids))


def test_eos_excludes_later_performance_only_tokens():
    x = answer_prefix([10, 20, 2, 30, 40], eos=(2,), cap=5, decode=decode)
    assert x["answer"] == "10 20"
    assert x["stopped_by_eos"] and not x["output_limit_reached"]
    assert x["answer_tokens"] == 2


def test_explicit_output_limit_not_silent_truncation():
    x = answer_prefix([10, 20, 30, 2], eos=(2,), cap=3, decode=decode)
    assert x["answer"] == "10 20 30"
    assert x["output_limit_reached"] and not x["stopped_by_eos"]


@pytest.mark.parametrize("cap", [0, -1, True])
def test_invalid_limit(cap):
    with pytest.raises(ValueError, match="positive"):
        answer_prefix([1, 2], eos=(2,), cap=cap, decode=decode)


def test_offline_scoring_uses_same_resolved_snapshot_as_adapter(monkeypatch):
    import transformers

    import mosaickv.adapters.huggingface.registry as registry
    from mosaickv.evaluation.precision_quality import load_processor

    calls = []

    def resolver(model, revision, *, local_files_only):
        assert (model, revision, local_files_only) == ("model", "revision", True)
        return "/scratch/pinned-snapshot"

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        return object()

    monkeypatch.setattr(registry, "_resolve_model_source", resolver)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", loader)
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", loader)
    load_processor(dict(id="model", revision="revision"))
    assert calls == [("/scratch/pinned-snapshot", {"local_files_only": True})] * 2
