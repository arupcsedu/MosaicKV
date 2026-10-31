import json

import evicastkv_request_validation as module
import pytest


def test_canary_keeps_dataset_configs_and_strips_all_saved_prefixes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    protocol = source / "mosaickv/docs/request_precision_validation_20261030.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("frozen protocol")
    parent = tmp_path / "parent"
    parent.mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text("[0.1,0.2]")
    media = tmp_path / "image.png"
    media.write_bytes(b"fixture image not decoded")
    jobs = []
    for index, task in enumerate(("description", "mmvet")):
        config = dict(
            dataset=dict(id=task, revision="pinned", split="validation"),
            bounded_decode=dict(incremental_history=dict(layer_profile_path=str(profile))),
        )
        payload = dict(
            task=task,
            sample_id=str(index),
            prompt="question",
            fullkv_prefix_tokens=[1, 2],
            media=[dict(kind="image", path=str(media), sha256=module.sha(media))],
        )
        target = parent / f"{index}.json"
        target.write_text(json.dumps(payload))
        jobs.append(dict(input=str(target), sample_id=str(index), arms=[dict(config=config)]))
    (parent / "plan.json").write_text(json.dumps(dict(jobs=jobs)))
    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "PARENT", parent)
    monkeypatch.setattr(module, "clean_source", lambda path: "a" * 40)
    root = tmp_path / "canary"
    digest = module.prepare_canary(root)
    assert digest == module.sha(root / "plan.json")
    plan = module.read(root / "plan.json")
    assert [j["config"]["dataset"]["id"] for j in plan["jobs"]] == ["description", "mmvet"]
    assert all("fullkv_prefix_tokens" not in module.read(j["inputs"][0]) for j in plan["jobs"])
    assert len(plan["arms"]) == 11 and plan["cohort_size"] == 128
    assert plan["paper_campaign_authorized"] is False
    with pytest.raises(ValueError, match="new bounded"):
        module.prepare_canary(root)


def test_messages_fail_closed_on_answer_or_saved_token_fields():
    base = dict(task="coco_caption", sample_id="ID", prompt="question", media=[])
    for forbidden in ("references", "answer", "fullkv_prefix_tokens"):
        with pytest.raises(ValueError, match="neither references nor draft tokens"):
            module.load_messages(dict(base, **{forbidden: ["secret"]}))


def test_media_identity_ignores_png_container_but_not_pixels(tmp_path):
    from PIL import Image

    image = Image.new("RGB", (5, 3), (10, 20, 30))
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    image.save(a, compress_level=0)
    image.save(b, compress_level=9)
    assert module.sha(a) != module.sha(b)
    assert module.rgb_identity(a) == module.rgb_identity(b)
    image.putpixel((0, 0), (11, 20, 30))
    image.save(b)
    assert module.rgb_identity(a) != module.rgb_identity(b)
