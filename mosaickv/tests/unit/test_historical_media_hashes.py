"""Finite synthetic ledgers/PNGs; no real historical image access."""

import io
import json
from pathlib import Path

import evicastkv_historical_media_hashes as historical
import pytest
from PIL import Image


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return historical.sha(path)


def png(color=(10, 20, 30)):
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), color).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    data = tmp_path / "images"
    data.mkdir()
    files = []
    for index in range(3):
        path = data / f"image-{index}.png"
        path.write_bytes(png())
        files.append(path)
    ledgers = []
    for index in range(2):
        path = tmp_path / f"coco-{index}.json"
        selected = files[:2] if index == 0 else files[2:]
        value = dict(
            dataset_id="lmms-lab/COCO-Caption",
            samples=[
                dict(
                    sample_id=f"{index}-{j}",
                    prompt="DO_NOT_READ_PROMPT",
                    references=["DO_NOT_READ_LABEL"],
                    media=[dict(path=str(p), sha256=historical.sha(p))],
                )
                for j, p in enumerate(selected)
            ],
        )
        ledgers.append((str(path), put(path, value)))
    extra = []
    for index in range(5):
        path = tmp_path / f"ID-only-{index}.json"
        extra.append(
            (str(path), put(path, dict(dataset_id=f"dataset-{index}", sample_ids=[f"id-{index}"])))
        )
    typed = tmp_path / "media_identities.json"
    typed_sha = put(
        typed, dict(identities={"old-asset": historical.image_identity(png((40, 50, 60)))})
    )
    monkeypatch.setattr(historical, "recovered_ledgers", lambda root: (extra, {}))
    monkeypatch.setattr(historical, "clean_source", lambda *args: "f" * 40)
    monkeypatch.setattr(historical, "cpu_gate", lambda *args: dict(status="passed"))
    return dict(
        root=tmp_path / "audit",
        cpu_gate_root=tmp_path,
        recovery_root=tmp_path / "recipe",
        coco_ledgers=ledgers,
        typed_media=typed,
        typed_media_sha256=typed_sha,
        allowed_roots=[data],
    ), files


def prepare(fixture):
    args, _files = fixture
    return historical.prepare(**args)


def test_prepare_recognizes_only_media_pairs_and_keeps_five_ID_only_gaps(fixture, monkeypatch):
    monkeypatch.setattr(
        historical, "image_identity", lambda *args: pytest.fail("prepare cannot read image bodies")
    )
    result = prepare(fixture)
    args, _files = fixture
    plan = historical.load(args["root"])
    assert result["recognized_media_refs"] == 3 and result["encoded_groups"] == 1
    assert len(plan["queue"][0]["aliases"]) == 3
    assert len(plan["unresolved"]) == 5
    assert all(
        row["reason"] == "ID_only_or_unrecognized_media_schema" for row in plan["unresolved"]
    )
    assert "DO_NOT_READ_LABEL" not in json.dumps(plan)


def test_verified_encoded_body_is_decoded_once_and_all_references_survive(fixture, monkeypatch):
    prepare(fixture)
    args, _files = fixture
    original, calls = historical.image_identity, []

    def decode(raw):
        calls.append(1)
        return original(raw)

    monkeypatch.setattr(historical, "image_identity", decode)
    for index in range(8):
        historical.worker(args["root"], index)
    report = historical.analyze(args["root"])
    assert len(calls) == 1 and report["verified_encoded_groups"] == 1
    assert report["typed_encoded_assets"] == 2
    assert report["unresolved_provider_entries"] == 5
    assert report["status"] == "finite_historical_media_audit_incomplete"
    output = historical.read(args["root"] / "typed_asset_inventory.json")
    new = next(row for row in output["assets"] if row.get("alias_proofs"))
    assert len(new["references"]) == len(new["alias_proofs"]) == 3
    assert not report["fresh_confirmation_authorized"]


def test_existing_bound_typed_body_reuses_pixels_without_decoding(fixture, monkeypatch):
    args, _files = fixture
    args["typed_media_sha256"] = put(
        args["typed_media"], dict(identities={"old-body": historical.image_identity(png())})
    )
    prepare(fixture)
    monkeypatch.setattr(
        historical,
        "image_identity",
        lambda *args: pytest.fail("already typed body must not re-decode"),
    )
    for index in range(8):
        historical.worker(args["root"], index)
    report = historical.analyze(args["root"])
    assert report["typed_encoded_assets"] == 1
    asset = historical.read(args["root"] / "typed_asset_inventory.json")["assets"][0]
    assert asset["pixel_method"] == "reused_bound_typed_encoded_body"


def test_source_change_blocks_group_without_pixel_emission(fixture):
    prepare(fixture)
    args, files = fixture
    files[0].write_bytes(b"changed")
    plan = historical.load(args["root"])
    index = plan["queue"][0]["index"]
    historical.worker(args["root"], index)
    result = historical.read(args["root"] / "workers" / f"shard-{index}.json")
    assert result["committed"] == [] and len(result["blocked"]) == 1


def test_timeout_keeps_exact_group_pending(fixture, monkeypatch):
    prepare(fixture)
    args, _files = fixture
    plan = historical.load(args["root"])
    group = plan["queue"][0]

    def pause(*args):
        raise TimeoutError("finite deadline")

    monkeypatch.setattr(historical, "project_group", pause)
    historical.worker(args["root"], group["index"])
    result = historical.read(args["root"] / "workers" / f"shard-{group['index']}.json")
    assert result["pending"] == [group] and result["committed"] == []


def test_same_path_conflicting_SHA_is_unresolved_not_arbitrarily_selected(fixture):
    args, _files = fixture
    name, _expected = args["coco_ledgers"][0]
    value = json.loads(Path(name).read_text())
    altered = dict(value["samples"][0], sample_id="conflicting-reference")
    altered["media"] = [dict(value["samples"][0]["media"][0], sha256="0" * 64)]
    value["samples"].append(altered)
    args["coco_ledgers"][0] = (name, put(Path(name), value))
    prepare(fixture)
    plan = historical.load(args["root"])
    assert any(
        r["reason"] == "conflicting_recorded_SHA_for_canonical_path" for r in plan["unresolved"]
    )
    assert len(plan["queue"][0]["aliases"]) == 2


def test_missing_alias_and_outside_root_are_explicit(fixture):
    args, files = fixture
    files[0].unlink()
    args["allowed_roots"] = [files[0].parent / "subdir"]
    args["allowed_roots"][0].mkdir()
    prepare(fixture)
    plan = historical.load(args["root"])
    assert not plan["queue"] and len(plan["unresolved"]) == 8


def test_missing_workers_do_not_complete_finite_media_coverage(fixture):
    prepare(fixture)
    args, _files = fixture
    report = historical.analyze(args["root"])
    assert report["missing_shards"] == list(range(8))
    assert report["status"] == "finite_historical_media_audit_incomplete"


def test_after_pixel_decode_source_change_does_not_emit_asset(fixture, monkeypatch):
    prepare(fixture)
    args, files = fixture
    original = historical.image_identity

    def decode(raw):
        result = original(raw)
        files[0].write_bytes(b"mutated-after-decode")
        return result

    monkeypatch.setattr(historical, "image_identity", decode)
    plan = historical.load(args["root"])
    index = plan["queue"][0]["index"]
    historical.worker(args["root"], index)
    result = historical.read(args["root"] / "workers" / f"shard-{index}.json")
    assert result["committed"] == [] and len(result["blocked"]) == 1


def test_canonical_symlink_alias_deduplicates_but_keeps_original_reference(fixture):
    args, files = fixture
    alias = files[0].parent / "alias.png"
    alias.symlink_to(files[0])
    name, _expected = args["coco_ledgers"][0]
    value = json.loads(Path(name).read_text())
    value["samples"].append(
        dict(
            sample_id="extra-alias", media=[dict(path=str(alias), sha256=historical.sha(files[0]))]
        )
    )
    args["coco_ledgers"][0] = (name, put(Path(name), value))
    prepare(fixture)
    group = historical.load(args["root"])["queue"][0]
    assert len(group["aliases"]) == 3
    references = [r for a in group["aliases"] for r in a["references"]]
    assert len(references) == 4 and any(r["original_media_path"] == str(alias) for r in references)


def test_recovered_five_parent_ledgers_are_bound_by_recipe_matrix(tmp_path):
    matrix = tmp_path / "matrix_manifest.json"
    tasks = ("chartqa", "docvqa", "mmstar", "textvqa", "videomme")
    matrix_sha = put(
        matrix,
        dict(
            sample_ledgers={
                name: dict(path=str(tmp_path / f"{name}.json"), sha256="a" * 64) for name in tasks
            }
        ),
    )
    recipe = tmp_path / "recipe"
    plan_sha = put(recipe / "plan.json", dict(bindings={str(matrix): matrix_sha}))
    put(
        recipe / "GATE_freshness_recipe_recovery.json",
        dict(plan_sha256=plan_sha, status="completed_exact_config_recovery_coverage_overlay"),
    )
    ledgers, bound = historical.recovered_ledgers(recipe)
    assert len(ledgers) == 5 and bound[str(matrix)] == matrix_sha
    matrix.write_text("{}")
    with pytest.raises(ValueError, match="hash changed"):
        historical.recovered_ledgers(recipe)


def test_same_inode_moved_outside_scope_and_symlinked_back_is_rejected(fixture):
    prepare(fixture)
    args, files = fixture
    original = files[0]
    moved = original.parent.parent / "outside.png"
    original.rename(moved)
    original.symlink_to(moved)
    plan = historical.load(args["root"])
    index = plan["queue"][0]["index"]
    historical.worker(args["root"], index)
    output = historical.read(args["root"] / "workers" / f"shard-{index}.json")
    assert not output["committed"] and output["blocked"]
