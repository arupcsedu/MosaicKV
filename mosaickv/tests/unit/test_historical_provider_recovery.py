"""Tiny synthetic providers; canonical RGB equivalence and selective resume."""

import hashlib
import io
import json

import evicastkv_historical_provider_recovery as recovery
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, ImageOps


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return recovery.sha(path)


def body(color):
    stream = io.BytesIO()
    Image.new("RGB", (8, 5), color).save(stream, format="PNG")
    return stream.getvalue()


def canonical(raw):
    with Image.open(io.BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        value = hashlib.sha256(
            f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()
        ).hexdigest()
    return dict(encoded_sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=value)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    parent, root = tmp_path / "parent", tmp_path / "recovery"
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    raw = [body((40 * i, 30, 20)) for i in range(3)]
    items, observations, assets = [], [], []
    for index, binary in enumerate(raw):
        temp = blobs / f"temporary-{index}"
        pq.write_table(pa.table(dict(image=[binary, binary], answer=["PRIVATE", "PRIVATE"])), temp)
        checksum = recovery.sha(temp)
        path = blobs / checksum
        temp.rename(path)
        pool = "docvqa/InfographicVQA/validation" if index else "mmstar/val/val"
        item = dict(
            path=str(path),
            expected_sha256=checksum,
            rows=2,
            pool=pool,
            index=index,
            signature=recovery.signature(path),
        )
        items.append(item)
        observation = dict(
            path=str(path),
            pool=pool,
            rows_expected=2,
            rows_hashed=0 if index else 2,
            complete=not index,
        )
        if index:
            observation["reason"] = "ValueError"
        else:
            observation.update(
                signature=item["signature"],
                whole_file_sha256_before=checksum,
                whole_file_sha256_after=checksum,
            )
            assets.append(
                dict(
                    canonical(binary),
                    references=[
                        dict(pool=pool, source_sha256=checksum, source_row=i) for i in range(2)
                    ],
                )
            )
        observations.append(observation)
    pools = [
        dict(pool="mmstar/val/val", expected_rows=2, metadata_pool_complete=True),
        dict(pool="docvqa/InfographicVQA/validation", expected_rows=4, metadata_pool_complete=True),
    ]
    put(
        parent / "plan.json",
        dict(
            source_sha="a" * 40,
            files=items,
            pools=pools,
            unresolved=[],
            route_contradictions=[],
            libraries={},
            maximum_file_bytes=2**30,
        ),
    )
    bound = {}
    for index in range(8):
        path = parent / "workers" / f"shard-{index}.json"
        bound[str(path)] = put(
            path,
            dict(
                source_sha="a" * 40,
                index=index,
                files=[observations[index]] if index < 3 else [],
                assets=assets if index == 0 else [],
            ),
        )
    inventory = parent / "typed_asset_inventory.json"
    inventory_sha = put(
        inventory, dict(assets=assets, unresolved=observations[1:], worker_bindings=bound)
    )
    put(
        parent / "GATE_historical_provider_superset.json",
        dict(
            source_sha="a" * 40,
            missing_shards=[],
            typed_inventory_sha256=inventory_sha,
            worker_bindings=bound,
        ),
    )
    monkeypatch.setattr(recovery, "FAILED_LFS", {r["expected_sha256"] for r in items[1:]})
    monkeypatch.setattr(
        recovery, "LARGE_BODIES", {hashlib.sha256(r).hexdigest(): [8, 5] for r in raw[1:]}
    )
    monkeypatch.setattr(recovery, "clean_source", lambda *args: "b" * 40)
    monkeypatch.setattr(recovery, "cpu_gate", lambda *args: dict(status="passed"))
    recovery.prepare(root, cpu_gate_root=tmp_path, parent=parent)
    return dict(parent=parent, root=root, raw=raw, items=items)


def test_known_large_identity_has_identical_canonical_RGB_and_no_global_bound_change(prepared):
    for binary in prepared["raw"]:
        assert recovery.image_identity(binary) == canonical(binary)
    assert recovery.parent_audit.image_identity is recovery.ORIGINAL_IDENTITY
    plan = recovery.load(prepared["root"])
    assert len(plan["files"]) == 2
    assert {r["path"] for r in plan["files"]} == {r["path"] for r in prepared["items"][1:]}


def test_exact_geometry_and_hard_resource_cap_are_enforced(prepared, monkeypatch):
    binary = prepared["raw"][1]
    encoded = hashlib.sha256(binary).hexdigest()
    monkeypatch.setattr(recovery, "LARGE_BODIES", {encoded: [9, 5]})
    with pytest.raises(ValueError, match="exact registered geometry"):
        recovery.image_identity(binary)
    monkeypatch.setattr(recovery, "LARGE_BODIES", {encoded: [8, 5]})
    monkeypatch.setattr(recovery, "MAX_PIXELS", 39)
    with pytest.raises(ValueError, match="exact registered geometry"):
        recovery.image_identity(binary)


def test_unknown_body_uses_unchanged_parent_resource_policy(prepared, monkeypatch):
    def blocked(raw):
        raise ValueError("old resource bound still enforced")

    monkeypatch.setattr(recovery, "ORIGINAL_IDENTITY", blocked)
    with pytest.raises(ValueError, match="old resource bound"):
        recovery.image_identity(prepared["raw"][0])


def test_recovery_preserves_all_parent_assets_and_errors_without_rerunning_success(
    prepared, monkeypatch
):
    root, parent = prepared["root"], prepared["parent"]
    parent_sha = recovery.sha(parent / "typed_asset_inventory.json")
    original = pq.ParquetFile
    opened = []

    def selected(path):
        opened.append(str(path))
        assert str(path) != prepared["items"][0]["path"]
        return original(path)

    monkeypatch.setattr(pq, "ParquetFile", selected)
    for index in range(2):
        recovery.worker(root, index)
    gate = recovery.analyze(root)
    assert gate["status"] == "selective_provider_recovery_complete"
    assert gate["successful_parent_files_rerun"] == 0 and len(opened) == 2
    assert recovery.sha(parent / "typed_asset_inventory.json") == parent_sha
    inventory = recovery.read(root / "typed_asset_inventory.json")
    assert len(inventory["historical_parent_errors_preserved"]) == 2
    assert len(inventory["assets"]) == 3
    assert sum(len(r["references"]) for r in inventory["assets"]) == 6
    assert all(r["conservative_media_superset_complete"] for r in gate["pool_coverage"])
    assert not gate["fresh_confirmation_authorized"] and not gate["video_exposure_resolved"]
    assert "PRIVATE" not in (root / "typed_asset_inventory.json").read_text()


def test_missing_worker_and_source_mutation_remain_explicit(prepared):
    root = prepared["root"]
    path = prepared["items"][1]["path"]
    __import__("pathlib").Path(path).write_bytes(b"mutated source")
    recovery.worker(root, 0)
    gate = recovery.analyze(root)
    assert gate["status"] == "selective_provider_recovery_incomplete"
    assert gate["missing_shards"] == [1]
    assert gate["worker_errors"] == 2


def test_mutated_parent_worker_binding_blocks_retry(prepared):
    path = prepared["parent"] / "workers/shard-1.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        recovery.worker(prepared["root"], 0)


def test_duplicate_or_dropped_image_references_cannot_be_promoted(prepared):
    root = prepared["root"]
    for index in range(2):
        recovery.worker(root, index)
    path = root / "workers/shard-0.json"
    value = recovery.read(path)
    value["assets"][0]["references"].append(value["assets"][0]["references"][0])
    put(path, value)
    with pytest.raises(ValueError, match="duplicates an already preserved"):
        recovery.analyze(root)
