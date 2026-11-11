"""Synthetic pinned repositories; no real cache images, labels or network."""

import copy
import hashlib
import io
import json
from pathlib import Path

import evicastkv_historical_provider_superset as audit
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return audit.sha(path)


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), color).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    providers = copy.deepcopy(audit.PROVIDERS)
    hub = tmp_path / "hub"
    ledgers = []
    snapshots = {}
    files = []
    for number, (task, spec) in enumerate(providers.items()):
        snapshot = (
            hub
            / f"datasets--{spec['repository'].replace('/', '--')}"
            / "snapshots"
            / spec["revision"]
        )
        snapshot.mkdir(parents=True)
        snapshots[task] = snapshot
        configs = []
        infos = []
        ledger_ids = []
        for config in spec["configs"]:
            pool_configs = [(c, s) for c, s in spec["pools"] if c == config]
            data_files = []
            splits = []
            for _config, split in pool_configs:
                relative = f"data/{config}-{split}-000.parquet"
                data_files.append(dict(split=split, path=f"data/{config}-{split}-*.parquet"))
                splits.append(dict(name=split, num_examples=2))
                rows = []
                for row in range(2):
                    image = png((number * 40, row * 20, 50))
                    value = dict(
                        image=image if task == "mmstar" else dict(bytes=image, path=None),
                        question="PRIVATE_QUESTION",
                        answer="PRIVATE_ANSWER",
                    )
                    if spec["id_column"]:
                        value[spec["id_column"]] = row + 100
                    rows.append(value)
                table = pa.Table.from_pylist(rows)
                temporary = tmp_path / f"{task}-{config}.parquet"
                pq.write_table(table, temporary)
                raw = temporary.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                blob = snapshot.parent.parent / "blobs" / sha
                blob.parent.mkdir(exist_ok=True)
                temporary.rename(blob)
                alias = snapshot / relative
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.symlink_to(blob)
                files.append(blob)
                ledger_ids.extend(f"{task}:{row + 100}" for row in range(2))
            configs.append(dict(config_name=config, data_files=data_files))
            infos.append(dict(config_name=config, splits=splits))
        card = dict(configs=configs, dataset_info=infos)
        (snapshot / "README.md").write_text(
            "---\n" + yaml.safe_dump(card) + "---\n# Metadata only\n"
        )
        raw = (snapshot / "README.md").read_bytes()
        spec["card_git_oid"] = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        path = tmp_path / f"{task}.json"
        ledgers.append(
            (
                str(path),
                put(
                    path,
                    dict(
                        task=task,
                        dataset_id=spec["repository"],
                        dataset_revision=spec["revision"],
                        split="test" if task == "mmstar" else spec["pools"][0][1],
                        sample_ids=sorted(set(ledger_ids)),
                    ),
                ),
            )
        )
    video = tmp_path / "videomme.json"
    ledgers.append((str(video), put(video, dict(task="videomme", sample_ids=["video-unknown"]))))
    monkeypatch.setattr(audit, "HUB", hub)
    monkeypatch.setattr(audit, "PROVIDERS", providers)
    monkeypatch.setattr(audit, "recovered_ledgers", lambda root: (ledgers, {}))
    monkeypatch.setattr(audit, "clean_source", lambda *args: "f" * 40)
    monkeypatch.setattr(audit, "cpu_gate", lambda *args: dict(status="passed"))
    return dict(root=tmp_path / "audit", hub=hub, files=files, snapshots=snapshots)


def prepare(fixture):
    return audit.prepare(fixture["root"])


def test_prepare_is_metadata_only_and_preserves_legacy_split_contradiction(fixture, monkeypatch):
    monkeypatch.setattr(
        audit, "image_identity", lambda *args: pytest.fail("prepare must not decode images")
    )
    prepare(fixture)
    plan = audit.load(fixture["root"])
    assert len(plan["pools"]) == 5 and all(p["metadata_pool_complete"] for p in plan["pools"])
    assert plan["route_contradictions"][0]["task"] == "mmstar"
    assert not plan["route_contradictions"][0]["legacy_split_equivalence_asserted"]
    assert all(not p["legacy_sample_to_row_parity_asserted"] for p in plan["pools"])
    assert "PRIVATE_ANSWER" not in json.dumps(plan)


def test_all_pinned_image_pools_can_pass_coverage_but_video_remains_unresolved(fixture):
    prepare(fixture)
    for index in range(8):
        audit.worker(fixture["root"], index)
    result = audit.analyze(fixture["root"])
    assert all(p["conservative_media_superset_complete"] for p in result["pool_coverage"])
    assert result["unresolved_providers"] == 1
    assert not result["legacy_sample_route_parity_certified"]
    assert not result["fresh_confirmation_authorized"]
    output = audit.read(fixture["root"] / "typed_asset_inventory.json")
    assert len(output["assets"]) > 0
    assert all(
        "PRIVATE_ANSWER" not in p.read_text() for p in (fixture["root"] / "workers").iterdir()
    )


def test_missing_Infographic_pool_cannot_be_silently_resolved(fixture):
    snapshot = fixture["snapshots"]["docvqa"]
    for p in (snapshot / "data").glob("InfographicVQA-*.parquet"):
        p.unlink()
    prepare(fixture)
    plan = audit.load(fixture["root"])
    missing = next(p for p in plan["pools"] if "InfographicVQA" in p["pool"])
    assert not missing["metadata_pool_complete"] and missing["expected_rows"] == 2
    assert any("InfographicVQA" in gap.get("pool", "") for gap in plan["unresolved"])


def test_worker_projects_only_image_column_not_questions_or_answers(fixture, monkeypatch):
    prepare(fixture)
    original = pq.ParquetFile
    seen = []

    class Guard:
        def __init__(self, path):
            self.real = original(path)

        def __getattr__(self, key):
            return getattr(self.real, key)

        def iter_batches(self, **kwargs):
            seen.append(kwargs["columns"])
            assert kwargs["columns"] == ["image"]
            yield from self.real.iter_batches(**kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Guard)
    for index in range(8):
        audit.worker(fixture["root"], index)
    assert seen and all(c == ["image"] for c in seen)


def test_corrupt_pinned_source_fails_before_image_decode(fixture, monkeypatch):
    prepare(fixture)
    plan = audit.load(fixture["root"])
    item = plan["files"][0]
    Path(item["path"]).write_bytes(b"corrupt")
    monkeypatch.setattr(
        audit, "image_identity", lambda *args: pytest.fail("corrupt source decoded")
    )
    audit.worker(fixture["root"], item["index"])
    output = audit.read(fixture["root"] / "workers" / f"shard-{item['index']}.json")
    assert output["files"][0]["rows_hashed"] == 0
    assert output["assets"] == []


def test_after_decode_source_change_drops_unverified_file_assets(fixture, monkeypatch):
    prepare(fixture)
    plan = audit.load(fixture["root"])
    item = plan["files"][0]
    original = audit.image_identity

    def decode(raw):
        result = original(raw)
        Path(item["path"]).write_bytes(b"changed-after-decode")
        return result

    monkeypatch.setattr(audit, "image_identity", decode)
    audit.worker(fixture["root"], item["index"])
    output = audit.read(fixture["root"] / "workers" / f"shard-{item['index']}.json")
    assert not output["files"][0]["complete"] and output["assets"] == []


def test_missing_workers_and_configurations_remain_blocking(fixture):
    prepare(fixture)
    result = audit.analyze(fixture["root"])
    assert result["missing_shards"] == list(range(8))
    assert not any(p["conservative_media_superset_complete"] for p in result["pool_coverage"])


def test_card_event_parser_does_not_mutate_shared_safe_key_policy(tmp_path):
    prior = audit.yaml_events.safe_key
    path = tmp_path / "README.md"
    path.write_text(
        "---\nconfigs: []\ndataset_info: {splits: [{name: test, num_examples: 2}]}\n---\n"
    )
    value, _ = audit.card(path)
    assert value["dataset_info"]["splits"][0]["num_examples"] == 2
    assert audit.yaml_events.safe_key is prior


def test_unregistered_repository_configuration_is_not_guessed_away(fixture):
    path = fixture["snapshots"]["chartqa"] / "README.md"
    value = path.read_text().replace(
        "configs:\n", "configs:\n- config_name: unexpected\n  data_files: []\n"
    )
    path.write_text(value)
    raw = path.read_bytes()
    audit.PROVIDERS["chartqa"]["card_git_oid"] = hashlib.sha1(
        b"blob " + str(len(raw)).encode() + b"\0" + raw
    ).hexdigest()
    with pytest.raises(ValueError, match="unregistered possible configurations"):
        prepare(fixture)


def test_pinned_card_mutation_cannot_redefine_possible_repository_pools(fixture):
    path = fixture["snapshots"]["mmstar"] / "README.md"
    path.write_text(path.read_text() + "Changed pinned metadata\n")
    with pytest.raises(ValueError, match="Git object differs"):
        prepare(fixture)


def test_partial_file_rows_remain_valid_only_after_full_source_reverification(fixture, monkeypatch):
    prepare(fixture)
    calls = []

    def clock():
        calls.append(1)
        return 0 if len(calls) < 3 else 211

    monkeypatch.setattr(audit.time, "monotonic", clock)
    monkeypatch.setattr(audit, "file_sha", lambda path, deadline: audit.sha(path))
    audit.worker(fixture["root"], 0)
    result = audit.read(fixture["root"] / "workers/shard-0.json")
    observation = result["files"][0]
    assert not observation["complete"] and observation["rows_hashed"] == 1
    assert observation["whole_file_sha256_before"] == observation["whole_file_sha256_after"]
    assert sum(len(row["references"]) for row in result["assets"]) == 1
