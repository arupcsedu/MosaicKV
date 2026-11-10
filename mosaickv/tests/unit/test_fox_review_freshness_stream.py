"""Synthetic CPU-only resumable coverage, lexical isolation and shard tests."""

import json
import os
from pathlib import Path

import evicastkv_review_freshness_stream as scan
import pytest


def historical_roots(tmp_path):
    roots = (tmp_path / "mosaickv", tmp_path / "evicastkv")
    for root in roots:
        root.mkdir()
    return roots


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def finish(roots, *, entries=500000, shard_index=0, shard_count=1):
    # Newly created Weka directory mtimes may settle asynchronously after the
    # child close. Freeze synthetic directories explicitly; production sources
    # retain their original mtimes and continue to fail closed on any change.
    for root in roots:
        for directory, _children, _files in os.walk(root):
            os.utime(directory, ns=(1600000000000000000, 1600000000000000000))
    state = scan.initial_state(roots)
    rounds = 0
    while not state.get("traversal_complete"):
        scan.advance(
            state,
            roots,
            shard_index=shard_index,
            shard_count=shard_count,
            budget=scan.Budget(maximum_entries=entries),
        )
        rounds += 1
        assert rounds < 200
    return state, rounds


def test_projection_only_constructs_whitelisted_leaves(monkeypatch):
    loads = scan.json.loads
    seen = []

    def inspect(text):
        if text.startswith("{"):
            assert "unseen-answer" not in text
            assert "new-undocumented-output" not in text
            seen.append(text)
        return loads(text)

    monkeypatch.setattr(scan.json, "loads", inspect)
    raw = json.dumps(
        dict(
            sample_id="same",
            answer="unseen-answer",
            newly_named_prediction="new-undocumented-output",
            nested=dict(dataset_id="COCO", row_id=3),
            media=[dict(path="COCO_val2014_000000000001.jpg", sha256="a" * 64)],
        )
    )
    projected = scan.identity_projection(raw)
    assert projected["answer"] is None
    assert projected["newly_named_prediction"] is None
    assert projected["nested"]["row_id"] == 3
    assert projected["media"][0]["sha256"] == "a" * 64
    assert seen


@pytest.mark.parametrize(
    "value",
    [
        '{"x": 1,}',
        '{"x": [1,]}',
        '{"x": 1} bad',
        '{"x":',
        '{"sample_id":"first", "sample_id":"second"}',
    ],
)
def test_malformed_generic_json_not_silently_ignored(value):
    with pytest.raises(ValueError):
        scan.identity_projection(value)


def test_stream_reads_generic_json_and_raw_jsonl_without_trace_directory_exclusion(tmp_path):
    roots = historical_roots(tmp_path)
    generic = write(
        roots[0] / "run" / "traces" / "odd-name.json", dict(sample_id="nested", answer="never-read")
    )
    raw = roots[1] / "legacy" / "raw.jsonl"
    raw.parent.mkdir()
    raw.write_text(json.dumps(dict(sample_id="raw-id", task="coco_caption")) + "\n")
    state, _ = finish(roots)
    assert state["sample_ids"] == ["nested", "raw-id"]
    assert state["processed_files"] == 2
    assert str(generic) in state["bindings"]
    assert state["bindings"][str(raw)]["segments"]
    assert state["bindings"][str(raw)]["sha256"] is None
    assert not state["errors"]


def test_bounded_walk_reads_files_before_whole_tree_enumeration(tmp_path):
    roots = historical_roots(tmp_path)
    write(roots[0] / "ledger.json", dict(sample_id="immediate"))
    for number in range(20):
        write(roots[0] / f"child-{number}" / "trace" / "anything.json", dict(sample_id=str(number)))
    state = scan.initial_state(roots)
    # First invocation consumes a bounded directory prefix and saves file work;
    # the second begins at that file, not at an all-roots discovery barrier.
    scan.advance(state, roots, shard_index=0, shard_count=1, budget=scan.Budget(maximum_entries=3))
    scan.advance(state, roots, shard_index=0, shard_count=1, budget=scan.Budget(maximum_entries=3))
    assert state["projected_records"] > 0
    assert not state["traversal_complete"]


def test_resume_frontier_eventually_covers_every_record_without_duplicate_reads(tmp_path):
    roots = historical_roots(tmp_path)
    for number in range(25):
        write(roots[number % 2] / f"run-{number}" / "metadata.json", dict(sample_id=str(number)))
    state, rounds = finish(roots, entries=4)
    assert rounds > 1
    assert len(state["sample_ids"]) == 25, (state["errors"], state["sample_ids"])
    assert state["projected_records"] == 25
    assert state["processed_files"] == 25
    assert not state["errors"]


def test_shards_partition_canonical_runs_and_union_complete(tmp_path):
    roots = historical_roots(tmp_path)
    for number in range(30):
        write(roots[number % 2] / f"run-{number}" / "raw.json", dict(sample_id=str(number)))
    states = [finish(roots, shard_index=index, shard_count=4)[0] for index in range(4)]
    files = [set(state["bindings"]) for state in states]
    assert len(set.union(*files)) == 30
    for index, selected in enumerate(files):
        assert all(not selected.intersection(other) for other in files[index + 1 :])
    assert sum(state["processed_files"] for state in states) == 30


def test_inside_alias_dedup_and_outside_broken_links_are_explicit_gaps(tmp_path):
    roots = historical_roots(tmp_path)
    target = write(roots[0] / "run" / "ledger.json", dict(sample_id="single"))
    (roots[1] / "alias").symlink_to(target.parent, target_is_directory=True)
    (roots[0] / "broken").symlink_to(tmp_path / "not-present")
    outside = write(tmp_path / "outside.json", {})
    (roots[0] / "escape.json").symlink_to(outside)
    state, _ = finish(roots)
    assert state["sample_ids"] == ["single"]
    assert state["processed_files"] == 1
    assert len(state["errors"]) == 2
    assert state["aliases"]


def test_scheduler_skip_rule_exact_and_malformed_canonical_raw_blocks(tmp_path):
    roots = historical_roots(tmp_path)
    job = roots[0] / "run" / "attempts" / "attempt-0"
    job.mkdir(parents=True)
    for name in ("stdout.jsonl", "stderr.jsonl", "raw.jsonl", "stdout-extra.jsonl"):
        (job / name).write_text("not JSON\n")
    state, _ = finish(roots)
    assert len(state["scheduler_logs_excluded"]) == 2
    assert len(state["errors"]) == 2
    assert {Path(row["path"]).name for row in state["errors"]} == {
        "raw.jsonl",
        "stdout-extra.jsonl",
    }


def test_oversize_json_and_yaml_are_counted_as_coverage_errors(tmp_path, monkeypatch):
    roots = historical_roots(tmp_path)
    write(roots[0] / "big.json", {"sample_id": "too-big"})
    (roots[0] / "old.yaml").write_text("sample_id: yaml-id\n")
    monkeypatch.setattr(scan, "MAX_JSON", 3)
    state, _ = finish(roots)
    assert len(state["errors"]) == 2
    assert state["sample_ids"] == []


def test_jsonl_resumes_raw_content_segments_without_rescoring(tmp_path):
    roots = historical_roots(tmp_path)
    raw = roots[0] / "raw.jsonl"
    raw.write_text(
        "".join(
            json.dumps(dict(sample_id=str(number), answer="redacted")) + "\n"
            for number in range(30)
        )
    )
    item = dict(kind="file", path=str(raw), signature=scan.signature(raw))
    state = scan.initial_state(roots)
    # Tiny byte bound is an internal synthetic stress only; production Budget
    # requires one full bounded line to avoid a zero-progress invocation.
    budget = scan.Budget()
    budget.maximum_bytes = 100
    assert not scan._process_file(item, state, budget)
    first = item["offset"]
    assert first > 0
    assert scan._process_file(item, state, scan.Budget())
    assert state["projected_records"] == 30
    segments = state["bindings"][str(raw)]["segments"]
    assert segments[0]["end"] == segments[1]["start"] == first
    assert segments[-1]["end"] == raw.stat().st_size


def test_mutation_after_saved_read_is_not_a_fresh_inventory(tmp_path):
    roots = historical_roots(tmp_path)
    path = write(roots[0] / "data.json", dict(sample_id="before"))
    state = scan.initial_state(roots)
    scan.advance(state, roots, shard_index=0, shard_count=1, budget=scan.Budget(maximum_entries=1))
    # Ensure a seen source is bound before mutating it.
    scan.advance(state, roots, shard_index=0, shard_count=1, budget=scan.Budget(maximum_entries=1))
    assert state["sample_ids"] == ["before"]
    write(path, dict(sample_id="after-and-longer"))
    for _ in range(10):
        scan.advance(
            state, roots, shard_index=0, shard_count=1, budget=scan.Budget(maximum_entries=3)
        )
        if state.get("traversal_complete"):
            break
    assert any("changed since" in row["error"] for row in state["errors"])


def test_parquet_reads_metadata_columns_not_answers(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    roots = historical_roots(tmp_path)
    path = roots[0] / "samples.parquet"
    pq.write_table(
        pa.table(
            dict(
                sample_id=["pq-a", "pq-b"],
                answer=["never-a", "never-b"],
                dataset_id=["COCO", "COCO"],
            )
        ),
        path,
    )
    columns = []
    original = pq.ParquetFile.iter_batches

    def inspect(self, *args, **kwargs):
        columns.extend(kwargs["columns"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", inspect)
    state, _ = finish(roots)
    assert state["sample_ids"] == ["pq-a", "pq-b"]
    assert "answer" not in columns
    assert not state["errors"]
    assert state["bindings"][str(path)]["segments"]


def test_dataset_namespaced_ids_not_silently_merged(tmp_path):
    roots = historical_roots(tmp_path)
    write(roots[0] / "a.json", dict(dataset_id="COCO", split="train", sample_id="17"))
    write(roots[1] / "b.json", dict(dataset_id="other", split="train", sample_id="17"))
    state, _ = finish(roots)
    assert state["sample_ids"] == ["17"]
    assert len(state["scoped_identities"]) == 2
    assert {row[0] for row in state["scoped_identities"]} == {"COCO", "other"}


@pytest.mark.parametrize("seconds", [0, -1, 1501])
def test_finite_invocation_bound(seconds):
    with pytest.raises(ValueError, match="1500 seconds"):
        scan.Budget(seconds=seconds)


def test_prior_resolutions_not_reintroduced_as_unknowns(tmp_path):
    roots = historical_roots(tmp_path)
    state = scan.initial_state(roots)
    state["sample_ids"] = ["prior-excluded"]
    scan.advance(state, roots, shard_index=0, shard_count=1, budget=scan.Budget())
    assert state["sample_ids"] == ["prior-excluded"]


def test_write_once_shards_resume_hashes_and_aggregate_never_authorize_freshness(
    tmp_path, monkeypatch
):
    roots = historical_roots(tmp_path)
    for number in range(10):
        write(roots[number % 2] / f"run-{number}" / "ledger.json", dict(sample_id=str(number)))
    monkeypatch.setattr(scan, "clean_source", lambda path: "a" * 40)
    outputs = [tmp_path / f"shard-{index}" for index in range(2)]
    for index, output in enumerate(outputs):
        gate = scan.run(
            output,
            roots=roots,
            shard_count=2,
            shard_index=index,
            maximum_entries=3,
            reviewed_previous=False,
        )
        assert not gate["traversal_complete"]
        for _ in range(40):
            gate = scan.run(
                output,
                roots=roots,
                shard_count=2,
                shard_index=index,
                maximum_entries=3,
                reviewed_previous=False,
                resume=True,
            )
            if gate["traversal_complete"]:
                break
        assert gate["traversal_complete"]
        assert gate["complete_declared_identity_field_scope"]
        assert not gate["complete_historical_id_inventory"]
        assert not gate["fresh_confirmation_authorized"]
        with pytest.raises(ValueError, match="terminal"):
            scan.run(
                output,
                roots=roots,
                shard_count=2,
                shard_index=index,
                reviewed_previous=False,
                resume=True,
            )
    result = scan.aggregate(tmp_path / "aggregate", outputs)
    assert result["processed_files"] == 10
    assert result["complete_declared_identity_field_scope"]
    assert not result["complete_historical_id_inventory"]
    assert not result["fresh_confirmation_authorized"]
    with pytest.raises(ValueError, match="immutable"):
        scan.aggregate(tmp_path / "aggregate", outputs)


def test_resume_rejects_source_configuration_or_checkpoint_changes(tmp_path, monkeypatch):
    roots = historical_roots(tmp_path)
    write(roots[0] / "run" / "ledger.json", dict(sample_id="a"))
    monkeypatch.setattr(scan, "clean_source", lambda path: "a" * 40)
    output = tmp_path / "shard"
    gate = scan.run(output, roots=roots, shard_count=2, maximum_entries=1, reviewed_previous=False)
    assert not gate["traversal_complete"]
    with pytest.raises(ValueError, match="registration changed"):
        scan.run(output, roots=roots, shard_count=3, reviewed_previous=False, resume=True)
    checkpoint = Path(gate["checkpoint"])
    altered = json.loads(checkpoint.read_text())
    altered["state"]["sample_ids"] = ["tampered"]
    checkpoint.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="content binding changed"):
        scan.run(output, roots=roots, shard_count=2, reviewed_previous=False, resume=True)


def test_scanner_output_cannot_be_its_own_historical_input(tmp_path, monkeypatch):
    roots = historical_roots(tmp_path)
    monkeypatch.setattr(scan, "clean_source", lambda path: "a" * 40)
    with pytest.raises(ValueError, match="outside scanned"):
        scan.run(roots[0] / "bad-output", roots=roots, reviewed_previous=False)
