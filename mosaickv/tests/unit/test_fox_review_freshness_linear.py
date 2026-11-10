"""Linear lexical work and cooperative deadlines, not wall-clock benchmarks."""

import json

import evicastkv_review_freshness_stream as scan
import pytest


class NoSuffixCopies(str):
    """The prior text[cursor:] implementation fails immediately on this input."""

    def __getitem__(self, key):
        if isinstance(key, slice) and key.stop is None:
            raise AssertionError("quadratic remaining-buffer suffix copy")
        return super().__getitem__(key)


@pytest.mark.parametrize("field", ["logits", "unregistered_numeric_tensor"])
def test_large_numeric_arrays_use_original_buffer_offsets_with_linear_work(field):
    observations = []
    for count in (4096, 8192):
        raw = NoSuffixCopies('{"' + field + '":[' + ",".join(["0.123456789"] * count) + "]}")
        stats = {}
        projected = scan.identity_projection(raw, statistics=stats)
        assert stats["scalar_matches"] == count
        assert projected[field] is None if field == "logits" else projected[field] == [None] * count
        observations.append(stats["work_units"])
    assert observations[1] <= 2 * observations[0] + 4


def test_scalar_regex_matches_same_original_text_at_nonzero_offsets(monkeypatch):
    original = scan.SCALAR
    observed = []

    class MatchAtOffset:
        def match(self, text, offset):
            observed.append((id(text), offset))
            return original.match(text, offset)

    monkeypatch.setattr(scan, "SCALAR", MatchAtOffset())
    raw = NoSuffixCopies('{"logits":[0,1,2,3,4]}')
    assert scan.identity_projection(raw) == {"logits": None}
    assert len(observed) == 5
    assert all(identity == id(raw) and offset > 0 for identity, offset in observed)


def test_deadline_checked_inside_forbidden_numeric_payload(monkeypatch):
    class Deadline:
        deadline = 1

    calls = []

    def clock():
        calls.append(True)
        return 0 if len(calls) < 3 else 2

    monkeypatch.setattr(scan.time, "monotonic", clock)
    stats = {}
    raw = '{"logits":[' + ",".join(["0"] * 10000) + "]}"
    with pytest.raises(scan.ProjectionPaused, match="deadline"):
        scan.identity_projection(raw, budget=Deadline(), statistics=stats)
    assert 0 < stats["scalar_matches"] < 10000
    assert stats["work_units"] <= 2 * scan.PROJECTION_CHECK_INTERVAL


@pytest.mark.parametrize(
    "raw",
    [
        '{"logits":[0,]}',
        '{"logits":{"a" 1}}',
        '{"logits":[0 1]}',
        '{"logits":[0}',
        '{"logits":falsee}',
        '{"logits":"unterminated}',
    ],
)
def test_forbidden_payload_malformed_json_still_fails_closed(raw):
    with pytest.raises(ValueError):
        scan.identity_projection(raw)


def test_forbidden_values_are_redacted_before_json_construction(monkeypatch):
    loads = scan.json.loads
    seen = []

    def checked(text):
        if text.startswith("{"):
            assert "secret-reference" not in text
            assert "unseen-caption" not in text
            seen.append(text)
        return loads(text)

    monkeypatch.setattr(scan.json, "loads", checked)
    projected = scan.identity_projection(
        '{"references":["secret-reference"],"caption":"unseen-caption","sample_id":"same"}'
    )
    assert projected == dict(references=None, caption=None, sample_id="same")
    assert seen


def test_timeout_inside_json_keeps_file_pending_and_does_not_claim_coverage(tmp_path, monkeypatch):
    roots = (tmp_path / "history",)
    roots[0].mkdir()
    path = roots[0] / "ledger.json"
    path.write_text(json.dumps(dict(sample_id="not-yet-projected")))
    state = scan.initial_state(roots)
    item = dict(kind="file", path=str(path), signature=scan.signature(path))
    state["frontier"] = [item]
    original = scan.identity_projection

    def paused(*args, **kwargs):
        raise scan.ProjectionPaused("controlled inside-record deadline")

    monkeypatch.setattr(scan, "identity_projection", paused)
    scan.advance(state, roots, shard_index=0, shard_count=1, budget=scan.Budget())
    assert state["frontier"][0]["path"] == str(path)
    assert not state["traversal_complete"]
    assert state["processed_files"] == 0
    assert not state["sample_ids"]
    assert state["projection_pauses"]
    assert not state["errors"]  # A pause is not reclassified as malformed input.
    monkeypatch.setattr(scan, "identity_projection", original)
    scan.advance(state, roots, shard_index=0, shard_count=1, budget=scan.Budget())
    assert state["traversal_complete"]
    assert state["sample_ids"] == ["not-yet-projected"]


def test_jsonl_pause_preserves_committed_segment_and_replays_only_unfinished_record(
    tmp_path, monkeypatch
):
    path = tmp_path / "raw.jsonl"
    records = [dict(sample_id="a"), dict(sample_id="b"), dict(sample_id="c")]
    lines = [json.dumps(record).encode() + b"\n" for record in records]
    path.write_bytes(b"".join(lines))
    state = scan.initial_state((tmp_path,))
    item = dict(kind="file", path=str(path), signature=scan.signature(path))
    original = scan.identity_projection

    def pause_second(raw, **kwargs):
        if b'"b"' in raw:
            raise scan.ProjectionPaused("second-record deadline")
        return original(raw, **kwargs)

    monkeypatch.setattr(scan, "identity_projection", pause_second)
    with pytest.raises(scan.ProjectionPaused):
        scan._process_file(item, state, scan.Budget())
    assert state["sample_ids"] == ["a"]
    assert state["projected_records"] == 1
    assert item["offset"] == len(lines[0])
    assert item["segments"][-1]["end"] == len(lines[0])
    monkeypatch.setattr(scan, "identity_projection", original)
    assert scan._process_file(item, state, scan.Budget())
    assert state["sample_ids"] == ["a", "b", "c"]
    assert state["projected_records"] == 3
    assert item["segments"][1]["start"] == len(lines[0])
    assert item["segments"][-1]["end"] == path.stat().st_size
