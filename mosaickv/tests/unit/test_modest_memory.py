"""CPU-only planning tests; synthetic IDs are not experimental evidence."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

P = Path(__file__).resolve().parents[2] / "scripts/prepare_modest_memory.py"
SPEC = importlib.util.spec_from_file_location("modest_memory", P)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_identity_extraction_does_not_use_answers():
    coco, mile = M.identities(
        dict(
            samples=[
                dict(
                    sample_id="coco_caption:COCO_val2014_000000000123.jpg",
                    answer="COCO_val2014_000000000999.jpg",
                )
            ],
            excluded_ids=["ActionPrediction-0001"],
            jobs=[dict(sample_ids=["COCO_val2014_000000000124.jpg"])],
        )
    )
    assert coco == {
        "coco_caption:COCO_val2014_000000000123.jpg",
        "coco_caption:COCO_val2014_000000000124.jpg",
    }
    assert mile == {"ActionPrediction-0001"}


def test_cohorts_are_deterministic_and_disjoint():
    catalog = [dict(sample_id=str(i), row=i) for i in range(20)]
    a = M.split_id_catalog(catalog, {"0", "1"}, seed=7, calibration=4, confirmation=8)
    b = M.split_id_catalog(
        list(reversed(catalog)), {"0", "1"}, seed=7, calibration=4, confirmation=8
    )
    assert a == b
    cal = {r["sample_id"] for r in a["calibration"]}
    confirm = {r["sample_id"] for r in a["confirmation_id_only_reservation"]}
    assert not cal & confirm and not (cal | confirm) & {"0", "1"}
    with pytest.raises(ValueError, match="insufficient"):
        M.split_id_catalog(catalog, set(), seed=7, calibration=15, confirmation=8)
    with pytest.raises(ValueError, match="duplicate"):
        M.split_id_catalog(catalog + catalog[:1], set(), seed=7, calibration=4, confirmation=8)


def test_metadata_scan_prunes_outputs_and_fails_closed(tmp_path):
    root = tmp_path / "runs"
    (root / "old" / "outputs").mkdir(parents=True)
    (root / "old" / "plan.json").write_text(
        json.dumps(dict(sample_ids=["COCO_val2014_000000000123.jpg"]))
    )
    (root / "old" / "outputs" / "plan.json").write_text("malformed scheduler output")
    result = M.prior_identity_inventory(root)
    assert result["coco_excluded_ids"] == ["coco_caption:COCO_val2014_000000000123.jpg"]
    assert result["scanned_metadata_files"] == 1
    (root / "old" / "plan.json").write_text("malformed canonical plan")
    with pytest.raises(json.JSONDecodeError):
        M.prior_identity_inventory(root)


def test_projected_arrow_catalog_uses_identity_column_only(tmp_path):
    import pyarrow as pa

    root = tmp_path / M.COCO_REVISION
    root.mkdir()
    for i in range(14):
        table = pa.table(
            dict(
                question_id=[f"COCO_val2014_{i:012d}.jpg"],
                image=[b"not a decodable image"],
                answer=[None],
            )
        )
        with (
            pa.OSFile(str(root / f"coco-caption-val-{i:05d}-of-00014.arrow"), "wb") as f,
            pa.ipc.new_stream(f, table.schema) as writer,
        ):
            writer.write_table(table)
    records = M.coco_catalog(root)
    assert len(records) == 14
    assert all(set(r) == {"sample_id", "arrow_file", "row"} for r in records)


def test_nearest_competitor_is_not_cherry_picked_and_negative_stops():
    sys.path.insert(0, str(P.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "modest_analysis", P.parent / "analyze_modest_memory.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(P.parent))
    scores = {
        ("full_kv", None): 100,
        ("prompt_attention_topk", 28): 95,
        ("prompt_attention_topk", 32): 100,
        ("prompt_attention_topk", 40): 103,
        ("vl_cache_reimpl", 28): 94,
        ("vl_cache_reimpl", 40): 99,
        ("mosaickv_exact", 28): 101,
        ("mosaickv_exact", 35): 105,
    }
    choice = module.select_pair(scores, target=100)
    assert choice["baseline"] == ["prompt_attention_topk", 32]
    assert choice["candidate"] == ["mosaickv_exact", 28]
    scores["mosaickv_exact", 28] = 99
    assert module.select_pair(scores, target=100)["status"] == "negative"
    assert module.select_pair(scores, target=110)["status"] == "negative"
    assert module.power_size(-1, 1, 128) is None
    assert module.power_size(0, 0, 128) == 512
    assert module.power_size(1, 2, 128) > 512


def test_analysis_metric_imports_with_older_runtime_without_point_alias(monkeypatch):
    old = ModuleType("mosaickv.evaluation.coco_shared_bytes")
    old.TokenizedCider = object
    monkeypatch.setitem(sys.modules, old.__name__, old)
    path = P.parents[1] / "src/mosaickv/evaluation/cider_resampling.py"
    spec = importlib.util.spec_from_file_location("old_runtime_metric_smoke", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "ResampledCider")
