"""Read-only closure must not convert unknown provenance into confirmation."""

import inspect

import evicastkv_review_closure as packet
import evicastkv_source_schema as schema
import pyarrow as pa


def test_schema_discovery_reads_field_names_not_sample_rows():
    fields = pa.schema(
        [
            pa.field("sample_id", pa.int64()),
            pa.field(
                "task_instance",
                pa.struct(
                    [
                        pa.field("images_path", pa.list_(pa.string())),
                        pa.field("episode_id", pa.string()),
                    ]
                ),
            ),
            pa.field("response", pa.string()),
        ]
    )
    paths = schema.schema_paths(fields)
    assert paths == [
        "sample_id",
        "task_instance",
        "task_instance.images_path",
        "task_instance.episode_id",
        "response",
    ]
    assert [p for p in paths if p.rsplit(".", 1)[-1] in schema.IDENTITY_NAMES] == [
        "task_instance.episode_id"
    ]
    source = inspect.getsource(schema.audit)
    assert ".read(" not in source and "to_pylist" not in source
    assert "confirmation_authorized=False" in source


def test_zero_touching_interval_is_not_strict_improvement_or_equivalence():
    assert packet.verdict(0.0, 1.0).startswith("inconclusive")
    assert packet.verdict(-1.0, 0.0).startswith("inconclusive")
    assert packet.verdict(0.0, 0.0).startswith("inconclusive")
    assert packet.verdict(0.1, 1.0).startswith("positive_in_consumed")
    assert packet.verdict(-1.0, -0.1).startswith("negative_in_consumed")


def test_closure_cannot_submit_gpu_or_freeze_an_author_review():
    source = inspect.getsource(packet.build)
    assert "sbatch" not in source and "load_hf_adapter" not in source
    assert "fresh_confirmation=False" in source
    assert "author_reviewed_config=False" in source
    assert "NO_CONFIG_FROZEN" in source
    assert "len(timing_rows) != 432" in source
    assert "check_exact=True" in source
