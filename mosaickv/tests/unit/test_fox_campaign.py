"""Synthetic scientific controls, not measured benchmark rows."""

from dataclasses import asdict

import pytest
import torch

from mosaickv.evaluation.fox_campaign import (
    CampaignRow,
    development_subset,
    independent_masked_attention,
    slurm_seconds,
    source_config,
    unique_storage_bytes,
    validate_locked_policy,
    write_once,
)


@pytest.mark.parametrize("heads,groups", [(1, 1), (2, 1), (2, 4)])
@pytest.mark.parametrize("counts", [(5, 5), (9, 1)])
def test_independent_masked_attention_matches_selected_softmax(heads, groups, counts):
    torch.manual_seed(17)
    key = torch.randn(2, heads, 13, 4, dtype=torch.float64)
    value = torch.randn_like(key)
    query = torch.randn(2, heads * groups, 3, 4, dtype=torch.float64)
    positions = tuple(
        tuple(range(0, 2 * counts[h % 2], 2)) if counts[h % 2] < 7 else tuple(range(9))
        for h in range(heads)
    )
    reference = independent_masked_attention(query, key, value, positions)
    for head, ids in enumerate(positions):
        # Different implementation and a genuinely compact key axis.
        actual = torch.nn.functional.scaled_dot_product_attention(
            query[:, head * groups : (head + 1) * groups],
            key[:, head : head + 1, list(ids)],
            value[:, head : head + 1, list(ids)],
        )
        torch.testing.assert_close(
            actual, reference[:, head * groups : (head + 1) * groups], atol=1e-10, rtol=1e-8
        )


def test_physical_padding_and_backing_storage_not_numel():
    cases = []
    for counts in ((5, 5), (9, 1)):
        key = torch.zeros((1, 2, max(counts), 8), dtype=torch.float16)
        value = torch.zeros_like(key)
        cases.append(unique_storage_bytes([key, value]))
    assert cases == [10 * 8 * 2 * 2, 18 * 8 * 2 * 2]
    base = torch.zeros((1, 2, 20, 8), dtype=torch.float16)
    view = base[:, :, :1]
    assert unique_storage_bytes([view, base, base[:, :, :2]]) == base.untyped_storage().nbytes()
    assert unique_storage_bytes([view]) > view.numel() * view.element_size()


@pytest.mark.parametrize("positions", [((0, 0),), ((3, 1),), ((-1, 2),), ((),)])
def test_reject_bad_membership_without_mutating_source(positions):
    key = torch.randn(1, 1, 6, 4, dtype=torch.float64)
    before = key.clone()
    with pytest.raises(ValueError):
        independent_masked_attention(key[:, :, :1], key, key, positions)
    assert torch.equal(key, before)


def test_mandatory_and_dtype_controls():
    key = torch.randn(1, 1, 6, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="mandatory"):
        independent_masked_attention(key[:, :, :1], key, key, ((1, 3),), mandatory=(5,))
    with pytest.raises(ValueError, match="dtype"):
        independent_masked_attention(key[:, :, :1].float(), key, key, ((1, 3),))


def test_input_only_subsets_are_nested_and_label_independent():
    samples = [
        dict(sample_id=str(i), prompt="x" * (i + 1), references=["secret"], score=-999)
        for i in range(80)
    ]
    one = development_subset(samples, count=64, task="synthetic")
    for s in samples:
        s["references"] = ["changed"]
        s["score"] = 1000
    two = development_subset(list(reversed(samples)), count=64, task="synthetic")
    assert [s["sample_id"] for s in one] == [s["sample_id"] for s in two]
    assert len({s["sample_id"] for s in one}) == 64


def test_atomic_write_once_preserves_original(tmp_path):
    path = tmp_path / "artifact.json"
    write_once(path, dict(a=1))
    with pytest.raises(FileExistsError):
        write_once(path, dict(a=2))
    assert "1" in path.read_text()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "value,seconds", [("2-03:04:05", 183845), ("00:02:30", 150), ("02:30", 150), ("90", 5400)]
)
def test_duration_parser(value, seconds):
    assert slurm_seconds(value) == seconds


def test_failed_sample_schema_requires_error():
    args = dict(
        run_id="r",
        sample_id="s",
        media_group_ids=(),
        workload="coco",
        subtask=None,
        method="mosaickv_exact",
        effective_method=None,
        budget_bytes=40,
        source_sha="a" * 40,
        config_sha="b" * 64,
        manifest_path="p",
        input_sha="c" * 64,
        processor_sha="d" * 64,
        scorer_sha="e" * 64,
        attempt="a",
        repeat=0,
        status="failed",
    )
    with pytest.raises(ValueError, match="error"):
        CampaignRow(**args)
    assert asdict(CampaignRow(**args, error="explicit failure"))["status"] == "failed"


@pytest.mark.parametrize("workload", ["coco", "milebench"])
def test_recovered_policy_matches_actual_campaign(workload):
    validate_locked_policy(source_config(workload))


def test_all_declared_mechanism_variants_have_distinct_configuration():
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "evicastkv_campaign", scripts / "evicastkv_campaign.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from mosaickv.config import config_sha256

        templates = {
            w: dict(
                path=str(
                    __import__(
                        "mosaickv.evaluation.fox_campaign", fromlist=["HISTORICAL"]
                    ).HISTORICAL[w]
                    / "configs/validation/mosaickv_exact-40.json"
                )
            )
            for w in ("coco", "milebench")
        }
        lock = dict(templates=templates)
        hashes = [
            config_sha256(module.resolved_config(lock, "coco", "mosaickv_exact", 40, variant=v))
            for v in ("complete", "routed-prompt", "coverage-off", "value-off", "cross-modal-off")
        ]
        assert len(set(hashes)) == 5
    finally:
        sys.path.remove(str(scripts))


def test_local_token_handoff_preserves_tiny_hf_outputs(tmp_path, monkeypatch):
    """Real HF tiny random weights, CPU only; never download checkpoints."""
    import json
    from dataclasses import replace
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from verify_hf_runtime import _config, _request, tiny_adapters

    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.config import (
        ForecastingConfig,
        GraphConfig,
        PrototypeConfig,
        RepairConfig,
        ResidualConfig,
        SelectionConfig,
    )
    from mosaickv.types import MosaicKVMethod, RepairPolicy

    _, adapter = tiny_adapters()[0]
    template = _config("tiny", MosaicKVMethod.MOSAICKV_EXACT, 1.0, max_new_tokens=4)
    cfg = replace(
        template,
        method=MosaicKVMethod.FULL_KV,
        forecasting=ForecastingConfig(enabled=False),
        graph=GraphConfig(enabled=False),
        selection=SelectionConfig(enabled=False),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE, max_blocks_per_step=0),
    )
    plain = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path / "plain")
    expected = plain.generate(_request("plain"))
    emitted = []
    stream = HuggingFaceMosaicKVModel(
        adapter,
        cfg,
        trace_directory=tmp_path / "stream",
        token_observer=lambda index, token: emitted.append((index, token.detach().cpu().tolist())),
    )
    actual = stream.generate(_request("stream"))
    assert actual.answer == expected.answer
    assert [index for index, _ in emitted] == list(range(4))
    trace = json.loads(next((tmp_path / "stream").rglob("*.json")).read_text())
    assert [ids[0][0] for _, ids in emitted] == trace["generated_token_ids"]
