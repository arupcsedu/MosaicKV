from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_source_caption_study as study
import pytest


def test_fixed_policy_scope_is_common_and_does_not_expand_text_window():
    assert set(study.SETTINGS) == {"source_boundary8", "delimiter_boundary8"}
    assert set(study.SETTINGS.values()) == {("head3_boundary8", 0)}
    with pytest.raises(ValueError, match="registered"):
        study.construct(*([None] * 8), "query3", policy="window32")


def test_unchanged_union_control_never_constructs_text_sources():
    packets = object()
    proof = {"source_memberships": ()}
    adapter = SimpleNamespace()
    args = (adapter, None, None, None, None, "mosaickv_exact", None, 0.5, "query3")
    with patch(
        "mosaickv.evaluation.head_local_precision.bundle", return_value=({"union3": packets}, proof)
    ) as bundle:
        actual = study.construct(*args, policy="union3")
    assert actual == (packets, proof)
    assert bundle.call_args.kwargs["allocation_mode"] == "profile"
