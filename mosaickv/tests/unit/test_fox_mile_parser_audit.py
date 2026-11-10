"""The diagnostic must observe—not replace—the pinned-compatible parser."""

import sys

import evicastkv_mile_parser_audit as module
import pytest


@pytest.mark.parametrize(
    "choices",
    [["red", "blue"], ["red", "blue", "green"], ["red", "blue", "green", "orange", "pink"]],
)
def test_actual_option_count_and_official_label_are_observed(choices):
    result = module.choice_audit("B", "blue", choices)
    assert result["option_count"] == len(choices)
    assert result["parsed_choice"] == "B"
    assert result["metric_value"] == 1.0
    assert sys.getprofile() is None


def test_upstream_empty_answer_default_is_not_changed_or_excluded():
    result = module.choice_audit("", "green", ["red", "blue", "green"])
    assert result["parse_status"] == "upstream_empty_answer_default_C"
    assert result["empty_answer"]
    assert result["metric_value"] == 1.0


def test_unparseable_output_retains_the_official_zero():
    result = module.choice_audit("zzz", "blue", ["red", "blue"])
    assert result["parse_status"] == "unparseable_choice"
    assert not result["valid_single_choice"]
    assert result["metric_value"] == 0.0


def test_scorer_failure_restores_thread_profiler():
    with pytest.raises(ValueError, match="reference"):
        module.choice_audit("B", "missing", ["red", "blue"])
    assert sys.getprofile() is None


def test_existing_profiler_is_never_replaced():
    def prior(*args):
        pass

    try:
        sys.setprofile(prior)
        with pytest.raises(RuntimeError, match="another profiler"):
            module.choice_audit("B", "blue", ["red", "blue"])
        assert sys.getprofile() is prior
    finally:
        sys.setprofile(None)
