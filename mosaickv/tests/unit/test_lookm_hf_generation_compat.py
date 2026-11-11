"""Explicit generation/cache transport metadata, no checkpoint or GPU."""

from types import SimpleNamespace

import audit_lookm_hf_generation_compat as study
import pytest


class Original:
    def forward(self):
        return "original arithmetic"

    def prepare_inputs_for_generation(self):
        return "original preparation"

    def generate(self):
        return super().generate()


class Mixin:
    @classmethod
    def _supports_default_dynamic_cache(cls):
        return True

    def generate(self):
        return "new public generation driver"


def test_dynamic_wrapper_retains_forward_and_original_super_dispatch():
    wrapper = study.compatibility_class(Original, Mixin)
    assert wrapper.forward is Original.forward
    assert wrapper.prepare_inputs_for_generation is Original.prepare_inputs_for_generation
    assert wrapper().generate() == "new public generation driver"
    assert wrapper._supports_default_dynamic_cache() is False
    assert Mixin._supports_default_dynamic_cache() is True
    assert wrapper.compatibility_label == "patched_official_hf_compat"


def test_unreviewed_cache_interface_is_not_guessed():
    with pytest.raises(ValueError, match="Unreviewed"):
        study.compatibility_class(Original, object)


def records(prompt, physical):
    return [
        {
            "position_first": 0 if step == 0 else prompt + step - 1,
            "position_last": prompt - 1 if step == 0 else prompt + step - 1,
            "position_count": prompt if step == 0 else 1,
            "physical_before": None if step == 0 else [physical + step - 1],
            "physical_after": [physical + step],
        }
        for step in range(study.NEW_TOKENS)
    ]


def test_compressed_physical_lengths_do_not_renumber_logical_positions():
    study.verify_positions(records(584, 116), prompt=584, initial_physical=116)


def test_physical_position_renumbering_is_rejected():
    values = records(584, 116)
    values[1]["position_last"] = 116
    with pytest.raises(ValueError, match="renumbered"):
        study.verify_positions(values, prompt=584, initial_physical=116)


def test_missing_native_steps_cannot_pass():
    with pytest.raises(ValueError, match="Incomplete"):
        study.verify_positions(records(16, 16)[:-1], prompt=16, initial_physical=16)


def test_unexpected_storage_growth_cannot_pass():
    values = records(16, 16)
    values[2]["physical_after"] = [50]
    with pytest.raises(ValueError, match="physical-cache growth"):
        study.verify_positions(values, prompt=16, initial_physical=16)


def test_disconnected_past_cache_cannot_pass():
    values = records(584, 116)
    values[2]["physical_before"] = [584]
    with pytest.raises(ValueError, match="preceding transaction"):
        study.verify_positions(values, prompt=584, initial_physical=116)


def test_cache_class_cannot_be_silently_substituted():
    tensor = SimpleNamespace(shape=(1, 2, 116, 128))
    assert study.cache_lengths(((tensor, tensor),)) == [116]
    with pytest.raises(ValueError, match="legacy tuple"):
        study.cache_lengths([[tensor, tensor]])
    short = SimpleNamespace(shape=(1, 116))
    with pytest.raises(ValueError, match="Unexpected"):
        study.cache_lengths(((short, short),))


def test_no_sampled_or_early_eos_smoke():
    policy = study.generation_kwargs()
    assert policy["do_sample"] is False
    assert policy["num_beams"] == 1
    assert policy["eos_token_id"] is None
    assert policy["max_new_tokens"] == 4
