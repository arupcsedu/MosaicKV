"""Synthetic evidence and scope guards; never open real captures in tests."""

import copy
from pathlib import Path

import pytest
import torch
import yaml
from audit_attention_compaction_mass import SCOPE, audit_layer, run, validate_protocol

from mosaickv.prototypes.attention_compaction import AttentionCompactionPolicy


def protocol():
    path = Path(__file__).resolve().parents[2] / "configs/experiments/attention_compaction_v1.yaml"
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("permission", SCOPE)
def test_audit_cannot_enable_downstream_scope(permission):
    config = protocol()
    validate_protocol(config)
    config[permission] = True
    with pytest.raises(ValueError, match="scope"):
        validate_protocol(config)


def test_label_reading_and_arbitrary_effective_methods_are_not_allowed():
    for change in ("labels", "method"):
        config = copy.deepcopy(protocol())
        if change == "labels":
            config["dictionary_audit"]["may_read_benchmark_annotations"] = True
        else:
            config["effective_method"] = "mosaickv_exact"
        with pytest.raises(ValueError, match="unregistered"):
            validate_protocol(config)


def test_dictionary_mass_audit_is_deterministic_and_does_not_read_values():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Uniform identical keys: a representative can match mass exactly.
        # No target outputs, original values, or annotations are present.
        data = dict(
            prompt_probabilities=torch.full((32, 32, 50), 1 / 51),
            generated_mass=torch.full((32, 32), 1 / 51),
        )
        policy = AttentionCompactionPolicy(True, mass_ridge=0)
        a, b = audit_layer(data, 39, policy), audit_layer(data, 39, policy)
        assert a == b and len(a) == 32
        assert all(row["fitted_mass_rmse"] < 1e-6 for row in a)
        assert all(row["selected_positions"][0] == 0 for row in a)
        assert all(row["selected_positions"][-1] == 49 for row in a)
        assert all(row["protected_bias_zero"] for row in a)
        assert all(row["numerical_rank"] == 1 for row in a)
        with pytest.raises(ValueError, match="complete finite"):
            audit_layer(dict(data, generated_mass=torch.full((32, 32), 2.0)), 39, policy)
    finally:
        torch.set_num_threads(old)


def test_existing_output_directory_is_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="NEW immutable"):
        run(tmp_path)
