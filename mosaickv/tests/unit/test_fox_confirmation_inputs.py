"""Synthetic CPU-only input binding fixtures, not captured model evidence."""

from __future__ import annotations

import copy
import math

import pytest

from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.fox_confirmation_inputs import (
    DATASET_ID,
    DATASET_REVISION,
    MAX_SCALAR_BYTES,
    MODEL_ID,
    MODEL_REVISION,
    PROCESSOR_DESCRIPTOR_SCHEMA,
    ROUTE_SOURCE_HASHES,
    assert_runtime_processor_matches,
    processor_fingerprints,
    resolve_input_dependencies,
)
from mosaickv.evaluation.paper_readiness_v63 import AUTHORIZATION_KEYS


def protocol():
    return {
        "enabled": False,
        "execution_enabled": False,
        "authorization": {key: False for key in AUTHORIZATION_KEYS},
        "recovered_reference_controls": {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "precision": "fp16"},
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "val"},
            "dataset_config_name": None,
        },
    }


def signature():
    return {
        "processor_class": "SyntheticProcessorFixture",
        "image_preprocessor": {"size": 2, "do_normalize": True},
        "chat_template": "{{ messages }}",
        "tokenizer_chat_template": None,
        "special_tokens": {"bos_token": "<s>"},
        "vocabulary": {"<s>": 0, "cat": 1, "dog": 2},
    }


def descriptor():
    identity = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "processor_id": MODEL_ID,
        "processor_revision": MODEL_REVISION,
    }
    actual = signature()
    return {
        "schema": PROCESSOR_DESCRIPTOR_SCHEMA,
        "reviewed": True,
        "local_files_only": True,
        **identity,
        "processor_class": actual["processor_class"],
        **processor_fingerprints(actual),
        "provenance": {
            **identity,
            "source_sha": "a" * 40,
            "capture_manifest_sha256": "b" * 64,
        },
    }


def resolve(**kwargs):
    return resolve_input_dependencies(protocol(), route_sources=dict(ROUTE_SOURCE_HASHES), **kwargs)


def test_declared_default_route_is_resolved_without_processor_or_dataset_access(monkeypatch):
    from pathlib import Path

    def forbidden(*args, **kwargs):
        raise AssertionError("pure resolver must never read data, models, or source files")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    result = resolve()
    assert result["status"] == "resolved_source_route_processor_capture_required"
    assert result["route"]["config_name"] == "default"
    assert result["route"]["lmms_task"] == "coco2014_cap_val"
    assert result["processor_bound"] is False
    assert result["unresolved"] == ["reviewed_actual_processor_capture_required"]
    assert not any(result["certifications"].values())
    assert result["execution_enabled"] is False
    assert not any(result["authorization"].values())


def test_reviewed_signature_is_bound_not_promoted_and_no_input_mutation():
    supplied = descriptor()
    original = copy.deepcopy(supplied)
    bound = resolve(processor_descriptor=supplied)
    assert bound["status"] == "bound_reviewed_processor"
    assert bound["processor_bound"] is True
    assert (
        bound["processor"]["processor_signature_sha256"] == supplied["processor_signature_sha256"]
    )
    assert "signature" not in bound["processor"]
    assert not any(bound["certifications"].values())
    assert_runtime_processor_matches(bound, signature())
    assert not any(bound["certifications"].values())
    supplied["provenance"]["source_sha"] = "c" * 40
    assert bound["processor"]["provenance"] == original["provenance"]
    saved = dict(bound)
    observed = saved.pop("binding_sha256")
    assert digest(saved) == observed


@pytest.mark.parametrize(
    "component", ["image_preprocessor", "chat_template", "special_tokens", "vocabulary"]
)
def test_actual_runtime_change_is_rejected(component):
    bound = resolve(processor_descriptor=descriptor())
    actual = signature()
    actual[component] = {
        "image_preprocessor": {"size": 3},
        "chat_template": "{{ other }}",
        "special_tokens": {"bos_token": "<different>"},
        "vocabulary": {"<s>": 0, "cat": 2, "dog": 1},
    }[component]
    with pytest.raises(ValueError, match="actual runtime processor"):
        assert_runtime_processor_matches(bound, actual)


@pytest.mark.parametrize(
    "field",
    [
        "processor_signature_sha256",
        "preprocessing_sha256",
        "tokenization_sha256",
        "chat_template_sha256",
    ],
)
def test_changed_component_fingerprint_rejected(field):
    supplied = descriptor()
    supplied[field] = "c" * 64
    # The caller verifies the descriptor's file SHA. The pure runtime matcher
    # additionally rejects an actual processor inconsistent with its claims.
    bound = resolve(processor_descriptor=supplied)
    with pytest.raises(ValueError, match="actual runtime processor"):
        assert_runtime_processor_matches(bound, signature())
    supplied["processor_signature"] = signature()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        resolve(processor_descriptor=supplied)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewed", 1),
        ("reviewed", False),
        ("local_files_only", 1),
        ("processor_id", "unknown"),
        ("processor_revision", "main"),
        ("model_revision", "c" * 40),
    ],
)
def test_unreviewed_placeholder_or_mismatched_descriptor_rejected(field, value):
    supplied = descriptor()
    supplied[field] = value
    with pytest.raises(ValueError, match="reviewed pinned"):
        resolve(processor_descriptor=supplied)


def test_capture_provenance_must_match_processor_identity():
    supplied = descriptor()
    supplied["provenance"]["processor_revision"] = "c" * 40
    with pytest.raises(ValueError, match="provenance identity mismatch"):
        resolve(processor_descriptor=supplied)
    supplied = descriptor()
    supplied["provenance"]["capture_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="non-placeholder"):
        resolve(processor_descriptor=supplied)


@pytest.mark.parametrize("route_sources", [{}, {"arbitrary.py": "a" * 64}])
def test_missing_authoritative_route_sources_rejected(route_sources):
    with pytest.raises(ValueError, match=r"source|sources"):
        resolve_input_dependencies(protocol(), route_sources=route_sources)


def test_changed_and_duplicate_source_bindings_rejected():
    sources = dict(ROUTE_SOURCE_HASHES)
    key = next(iter(sources))
    sources[key] = "a" * 64
    with pytest.raises(ValueError, match="changed"):
        resolve_input_dependencies(protocol(), route_sources=sources)
    sources = dict(ROUTE_SOURCE_HASHES)
    sources["/synthetic/source/" + key] = sources[key]
    with pytest.raises(ValueError, match="duplicate"):
        resolve_input_dependencies(protocol(), route_sources=sources)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("dataset", "split", "train"),
        ("dataset", "revision", "main"),
        ("model", "precision", "bf16"),
    ],
)
def test_changed_registered_route_or_model_rejected(section, field, value):
    supplied = protocol()
    supplied["recovered_reference_controls"][section][field] = value
    with pytest.raises(ValueError, match="controls differ"):
        resolve_input_dependencies(supplied, route_sources=dict(ROUTE_SOURCE_HASHES))


def test_nondefault_dataset_config_rejected():
    supplied = protocol()
    supplied["recovered_reference_controls"]["dataset_config_name"] = "test"
    with pytest.raises(ValueError, match="configuration differs"):
        resolve_input_dependencies(supplied, route_sources=dict(ROUTE_SOURCE_HASHES))


@pytest.mark.parametrize("value", [True, 0, 1, None, "false"])
def test_strict_disabled_authorization_required(value):
    supplied = protocol()
    supplied["authorization"][AUTHORIZATION_KEYS[0]] = value
    with pytest.raises(ValueError, match="permissions"):
        resolve_input_dependencies(supplied, route_sources=dict(ROUTE_SOURCE_HASHES))


@pytest.mark.parametrize("value", [math.inf, math.nan, object(), (1, 2)])
def test_nonfinite_or_nonprimitive_metadata_rejected(value):
    supplied = descriptor()
    supplied["extra"] = value
    with pytest.raises(ValueError, match="primitive"):
        resolve(processor_descriptor=supplied)


def test_metadata_scalar_depth_and_projected_size_limits_are_unchanged():
    supplied = descriptor()
    supplied["extra"] = "a" * (MAX_SCALAR_BYTES + 1)
    with pytest.raises(ValueError, match="scalar"):
        resolve(processor_descriptor=supplied)
    nested = None
    for _ in range(66):
        nested = [nested]
    supplied["extra"] = nested
    with pytest.raises(ValueError, match="depth"):
        resolve(processor_descriptor=supplied)
    supplied["extra"] = [str(i) for i in range(70000)]
    with pytest.raises(ValueError, match="byte"):
        resolve(processor_descriptor=supplied)


def test_unknown_or_tampered_binding_is_not_runtime_verification():
    with pytest.raises(ValueError, match="reviewed bound"):
        assert_runtime_processor_matches(resolve(), signature())
    bound = resolve(processor_descriptor=descriptor())
    bound["processor"]["processor_signature_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="binding changed"):
        assert_runtime_processor_matches(bound, signature())


def test_full_vocabulary_is_stream_hashed_without_ingesting_it_into_binding():
    actual = signature()
    actual["vocabulary"] = {f"synthetic-token-{i}": i for i in range(32000)}
    hashes = processor_fingerprints(actual)
    assert hashes["processor_signature_sha256"] == digest(actual)
    supplied = descriptor()
    supplied.update(hashes)
    bound = resolve(processor_descriptor=supplied)
    assert_runtime_processor_matches(bound, actual)
    assert "vocabulary" not in str(bound)
    assert len(str(bound)) < 8192
    assert bound["processor"]["runtime_verified"] is False
