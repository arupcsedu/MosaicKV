"""Pure, bounded COCO route/processor bindings; never experiment admission.

Only caller-supplied source bindings and reviewed processor metadata are used.
No model, dataset, cache, raw result or historical proof is opened here. A
source-declared route is not a dataset/media/freshness certificate. A reviewed
processor signature still requires a match against the actual runtime object.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import PurePosixPath
from typing import Any

from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.paper_readiness_v63 import disabled_authorization

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
MODEL_REVISION = "2424fdd47412fccc66d91719126b420e9fbd7065"
DATASET_ID = "lmms-lab/COCO-Caption"
DATASET_REVISION = "c3213b54e173ae86517d481de17f31439d6f8c72"
COCO_PROMPT = "Provide a one-sentence caption for the provided image."
PROCESSOR_DESCRIPTOR_SCHEMA = "evicastkv_reviewed_processor_fingerprints_v1"
ROUTE_SOURCE_HASHES = {
    "lmms_eval/tasks/coco_cap/coco2014_cap_val.yaml": (
        "a2605acadb669ac4afa066739e0fe36bcff415d8c0dd496f53d8a7617c33f1e1"
    ),
    "mosaickv/src/mosaickv/evaluation/lmms_adapter.py": (
        "93dc1519e6d16192f6285856022439ee8d89773ef8fab314a5065ca211d92643"
    ),
}
MAX_METADATA_BYTES = 8 * 1024**2
MAX_METADATA_NODES = 100000
MAX_METADATA_DEPTH = 64
MAX_SCALAR_BYTES = 1024**2
_SIGNATURE_FIELDS = {
    "processor_class",
    "image_preprocessor",
    "chat_template",
    "tokenizer_chat_template",
    "special_tokens",
    "vocabulary",
}
_IDENTITY = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "processor_id": MODEL_ID,
    "processor_revision": MODEL_REVISION,
}


def _primitives(*values: Any, model_owned_runtime: bool = False) -> None:
    """Retain the existing scalar/node/depth/projected-metadata ceilings."""
    nodes = projected = 0
    pending = [(value, 0) for value in values]
    while pending:
        value, depth = pending.pop()
        nodes += 1
        # Ingested descriptors retain the historical conservative capsule
        # allowance. An already model-owned runtime vocabulary is not ingested
        # or copied: its canonical JSON stream is bounded separately below.
        projected += 0 if model_owned_runtime else 128
        if (
            nodes > MAX_METADATA_NODES
            or depth > MAX_METADATA_DEPTH
            or projected > MAX_METADATA_BYTES
        ):
            raise ValueError("unchanged metadata node/depth/byte limit exceeded")
        if type(value) is dict:
            if nodes + 2 * len(value) > MAX_METADATA_NODES:
                raise ValueError("unchanged metadata node limit exceeded")
            if any(type(key) is not str for key in value):
                raise ValueError("JSON object keys must be strings")
            pending.extend((key, depth + 1) for key in value)
            pending.extend((child, depth + 1) for child in value.values())
        elif type(value) is list:
            if nodes + len(value) > MAX_METADATA_NODES:
                raise ValueError("unchanged metadata node limit exceeded")
            pending.extend((child, depth + 1) for child in value)
        elif type(value) is str:
            size = len(value.encode("utf-8"))
            if size > MAX_SCALAR_BYTES:
                raise ValueError("unchanged scalar byte limit exceeded")
            projected += size
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("finite primitive metadata required")
        elif type(value) not in (bool, int, type(None)):
            raise ValueError("JSON primitive metadata required")
        if projected > MAX_METADATA_BYTES:
            raise ValueError("unchanged projected metadata byte limit exceeded")


def _hash(value: Any, *, length: int = 64) -> str:
    if (
        type(value) is not str
        or re.fullmatch(r"[a-f0-9]{" + str(length) + r"}", value) is None
        or value == "0" * length
    ):
        raise ValueError("non-placeholder lowercase hash binding required")
    return value


def _route_sources(route_sources: dict[str, str]) -> dict[str, str]:
    if type(route_sources) is not dict or not route_sources:
        raise ValueError("explicit pinned route-source bindings required")
    observed = {}
    for path, expected in route_sources.items():
        parts = path.split("/")
        if (
            not path
            or "\x00" in path
            or ".." in parts
            or "." in parts
            or "\\" in path
            or any(part.lower() in {".codex", ".ssh", "auth.json", ".env"} for part in parts)
        ):
            raise ValueError("safe explicit route-source paths required")
        _hash(expected)
        for suffix, pinned in ROUTE_SOURCE_HASHES.items():
            if PurePosixPath(path).parts[-len(PurePosixPath(suffix).parts) :] == (
                PurePosixPath(suffix).parts
            ):
                if suffix in observed or expected != pinned:
                    raise ValueError("duplicate or changed pinned route-source binding")
                observed[suffix] = expected
    if set(observed) != set(ROUTE_SOURCE_HASHES):
        raise ValueError("both pinned evaluator and omitted-name normalization sources required")
    return dict(route_sources)


def processor_fingerprints(signature: dict[str, Any]) -> dict[str, str]:
    """Stream original runtime fingerprints without persisting a vocabulary.

    Caller-owned model state is not a metadata capsule. The original complete
    signature and each component retain the same canonical JSON semantics;
    output is limited to 8 MiB and no full serialized copy is constructed.
    Node/depth/scalar/finite-primitive limits still apply.
    """
    _primitives(signature, model_owned_runtime=True)
    if type(signature) is not dict or set(signature) != _SIGNATURE_FIELDS:
        raise ValueError("complete original processor_signature object required")
    name = signature["processor_class"]
    if (
        type(name) is not str
        or not name.strip()
        or name.strip().lower() in {"tbd", "unknown", "pending", "placeholder", "not_used"}
    ):
        raise ValueError("actual processor class required")
    for key in ("image_preprocessor", "special_tokens", "vocabulary"):
        if type(signature[key]) is not dict or not signature[key]:
            raise ValueError("nonempty actual processor components required")
    for key in ("chat_template", "tokenizer_chat_template"):
        value = signature[key]
        if value is not None and (type(value) is not str or not value.strip()):
            raise ValueError("actual chat template or explicit null required")
    if any(type(value) is not int or value < 0 for value in signature["vocabulary"].values()):
        raise ValueError("actual nonnegative vocabulary token IDs required")

    def stream_digest(value):
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
        state = hashlib.sha256()
        size = 0
        for chunk in encoder.iterencode(value):
            raw = chunk.encode("utf-8")
            size += len(raw)
            if size > MAX_METADATA_BYTES:
                raise ValueError("unchanged canonical runtime metadata byte limit exceeded")
            state.update(raw)
        return state.hexdigest()

    return {
        "processor_signature_sha256": stream_digest(signature),
        "preprocessing_sha256": stream_digest(signature["image_preprocessor"]),
        "tokenization_sha256": stream_digest(
            {key: signature[key] for key in ("vocabulary", "special_tokens")}
        ),
        "chat_template_sha256": stream_digest(
            {
                "processor": signature["chat_template"],
                "tokenizer": signature["tokenizer_chat_template"],
            }
        ),
    }


def _processor_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    if (
        type(descriptor) is not dict
        or descriptor.get("schema") != PROCESSOR_DESCRIPTOR_SCHEMA
        or descriptor.get("reviewed") is not True
        or descriptor.get("local_files_only") is not True
        or any(descriptor.get(key) != value for key, value in _IDENTITY.items())
    ):
        raise ValueError("reviewed pinned model/processor descriptor required")
    provenance = descriptor.get("provenance")
    if type(provenance) is not dict:
        raise ValueError("processor capture provenance required")
    _hash(provenance.get("source_sha"), length=40)
    _hash(provenance.get("capture_manifest_sha256"))
    if any(provenance.get(key) != value for key, value in _IDENTITY.items()):
        raise ValueError("processor capture provenance identity mismatch")
    name = descriptor.get("processor_class")
    if (
        type(name) is not str
        or not name.strip()
        or name.strip().lower() in {"tbd", "unknown", "pending", "placeholder", "not_used"}
    ):
        raise ValueError("actual captured processor class required")
    hashes = {
        key: _hash(descriptor.get(key))
        for key in (
            "processor_signature_sha256",
            "preprocessing_sha256",
            "tokenization_sha256",
            "chat_template_sha256",
        )
    }
    # Optional tiny fixtures/legacy records can prove their own digest
    # consistency. Real reviewed captures use only fingerprints here.
    if "processor_signature" in descriptor:
        actual = processor_fingerprints(descriptor["processor_signature"])
        if actual != hashes or descriptor["processor_signature"]["processor_class"] != name:
            raise ValueError("processor component fingerprint mismatch")
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "local_files_only": True,
        "processor_class": name,
        **hashes,
        "provenance": copy.deepcopy(provenance),
        "reviewed": True,
        "runtime_match_required": True,
        "runtime_verified": False,
    }


def resolve_input_dependencies(
    protocol: dict[str, Any],
    *,
    route_sources: dict[str, str],
    processor_descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve source declarations and bind supplied metadata without any I/O."""
    _primitives(protocol, route_sources, processor_descriptor)
    if type(protocol) is not dict:
        raise ValueError("disabled review protocol object required")
    disabled_authorization(protocol.get("authorization"))
    if protocol.get("enabled") is not False or protocol.get("execution_enabled") is not False:
        raise ValueError("input binding must retain disabled execution")
    controls = protocol.get("recovered_reference_controls")
    if type(controls) is not dict:
        raise ValueError("recovered reference controls required")
    model = {"id": MODEL_ID, "revision": MODEL_REVISION, "precision": "fp16"}
    dataset = {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "val"}
    if controls.get("model") != model or controls.get("dataset") != dataset:
        raise ValueError("pinned COCO model/dataset controls differ")
    if controls.get("dataset_config_name") not in (None, "default"):
        raise ValueError("source-declared COCO dataset configuration differs")
    bindings = _route_sources(route_sources)
    bound = processor_descriptor is not None
    processor = (
        _processor_descriptor(processor_descriptor)
        if bound
        else {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_files_only": True,
            "processor_class": None,
            "processor_signature_sha256": None,
            "reviewed": False,
            "runtime_match_required": True,
            "runtime_verified": False,
        }
    )
    result = {
        "schema": "evicastkv_confirmation_input_dependencies_v1",
        "status": (
            "bound_reviewed_processor"
            if bound
            else "resolved_source_route_processor_capture_required"
        ),
        "classification": "dependency_binding_only_not_empirical_evidence",
        "route": {
            "task": "coco_caption",
            "lmms_task": "coco2014_cap_val",
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "split": "val",
            "config_name": "default",
            "reference_field": "answer",
            "prompt": COCO_PROMPT,
        },
        "route_source_bindings": bindings,
        "source_binding_scope": "caller_supplied_pinned_hashes_no_files_opened",
        "model": model,
        "processor": processor,
        "processor_bound": bound,
        "unresolved": [] if bound else ["reviewed_actual_processor_capture_required"],
        "certifications": {
            "dataset_content_verified": False,
            "media_provenance_verified": False,
            "freshness_certified": False,
            "runtime_processor_verified": False,
        },
        "execution_enabled": False,
        "authorization": dict(protocol["authorization"]),
    }
    result["binding_sha256"] = digest(result)
    return result


def assert_runtime_processor_matches(
    binding: dict[str, Any], actual_signature: dict[str, Any]
) -> None:
    """Reject a runtime processor different from the reviewed original signature."""
    _primitives(binding)
    if (
        type(binding) is not dict
        or binding.get("schema") != "evicastkv_confirmation_input_dependencies_v1"
        or binding.get("status") != "bound_reviewed_processor"
        or binding.get("processor_bound") is not True
        or binding.get("execution_enabled") is not False
    ):
        raise ValueError("reviewed bound processor dependency object required")
    disabled_authorization(binding.get("authorization"))
    saved_binding = dict(binding)
    expected_binding_hash = saved_binding.pop("binding_sha256", None)
    if _hash(expected_binding_hash) != digest(saved_binding):
        raise ValueError("input dependency binding changed")
    expected = binding.get("processor")
    if type(expected) is not dict:
        raise ValueError("bound processor object required")
    if (
        expected.get("id") != MODEL_ID
        or expected.get("revision") != MODEL_REVISION
        or expected.get("local_files_only") is not True
        or expected.get("reviewed") is not True
    ):
        raise ValueError("pinned reviewed runtime processor identity differs")
    actual = processor_fingerprints(actual_signature)
    if expected.get("processor_class") != actual_signature["processor_class"] or any(
        expected.get(key) != value for key, value in actual.items()
    ):
        raise ValueError("actual runtime processor differs from reviewed signature")
