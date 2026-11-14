"""Pure validation of a finite COCO image-identity availability certificate.

This is not a universal crop/scene/source-family guarantee and grants no job
submission permission. Its immutable inputs must come from audited metadata,
not inferred innocence from an absent hash or an inaccessible history source.
"""

from __future__ import annotations

import re
from collections import Counter

from mosaickv.evaluation.coco_shared_bytes import COCO_REVISION
from mosaickv.evaluation.fox_campaign import digest

CONTRACT = "COCO-exact-ID-encoded-EXIF-RGB-within-declared-history-v1"
COCO = re.compile(r"COCO_val2014_\d{12}\.jpg")
HISTORICAL_COCO = re.compile(r"COCO_(?:train|val)2014_\d{12}\.jpg")
SHA = re.compile(r"[0-9a-f]{64}")
HASHES = ("encoded_sha256", "pixel_sha256")


def evaluate(scope, history, candidates):
    """Evaluate complete, hash-bound input metadata; never discover/select data.

    ``candidates`` is the complete registered ID pool with two media hashes.
    ``history`` enumerates declared relevant image assets plus canonical COCO
    IDs. Every historical coverage gap needs an explicit audited resolution;
    original negative evidence remains in its immutable parent artifacts.
    """
    errors = []
    if (
        scope.get("contract") != CONTRACT
        or scope.get("dataset_id") != "lmms-lab/COCO-Caption"
        or scope.get("dataset_revision") != COCO_REVISION
        or scope.get("split") != "val"
    ):
        errors.append("registered finite COCO identity contract differs")
    roots = scope.get("historical_roots", [])
    if (
        not roots
        or len(roots) != len(set(roots))
        or sorted(roots) != sorted(history.get("historical_roots", []))
    ):
        errors.append("complete declared historical-root scope not matched")
    if (
        history.get("metadata_scope_complete") is not True
        or history.get("media_scope_complete") is not True
    ):
        errors.append("declared historical identity/media scope incomplete")
    unresolved = history.get("unresolved_sources")
    if type(unresolved) is not list or unresolved:
        errors.append("unresolved historical sources remain")
    bindings = history.get("bound_artifacts", {})
    if not bindings or any(
        not isinstance(v, str) or not SHA.fullmatch(v) for v in bindings.values()
    ):
        errors.append("audited historical artifacts are not hash-bound")
    assets = history.get("image_records", [])
    keys = [row.get("asset_key") for row in assets]
    if (
        len(assets) != history.get("declared_image_assets")
        or len(keys) != len(set(keys))
        or any(not isinstance(k, str) or not k for k in keys)
    ):
        errors.append("historical image-asset inventory incomplete or duplicated")
    if any(
        not isinstance(row.get(k), str) or not SHA.fullmatch(row[k])
        for row in assets
        for k in HASHES
    ):
        errors.append("historical encoded/canonical-RGB media hashes incomplete")
    old_ids = history.get("canonical_coco_ids")
    if not isinstance(old_ids, list) or any(
        not isinstance(value, str) or not HISTORICAL_COCO.fullmatch(value) for value in old_ids
    ):
        errors.append("historical canonical COCO identity list missing or malformed")
    if any(
        not isinstance(row.get("sample_id"), str)
        or not COCO.fullmatch(row["sample_id"])
        or any(not isinstance(row.get(k), str) or not SHA.fullmatch(row[k]) for k in HASHES)
        for row in candidates
    ):
        errors.append("candidate canonical identity/media hashes incomplete")
    ids = [row.get("sample_id") for row in candidates]
    if (
        len(ids) != len(set(ids))
        or len(ids) != scope.get("candidate_count")
        or digest(sorted(ids, key=str)) != scope.get("candidate_IDs_sha256")
    ):
        errors.append("complete registered candidate ID pool differs")
    minimum = scope.get("minimum_available_images")
    if type(minimum) is not int or minimum < 1:
        errors.append("positive preregistered availability count required")
    if errors:
        return dict(
            contract=CONTRACT,
            scope_certified=False,
            eligible_for_preregistered_image_identity_cohort=False,
            errors=errors,
            input_sha256=digest([scope, history, candidates]),
            experiment_submission_authorized=False,
            universal_crop_resize_scene_independence_claimed=False,
        )
    old_ids = set(old_ids)
    old_hashes = {kind: {row[kind] for row in assets} for kind in HASHES}
    duplicate_pixels = {
        key
        for key, count in Counter(row["pixel_sha256"] for row in candidates).items()
        if count > 1
    }
    reasons, available = Counter(), []
    for row in candidates:
        excluded = []
        if row["sample_id"] in old_ids:
            excluded.append("canonical_COCO_ID_overlap")
        for kind in HASHES:
            if row[kind] in old_hashes[kind]:
                excluded.append(f"historical_{kind}_overlap")
        if row["pixel_sha256"] in duplicate_pixels:
            excluded.append("within_pool_pixel_duplicate_exclude_all_members")
        reasons.update(excluded)
        if not excluded:
            available.append(row["sample_id"])
    return dict(
        contract=CONTRACT,
        scope_certified=True,
        errors=[],
        eligible_for_preregistered_image_identity_cohort=len(available) >= minimum,
        available_images=len(available),
        available_IDs_sha256=digest(sorted(available)),
        exclusion_counts=dict(sorted(reasons.items())),
        duplicate_policy="exclude_all_members_of_each_canonical_pixel_duplicate_class",
        input_sha256=digest([scope, history, candidates]),
        certificate_scope=(
            "declared historical inputs; canonical ID, encoded bytes, EXIF-RGB pixels"
        ),
        residual_limitations=[
            "unobserved crops/resizes",
            "shared scene or source-series independence",
        ],
        universal_crop_resize_scene_independence_claimed=False,
        cohort_selected=False,
        experiment_submission_authorized=False,
    )
