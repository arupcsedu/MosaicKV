#!/usr/bin/env python3
"""Freeze COCO val images/references and pinned processor assets, CPU-only.

Requires an explicit ID-only parent pool (e.g. historical 4096 IDs). Reads only
the 14 original validation Arrow shards, never downloads/prepares other splits.
Output is immutable; no inference or job submission happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_prefixkv_decode_preflight import write_once

from mosaickv.evaluation.coco_shared_bytes import COCO_PROMPT, COCO_REVISION, processor_signature
from mosaickv.evaluation.lmms_adapter import _source_ids


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def freeze_sample(doc, sid, root):
    from lmms_eval.tasks.coco_cap.utils import coco_doc_to_text, coco_doc_to_visual

    references = doc["answer"]
    # The pinned COCO split includes images with six or seven annotations.
    # Preserve every annotation in source order, including repeated captions;
    # exactly five is not an evaluator requirement. Fewer than five indicates
    # an incomplete record for this route and must still fail closed.
    if (
        not isinstance(references, list)
        or len(references) < 5
        or any(not isinstance(r, str) or not r.strip() for r in references)
    ):
        raise ValueError(
            "COCO needs all supplied caption references (at least five nonempty strings)"
        )
    prompt = coco_doc_to_text(doc)
    if prompt != COCO_PROMPT:
        raise ValueError("installed COCO prompt differs from frozen route")
    images = coco_doc_to_visual(doc)
    if len(images) != 1:
        raise ValueError("COCO route requires one image")
    path = root / "media" / (hashlib.sha256(sid.encode()).hexdigest() + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        images[0].save(stream, format="PNG")
    return dict(
        sample_id=sid,
        prompt=prompt,
        references=references,
        media=[dict(kind="image", path=str(path), sha256=sha(path))],
    )


def historical_id_pool(root):
    """Reuse identities only, never captions/predictions from BF16 execution."""
    ids, bindings, precisions = [], {}, set()
    files = sorted(root.glob("shard-*/raw.jsonl"))
    if not files:
        raise ValueError("no historical FullKV raw shards")
    for path in files:
        manifest = path.parent / "manifest.json"
        provenance = json.loads(manifest.read_text())
        if provenance["dataset"]["revision"] != COCO_REVISION:
            raise ValueError("historical dataset revision mismatch")
        precisions.add(provenance["model"]["precision"])
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if (
                row["task"] != "coco_caption"
                or row["method"] != "full_kv"
                or row["status"] != "completed"
            ):
                raise ValueError("historical pool is not complete FullKV COCO")
            ids.append(row["sample_id"])
        bindings[str(path)] = sha(path)
        bindings[str(manifest)] = sha(manifest)
    if len(ids) != 4096 or len(set(ids)) != len(ids):
        raise ValueError("historical pool must contain exactly 4096 unique identities")
    return dict(
        schema_version=1,
        task="coco_caption",
        dataset_id="lmms-lab/COCO-Caption",
        dataset_revision=COCO_REVISION,
        split="val",
        sample_ids=sorted(ids),
        bound_files=bindings,
        historical_precisions=sorted(precisions),
        prediction_reuse=False,
        reuse_decision="IDs only; historical nominal ratios/precision/measurement controls differ",
    )


def materialize(args):
    from datasets import Dataset, concatenate_datasets
    from lmms_eval.tasks.coco_cap import utils
    from transformers import AutoProcessor

    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, args.source_sha)
    root = args.root.resolve()
    if (
        root.exists()
        or Path("/scratch/djy8hg").resolve() not in root.parents
        or source in root.parents
    ):
        raise ValueError("require new immutable external scratch root")
    payload = (
        json.loads(args.ids.read_text())
        if args.ids
        else historical_id_pool(args.historical_fullkv_root)
    )
    if any(
        payload.get(k) != v
        for k, v in dict(
            task="coco_caption",
            dataset_id="lmms-lab/COCO-Caption",
            dataset_revision=COCO_REVISION,
            split="val",
        ).items()
    ):
        raise ValueError("ID pool provenance mismatch")
    ids = payload["sample_ids"]
    if len(ids) < 288 or len(ids) != len(set(ids)):
        raise ValueError("need at least 288 distinct frozen identities")
    cache = args.arrow_cache.resolve(strict=True)
    if cache.name != COCO_REVISION or Path("/scratch/djy8hg").resolve() not in cache.parents:
        raise ValueError("requires pinned scratch dataset cache")
    files = [cache / f"coco-caption-val-{i:05d}-of-00014.arrow" for i in range(14)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset = concatenate_datasets([Dataset.from_file(str(p)) for p in files])
    index = {sid: i for i, sid in enumerate(_source_ids(dataset, "coco_caption"))}
    if not set(ids) <= set(index):
        raise ValueError("ID pool contains identities absent from pinned validation split")
    revision = "2424fdd47412fccc66d91719126b420e9fbd7065"
    snapshot = args.model_snapshot.resolve(strict=True)
    if snapshot.name != revision or Path("/scratch/djy8hg").resolve() not in snapshot.parents:
        raise ValueError("requires pinned scratch model snapshot")
    assets = [
        p for p in snapshot.iterdir() if p.suffix in {"json", ".json", ".model", ".txt", ".jinja"}
    ]
    required = {"config.json", "preprocessor_config.json", "tokenizer_config.json"}
    if not required <= {p.name for p in assets}:
        raise ValueError("model snapshot lacks processor/config assets")
    id_path = args.ids or root / "historical_ids.json"
    if args.ids is None:
        write_once(id_path, payload)
    bound = {
        str(p): sha(p)
        for p in [
            *files,
            cache / "dataset_info.json",
            id_path,
            *assets,
            Path(utils.__file__),
            Path(utils.__file__).parent / "coco2014_cap_val.yaml",
        ]
    }
    samples = [freeze_sample(dataset[index[sid]], sid, root) for sid in ids]
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    write_once(
        root / "ledger.json",
        dict(
            schema_version=1,
            task="coco_caption",
            dataset_id="lmms-lab/COCO-Caption",
            dataset_revision=COCO_REVISION,
            split="val",
            samples=samples,
            source_sha=args.source_sha,
            parent_id_pool_sha256=sha(id_path),
            bound_files=bound,
            model_id="llava-hf/llava-v1.6-mistral-7b-hf",
            model_revision=revision,
            processor_signature=processor_signature(processor),
            historical_prior_use_disclosure=args.prior_use,
            heldout_claim=False,
            precision="fp16",
            historical_bf16_prediction_reuse=False,
            preprocessing="pinned lmms RGB -> lossless PNG -> pinned HF processor",
        ),
    )
    print(
        json.dumps(
            dict(
                ledger=str(root / "ledger.json"),
                sha256=sha(root / "ledger.json"),
                samples=len(samples),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    ids = parser.add_mutually_exclusive_group(required=True)
    ids.add_argument("--ids", type=Path)
    ids.add_argument("--historical-fullkv-root", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arrow-cache", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--prior-use", required=True, help="disclose historical cohort use; not held-out"
    )
    materialize(parser.parse_args())
