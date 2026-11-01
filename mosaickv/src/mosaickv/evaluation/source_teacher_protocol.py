"""Reference data access only after immutable online formats have been built."""

import json
from pathlib import Path

from mosaickv.evaluation.fox_campaign import sha
from mosaickv.evaluation.prefixkv_metrics import target_artifact_sha256
from mosaickv.evaluation.storage_teacher_forcing import validate_targets


def load_targets(plan, payload, tokenizer):
    path = Path(plan["teacher_ledgers"][payload["task"]])
    if sha(path) != plan["bound_files"][str(path)]:
        raise ValueError("native teacher ledger changed")
    ledger = json.loads(path.read_text())
    matches = [s for s in ledger["samples"] if s["sample_id"] == payload["sample_id"]]
    if len(matches) != 1:
        raise ValueError("exactly one same-input native annotation required")
    sample = matches[0]
    if sample["prompt"] != payload["prompt"] or sample["media"] != payload["media"]:
        raise ValueError("native teacher prompt/media do not match consumed input")
    target = sample["teacher_forcing_target"]
    targets = tuple(tokenizer(target, add_special_tokens=False)["input_ids"])
    validate_targets(targets, maximum=plan["teacher_maximum_tokens"])
    return targets, dict(
        ledger_path=str(path),
        ledger_sha256=sha(path),
        target_artifact_sha256=target_artifact_sha256({payload["sample_id"]: target}),
        dataset={k: ledger[k] for k in ("dataset_id", "dataset_revision", "split")},
        sample_id=payload["sample_id"],
        evaluated_token_count=len(targets),
        no_added_EOS_or_BOS=True,
        accessed_only_after_immutable_formats_constructed=True,
        annotation_not_used_for_fitting=True,
    )
