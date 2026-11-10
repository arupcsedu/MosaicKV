"""CPU-only descriptive caption audit of the complete, consumed COCO cohort.

This does not choose examples, change the evaluator, or explain quality gains
causally. Per-image CIDEr contributions use the complete point's reference DF;
they are never reused as a corpus bootstrap or independent quality observations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from evicastkv_analyze import collect

from mosaickv.evaluation.coco_shared_bytes import TokenizedCider
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.metrics.types import MetricSample

SOURCE = Path(__file__).resolve().parents[2]
METHODS = ("prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")


def caption_features(row, trace):
    """Validate the scored boundary before describing the already-saved answer."""
    tokens = row["token_ids"]
    quality = trace["quality_decoding"]
    terminals = set(quality["configured_terminal_token_ids"])
    first = next((i for i, token in enumerate(tokens) if token in terminals), None)
    scored = first if first is not None else len(tokens)
    if (
        row["status"] != "completed"
        or row["answer"] != trace["answer"]
        or tokens != trace["generated_token_ids"]
        or len(tokens) != 64
        or quality["timed_token_count"] != 64
        or quality["terminal_token_index"] != first
        or quality["scored_token_count"] != scored
        or row["forward_steps"] != 63
    ):
        raise ValueError("scored/timed token boundaries or saved prediction disagree")
    references = row["references"]
    if not references or any(not isinstance(ref, str) or not ref.strip() for ref in references):
        raise ValueError("all nonempty reference annotations must be retained")
    words = re.findall(r"\w+(?:['\u2019-]\w+)*", row["answer"].casefold())
    grams = [tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    repeated = sum(count - 1 for count in Counter(grams).values())
    return dict(
        scored_tokens=scored,
        timed_tokens=len(tokens),
        word_count=len(words),
        character_count=len(row["answer"]),
        terminal_observed=first is not None,
        output_limit_hit=first is None,
        empty_answer=not row["answer"].strip(),
        fourgram_count=len(grams),
        repeated_fourgram_count=repeated,
        repeated_fourgram_fraction=repeated / len(grams) if grams else 0.0,
        reference_count=len(references),
    )


def descriptive(values):
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("complete finite observations required")
    return dict(
        mean=float(array.mean()),
        median=float(np.median(array)),
        p95=float(np.quantile(array, 0.95)),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


def correlation(left, right):
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.shape != y.shape or len(x) < 2 or not np.isfinite([x, y]).all():
        raise ValueError("aligned finite association inputs required")
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def audit(root, output):
    import pandas as pd

    revision = clean_source(SOURCE)
    if output.exists():
        raise FileExistsError("derived caption-audit output must be new")
    lock_path = root / "protocol_lock.json"
    lock = json.loads(lock_path.read_text())
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    gate_path = root / "analysis/scoring/GATE_scoring.json"
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "passed" or gate["protocol_sha256"] != sha(lock_path):
        raise ValueError("hash-bound scoring gate must pass before caption analysis")
    _spec, workers, rows, errors = collect(root, "scoring")
    rows = [row for row in rows if row["workload"] == "coco"]
    if errors or not workers:
        raise ValueError("failed/incomplete scoring is not a surviving caption cohort")
    info = lock["cohorts"]["coco"]["validation"]
    ledger = Path(info["path"])
    if sha(ledger) != info["sha256"]:
        raise ValueError("input ledger changed")
    samples = json.loads(ledger.read_text())["samples"]
    refs = {s["sample_id"]: s["references"] for s in samples}
    expected = {(sid, "full_kv", None) for sid in refs} | {
        (sid, method, cap * 2**20) for sid in refs for method in METHODS for cap in (20, 40)
    }
    if (
        len(refs) != 32
        or len(rows) != len(expected)
        or {(r["sample_id"], r["method"], r["budget_bytes"]) for r in rows} != expected
    ):
        raise ValueError("complete registered 32-input/seven-arm cohort required")
    bindings = {str(p): sha(p) for p in (lock_path, ledger, gate_path)}
    groups = defaultdict(list)
    observations = []
    for row in rows:
        if row["references"] != refs[row["sample_id"]]:
            raise ValueError("reference list differs from the immutable annotation ledger")
        path = Path(row["trace_path"])
        features = caption_features(row, json.loads(path.read_text()))
        obs = dict(
            sample_id=row["sample_id"],
            method=row["method"],
            budget_bytes=row["budget_bytes"],
            source_run_id=row["run_id"],
            manifest_path=row["manifest_path"],
            source_sha=row["source_sha"],
            config_sha=row["config_sha"],
            trace_path=str(path),
            trace_sha256=sha(path),
            **features,
        )
        observations.append(obs)
        groups[(row["method"], row["budget_bytes"])].append((row, obs))
        bindings[str(path)] = row["trace_sha256"]
        bindings[row["manifest_path"]] = row["manifest_sha256"]
    point_samples = {
        key: tuple(
            MetricSample(r["sample_id"], r["answer"], tuple(r["references"]))
            for r, _ in sorted(values, key=lambda item: item[0]["sample_id"])
        )
        for key, values in groups.items()
    }
    tokenizer = TokenizedCider([s for values in point_samples.values() for s in values])
    native_path = root / "analysis/scoring/native/native_metrics.json"
    native = json.loads(native_path.read_text())[0]
    summaries = []
    for (method, budget), values in sorted(groups.items(), key=str):
        ordered = sorted(values, key=lambda item: item[0]["sample_id"])
        samples = point_samples[method, budget]
        pairs = [tokenizer._tokens[(s.prediction, s.references)] for s in samples]
        corpus, contributions = tokenizer._scorer().compute_score(
            {str(i): pair[0] for i, pair in enumerate(pairs)},
            {str(i): pair[1] for i, pair in enumerate(pairs)},
        )
        point = f"{method}:{budget if budget is not None else 'full'}"
        if not math.isclose(float(corpus) * 100, native["scores"][point], abs_tol=1e-10, rel_tol=0):
            raise ValueError(
                "descriptive CIDEr point differs from original official corpus reducer"
            )
        for (_, obs), score in zip(ordered, contributions, strict=True):
            obs["descriptive_cider_contribution"] = float(score) * 100
        scored = [obs["scored_tokens"] for _, obs in ordered]
        words = [obs["word_count"] for _, obs in ordered]
        summaries.append(
            dict(
                method=method,
                budget_bytes=budget,
                attempted=32,
                scored=32,
                failed=0,
                corpus_cider=float(corpus) * 100,
                scored_tokens=descriptive(scored),
                words=descriptive(words),
                reference_counts=dict(
                    sorted(Counter(obs["reference_count"] for _, obs in ordered).items())
                ),
                terminal_observed=sum(obs["terminal_observed"] for _, obs in ordered),
                output_limit_hit=sum(obs["output_limit_hit"] for _, obs in ordered),
                empty_answers=sum(obs["empty_answer"] for _, obs in ordered),
                captions_with_repeated_fourgrams=sum(
                    obs["repeated_fourgram_count"] > 0 for _, obs in ordered
                ),
                scored_tokens_vs_cider_contribution_pearson=correlation(scored, contributions),
                words_vs_cider_contribution_pearson=correlation(words, contributions),
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(observations).to_parquet(output / "parsing_audit.parquet", index=False)
    pd.DataFrame(observations).to_csv(output / "parsing_audit.csv", index=False)
    write_once(output / "caption_diagnostics.json", summaries)
    bindings[str(native_path)] = sha(native_path)
    for worker in workers:
        for binding in worker["records"]:
            bindings[binding["path"]] = binding["sha256"]
    write_once(
        output / "GATE_caption_audit.json",
        dict(
            status="passed_descriptive_consumed_development_audit",
            source_sha=revision,
            measurement_type="derived",
            attempted=224,
            audited=224,
            failed=0,
            new_GPU_inference=0,
            all_reference_annotations_preserved=True,
            no_post_EOS_scoring=True,
            metrics_not_changed=True,
            descriptive_associations_not_causal_explanations=True,
            per_image_CIDEr_not_used_as_corpus_bootstrap=True,
            fresh_confirmation=False,
            superiority_established=False,
            bindings=bindings,
            audit_script_sha256=sha(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
