#!/usr/bin/env python3
"""Consumed-input, fixed-domain independent utility/coverage ablations.

This evaluation-only overlay changes registered objective coefficients, not
production math. All eight arms retain the same nonzero routing quota, graph
candidate domain and exact physical capacity. Cross-modal removal is the one
explicit edge intervention; it is not a graph-coverage ablation. No seed search,
new cohort, timing claim, winner freezing or confirmation is authorized here.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from evicastkv_campaign import SOURCE, load_lock, resolved_config
from evicastkv_development import runtime_files

from mosaickv.config import canonical_config
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

STAGE = "mechanisms"
CAP = 40 * 2**20
ROUTING = 1 / 16
SAMPLES = 16
VARIANTS = (
    "term-full",
    "term-attention-only",
    "term-attention-plus-value",
    "term-value-off",
    "term-novelty-off",
    "term-graph-coverage-off",
    "term-modality-coverage-off",
    "term-cross-modal-off",
)
FAMILY_SIZE = len(VARIANTS) - 1
DOMAIN_FIELDS = (
    "candidate_universe_sha256",
    "shortlist_sha256",
    "quota_sha256",
    "descriptors_sha256",
    "optimization_node_ids_sha256",
    "planner_routes",
    "graph_routes",
    "graph_nodes",
)


def arms():
    return [["mosaickv_exact", 40, variant] for variant in VARIANTS]


def jobs():
    return [
        dict(
            workload="coco",
            subset="development",
            stage=STAGE,
            indices=list(range(start, start + 4)),
            arms=arms(),
        )
        for start in range(0, SAMPLES, 4)
    ]


def coefficients(config):
    return dict(
        lambda_q=config.utility.lambda_q,
        lambda_v=config.utility.lambda_v,
        lambda_o=config.utility.lambda_o,
        lambda_g=config.selection.lambda_g,
        lambda_m=config.selection.lambda_m,
    )


def term_config(config, variant):
    if variant not in VARIANTS or config.cache.budget_value != CAP:
        raise ValueError("registered independent term and exact 40 MiB cap required")
    utility, selection = config.utility, config.selection
    if variant in {"term-attention-only", "term-value-off"}:
        utility = replace(utility, lambda_v=0.0)
    if variant in {"term-attention-only", "term-attention-plus-value", "term-novelty-off"}:
        utility = replace(utility, lambda_o=0.0)
    if variant in {"term-attention-only", "term-attention-plus-value", "term-graph-coverage-off"}:
        selection = replace(selection, lambda_g=0.0)
    if variant in {
        "term-attention-only",
        "term-attention-plus-value",
        "term-modality-coverage-off",
    }:
        selection = replace(selection, lambda_m=0.0)
    graph = replace(config.graph, uncertain_head_fraction=ROUTING)
    if variant == "term-cross-modal-off":
        graph = replace(graph, cross_modal_weight=0.0)
    return replace(
        config,
        utility=utility,
        selection=selection,
        graph=graph,
        prototypes=replace(config.prototypes, uncertain_head_fraction=ROUTING),
    )


@contextlib.contextmanager
def config_overlay():
    import evicastkv_worker as worker

    import mosaickv.evaluation.fox_mechanisms as audits

    original_config, original_audit = worker.resolved_config, audits.mechanism_audit

    def resolved(lock, workload, method, cap, *, variant="complete"):
        if variant.startswith("term-"):
            if workload != "coco" or method != "mosaickv_exact" or cap != 40:
                raise ValueError("unregistered independent-term workload/method/cap")
            return term_config(original_config(lock, workload, method, cap), variant)
        base = original_config(lock, workload, method, cap, variant=variant)
        if (
            workload == "coco"
            and method == "mosaickv_exact"
            and cap == 40
            and variant == "complete"
        ):
            # The worker passes this full policy to the graph-deletion audit.
            return term_config(base, "term-full")
        return base

    def audit(variant, *args, **kwargs):
        if variant not in VARIANTS:
            raise ValueError("unregistered independent-term audit variant")
        mapped = "cross-modal-off" if variant == "term-cross-modal-off" else "mechanism-complete"
        return original_audit(mapped, *args, **kwargs)

    with (
        patch.object(worker, "resolved_config", resolved),
        patch.object(audits, "mechanism_audit", audit),
    ):
        yield


def comparisons():
    return [("term-full", variant) for variant in VARIANTS[1:]]


def prepare(root, parent, mechanisms):
    source_sha = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("independent-term output root must be new")
    parent_path = parent / "protocol_lock.json"
    lock = json.loads(parent_path.read_text())
    prior = Path(lock["source_root"])
    clean_source(prior, lock["source_sha"])
    if runtime_files(SOURCE) != runtime_files(prior):
        raise ValueError("production runtime differs from passed controls")
    bindings = {str(parent_path): sha(parent_path)}
    inherited = {}
    for stage in ("correctness", "scoring", "profiling", "measurement"):
        path = parent / "analysis" / stage / f"GATE_{stage}.json"
        gate = json.loads(path.read_text())
        if (
            gate.get("status") != "passed"
            or gate.get("blockers")
            or gate.get("source_sha") != lock["source_sha"]
            or gate.get("protocol_sha256") != sha(parent_path)
        ):
            raise ValueError(f"parent gate not passed or bound: {stage}")
        inherited[str(path)] = sha(path)
    mechanism_lock = mechanisms / "protocol_lock.json"
    mechanism_gate = mechanisms / "analysis/mechanisms/GATE_mechanisms.json"
    mlock, mgate = json.loads(mechanism_lock.read_text()), json.loads(mechanism_gate.read_text())
    if (
        mgate.get("status") != "passed"
        or mgate.get("blockers")
        or mgate.get("protocol_sha256") != sha(mechanism_lock)
        or mgate.get("source_sha") != mlock.get("source_sha")
    ):
        raise ValueError("completed, hash-bound mechanism prerequisite required")
    for name in ("evicastkv_worker.py", "evicastkv_campaign.py"):
        relative = "mosaickv/scripts/" + name
        if sha(SOURCE / relative) != sha(prior / relative):
            raise ValueError("shared execution/config resolver changed")
    cohort = lock["cohorts"]["coco"]["development"]
    mcohort = mlock["cohorts"]["coco"]["development"]
    if cohort != mcohort or sha(Path(cohort["path"])) != cohort["sha256"]:
        raise ValueError("consumed mechanism cohort changed")
    identities = list(cohort["sample_ids"][:SAMPLES])
    if len(identities) != SAMPLES or len(set(identities)) != SAMPLES:
        raise ValueError("sixteen distinct consumed IDs required")
    base = term_config(resolved_config(lock, "coco", "mosaickv_exact", 40), "term-full")
    if any(value == 0 for value in coefficients(base).values()):
        raise ValueError("full policy must have every registered term nonzero")
    lock = copy.deepcopy(lock)
    lock.update(
        name="evicastkv_independent_terms_consumed_v1",
        source_root=str(SOURCE),
        source_sha=source_sha,
    )
    lock["resources"].update(
        maximum_a100s=6,
        array_concurrency=4,
        six_a100s_explicitly_authorized=True,
        cpus_per_worker=12,
        wall_minutes=120,
    )
    lock["authorization"].update(automatic_stage_progression=False, confirmation_authorized=False)
    lock.setdefault("inherited_gates", {}).update(inherited)
    lock["historical_bindings"].update(bindings)
    lock["historical_bindings"].update(
        {str(mechanism_lock): sha(mechanism_lock), str(mechanism_gate): sha(mechanism_gate)}
    )
    lock["independent_terms"] = dict(
        sample_ids=identities,
        variants=list(VARIANTS),
        initial_cap_bytes=CAP,
        routing_fraction=ROUTING,
        full_coefficients=coefficients(base),
        resolved_coefficients={
            variant: coefficients(term_config(base, variant)) for variant in VARIANTS
        },
        full_config=canonical_config(base),
        expected_rows=SAMPLES * len(VARIANTS),
        bootstrap_resamples=10000,
        statistical_family_size=FAMILY_SIZE,
        contrasts=comparisons(),
        scope="consumed development; independent fixed-domain objective terms",
        contrast_direction="ablation minus complete policy",
        fresh_inputs=False,
        winner_freezing_authorized=False,
        headline_latency_eligible=False,
        cross_modal_intervention=(
            "delete cross-modality edges without refill; other arms keep graph"
        ),
        graph_coverage_intervention="lambda_g=0 only; graph retained",
        modality_coverage_intervention="lambda_m=0 only; graph retained",
        already_complete_random_and_routing_arms_not_rerun=True,
    )
    write_once(root / "protocol_lock.json", lock)
    write_once(
        root / "jobs" / f"{STAGE}.json",
        dict(protocol_sha256=sha(root / "protocol_lock.json"), jobs=jobs()),
    )
    load_lock(root)
    return dict(
        root=str(root),
        workers=4,
        expected_rows=128,
        stage=STAGE,
        spec_sha256=sha(root / "jobs" / f"{STAGE}.json"),
    )


def worker(root, index, spec_sha):
    from evicastkv_worker import run_worker

    load_lock(root)
    path = root / "jobs" / f"{STAGE}.json"
    if (
        sha(path) != spec_sha
        or json.loads(path.read_text()).get("jobs") != jobs()
        or type(index) is not int
        or index not in range(4)
    ):
        raise ValueError("registered independent-term grid/index/hash required")
    with config_overlay():
        run_worker(root, STAGE, index, spec_sha)


def validate_rows(rows, contract):
    from mosaickv.config import RunConfig

    ids = contract["sample_ids"]
    expected = {(sid, variant) for sid in ids for variant in VARIANTS}
    actual = [(row["sample_id"], row["variant"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("incomplete or duplicate independent-term grid")
    base = RunConfig.from_mapping(contract["full_config"])
    records, domains, controls, graph_bindings, storage, anchors, selections = (
        [],
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for row in rows:
        sid, variant = row["sample_id"], row["variant"]
        if (
            row["status"] != "completed"
            or row.get("effective_method") != "mosaickv_exact"
            or row["method"] != "mosaickv_exact"
        ):
            raise ValueError("failed sample or fallback in independent terms")
        if (
            row["budget_bytes"] != CAP
            or row["initial_kv_bytes"] != CAP
            or row.get("emitted_tokens") != 64
            or row.get("forward_steps") != 63
        ):
            raise ValueError("registered physical cap/generation changed")
        config_path, audit_path = (
            Path(row["resolved_config_path"]),
            Path(row["mechanism_audit_path"]),
        )
        if (
            sha(config_path) != row["resolved_config_file_sha256"]
            or sha(audit_path) != row["mechanism_audit_sha256"]
        ):
            raise ValueError("resolved configuration or activity artifact changed")
        config = json.loads(config_path.read_text())
        if digest(config) != digest(canonical_config(term_config(base, variant))):
            raise ValueError("unregistered coefficient, route or common-control change")
        activity = json.loads(audit_path.read_text())
        raw = activity.get("raw_terms", {})
        if len(raw) != 2 or {Path(path).name for path in raw} != {
            "raw_marginal_terms.parquet",
            "edges.parquet",
        }:
            raise ValueError("complete raw marginal/edge artifacts required")
        for path, value in raw.items():
            if sha(Path(path)) != value:
                raise ValueError("raw marginal/edge artifact changed")
        domain = digest({name: activity[name] for name in DOMAIN_FIELDS})
        if sid in domains and domains[sid] != domain:
            raise ValueError("candidate/descriptor/quota/routed-head domain changed")
        domains[sid] = domain
        common = digest(
            dict(
                controls=[
                    row[key]
                    for key in ("input_sha", "processor_sha", "scorer_sha", "forward_steps")
                ],
                references=row["references"],
                media_group_ids=row["media_group_ids"],
            )
        )
        if not row["references"] or (sid in controls and controls[sid] != common):
            raise ValueError("matched prompt/media/reference/evaluator controls changed")
        controls[sid] = common
        bytes_now = tuple(
            row[name] for name in ("initial_kv_bytes", "final_kv_bytes", "peak_kv_bytes")
        )
        if sid in storage and storage[sid] != bytes_now:
            raise ValueError("measured initial/final/peak bytes differ")
        storage[sid] = bytes_now
        # Other objective ablations must retain the exact full graph; deleting
        # cross-modal edges is explicitly the sole exception.
        edge_hash = next(value for path, value in raw.items() if Path(path).name == "edges.parquet")
        if variant != "term-cross-modal-off":
            if sid in graph_bindings and graph_bindings[sid] != edge_hash:
                raise ValueError("objective-only ablation changed graph edges")
            graph_bindings[sid] = edge_hash
        chosen = set(activity["selected_source_ids"])
        if not chosen or not activity["graph_nodes"] or not activity["graph_routes"]:
            raise ValueError("nonzero routing/selection activity required")
        selections[(sid, variant)] = chosen
        if variant == "term-full":
            anchors[sid] = chosen
        records.append(
            dict(
                sample_id=sid,
                variant=variant,
                coefficients=coefficients(term_config(base, variant)),
                graph_nodes=activity["graph_nodes"],
                graph_edges=activity["graph_edges"],
                cross_modal_edges_removed=activity.get("cross_modal_edges_removed", 0),
                raw_terms=raw,
                fixed_domain_sha256=domain,
            )
        )
    for record in records:
        anchor = anchors[record["sample_id"]]
        selected = selections[(record["sample_id"], record["variant"])]
        record["selected_jaccard_vs_full"] = len(anchor & selected) / len(anchor | selected)
        record["changed_source_memberships_vs_full"] = len(anchor ^ selected)
    return records


def analyze(root):
    from evicastkv_analyze import collect, export_descriptive

    from mosaickv.evaluation.fox_statistics import corpus_cider_statistics
    from mosaickv.evaluation.metrics.types import MetricSample

    lock = load_lock(root)
    _, _, rows, errors = collect(root, STAGE)
    output = root / "analysis" / STAGE
    try:
        activities = validate_rows(rows, lock["independent_terms"])
    except (ValueError, KeyError, FileNotFoundError) as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
        activities = []
    write_once(output / "failures.json", errors)
    write_once(output / "independent_term_activity.json", activities)
    gate = dict(
        status="blocked" if errors else "passed_exploratory_not_confirmation",
        actual_rows=len(rows),
        expected_rows=128,
        blockers=errors,
        source_sha=lock["source_sha"],
        protocol_sha256=sha(root / "protocol_lock.json"),
        fresh_confirmation_authorized=False,
        winner_freezing_authorized=False,
        superiority_claim_allowed=False,
        headline_latency_eligible=False,
        all_points_reported=True,
    )
    if not errors:
        export_descriptive(root, STAGE, rows)
        points, groups = {}, {}
        for row in rows:
            points.setdefault(row["variant"], []).append(
                MetricSample(row["sample_id"], row["answer"], tuple(row["references"]))
            )
            groups[row["sample_id"]] = tuple(row["media_group_ids"])
        stats = corpus_cider_statistics(
            {key: tuple(value) for key, value in points.items()},
            comparisons(),
            groups,
            resamples=10000,
            seed=lock["seed"],
            workers=1,
            family_size=FAMILY_SIZE,
        )
        write_once(output / "corpus_cider_statistics.json", stats)
        gate.update(
            contrasts=FAMILY_SIZE,
            statistical_family_size=FAMILY_SIZE,
            bootstrap_resamples=10000,
            contrast_direction="ablation minus complete policy",
        )
    write_once(output / "GATE_independent_terms.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--mechanisms", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--spec-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.parent is None or args.mechanisms is None:
            parser.error("prepare requires --parent and --mechanisms")
        result = prepare(args.root, args.parent, args.mechanisms)
    elif args.mode == "worker":
        if args.spec_sha is None:
            parser.error("worker requires --spec-sha")
        result = worker(args.root, args.index, args.spec_sha)
    else:
        result = analyze(args.root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
