"""Pure synthetic capsules; no scheduler, Git, historical files or artifact I/O."""

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from mosaickv.evaluation import freeze_writer_bindings_v54 as freeze


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _main(*, robust=False):
    default = "v86" if robust else "v74"
    sorting = ", sort_keys=True" if robust else ""
    return f'''def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frozen-output", required=True, type=Path)
    parser.add_argument("--run-prefix", default="{default}")
    args = parser.parse_args()
    result = analyze(
        args.spec.resolve(), args.output_root.resolve(), args.frozen_output.resolve(),
        run_prefix=str(args.run_prefix),
    )
    print(json.dumps(
        {{"selected_candidate": result["selected_candidate"], "labels_accessed": False}}
        {sorting}
    ))
    return 0 if result["selected_candidate"] is not None else 2

if __name__ == "__main__":
    raise SystemExit(main())
'''


def capsule(version=76):
    """Build deliberately synthetic source, never execute any writer code."""
    job = str(18570000 + version)
    root = f"/scratch/djy8hg/runs/example-v{version}"
    protocol = f"mosaickv/configs/experiments/example_v{version}.yaml"
    wrapper_path = f"mosaickv/slurm/choice_c_v{version}_freeze.sbatch"
    entrypoint = freeze.ROBUST_WRITER if version == 88 else freeze.BASE_WRITER
    wrapper = """#!/bin/bash
#SBATCH --job-name=synthetic-pure-test

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V@V@_OUTPUT_ROOT:-@ROOT@}
SPEC=${MOSAICKV_V@V@_SPEC:-${PACKAGE_ROOT}/@PROTOCOL@}
FROZEN=${MOSAICKV_V@V@_FROZEN:-${OUTPUT_ROOT}/analysis/frozen_candidate.yaml}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
"""
    command = """"${PYTHON}" @ENTRY@ \\
  --spec "${SPEC}" \\
  --output-root "${OUTPUT_ROOT}" \\
  --frozen-output "${FROZEN}" \\
  --run-prefix v@V@
"""
    if version == 88:
        wrapper += (
            """
set +e
"""
            + command
            + """ANALYSIS_STATUS=$?
set -e

if [[ "${ANALYSIS_STATUS}" -eq 2 && ! -e "${FROZEN}" ]]; then
  printf 'V88 completed as a negative calibration: no candidate passed every unchanged gate.\\n'
  exit 0
fi
exit "${ANALYSIS_STATUS}"
"""
        )
    else:
        wrapper += command
    for key, value in {
        "@V@": str(version),
        "@ROOT@": root,
        "@PROTOCOL@": protocol.removeprefix("mosaickv/"),
        "@ENTRY@": entrypoint.removeprefix("mosaickv/"),
    }.items():
        wrapper = wrapper.replace(key, value)
    base = """def analyze(spec_path, output_root, frozen_output, *, run_prefix="v74"):
    analysis_root = output_root / "analysis"
    pd.DataFrame(candidate_ranking).to_parquet(
        analysis_root / "label_blind_candidate_ranking.parquet", index=False,
    )
    return result
""" + _main()
    sources = {
        freeze.BASE_WRITER: base.encode(),
        "mosaickv/scripts/assert_clean_worktree.sh": b"synthetic guard; never executed\n",
        "mosaickv/scripts/cache_env.sh": b"synthetic cache; never executed\n",
        "mosaickv/scripts/choice_c_v74_protocol.py": b"# synthetic protocol helper\n",
        "mosaickv/scripts/run_choice_c_gate.py": b"# synthetic sha helper\n",
        "mosaickv/scripts/run_choice_c_v72_heldout_quality_memory_worker.py": b"# resolve\n",
        protocol: b"synthetic: true\n",
        wrapper_path: wrapper.encode(),
    }
    if version == 88:
        robust = """from analyze_choice_c_v74_label_blind_calibration import analyze as analyze_base
def analyze(spec_path, output_root, frozen_output, *, run_prefix="v86"):
    analysis_root = output_root / "analysis"
    with never_executed():
        base = analyze_base(spec_path, output_root, provisional_path, run_prefix=run_prefix)
    pd.DataFrame(robust_ranking).to_parquet(
        analysis_root / "robust_fidelity_candidate_ranking.parquet", index=False,
    )
    return result
""" + _main(robust=True)
        sources[freeze.ROBUST_WRITER] = robust.encode()
    state, code = ("COMPLETED", "0:0") if version == 88 else ("FAILED", "2:0")
    flags = "--nodes=2 --ntasks=2 " if version in (78, 81) else ""
    accounting = (
        f"{job}|djy8hg|{state}|{code}|/scratch/djy8hg/workdir/MosaicKV/mosaickv|"
        f"sbatch --parsable {flags}--dependency=afterok:12345 --export=ALL,"
        f"MOSAICKV_V{version}_OUTPUT_ROOT={root},"
        f"MOSAICKV_V{version}_SPEC=/scratch/djy8hg/workdir/MosaicKV/{protocol},"
        f"MOSAICKV_V{version}_FROZEN={root}/analysis/frozen_candidate.yaml "
        f"{wrapper_path.removeprefix('mosaickv/')}\n"
    ).encode()
    startup = f"clean_git_sha={'a' * 40}\n".encode()
    batch = (f"Batch Script for {job}\n" + "-" * 80 + "\n").encode() + wrapper.encode()
    outputs = [
        dict(
            path=root + "/analysis/label_blind_candidate_ranking.parquet",
            artifact_sha256="d" * 64,
            writer_path=freeze.BASE_WRITER,
            writer_sha256=_sha(sources[freeze.BASE_WRITER]),
        )
    ]
    if version == 88:
        outputs.append(
            dict(
                path=root + "/analysis/robust_fidelity_candidate_ranking.parquet",
                artifact_sha256="e" * 64,
                writer_path=freeze.ROBUST_WRITER,
                writer_sha256=_sha(sources[freeze.ROBUST_WRITER]),
            )
        )
    spec = dict(
        version=version,
        job_id=job,
        source_sha="a" * 40,
        repository_tree_oid="b" * 40,
        seal_path=f"mosaickv/configs/ledgers/example_v{version}.yaml",
        seal_sha256="f" * 64,
        seal_commit="c" * 40,
        seal_blob_oid="d" * 40,
        protocol_path=protocol,
        protocol_sha256=_sha(sources[protocol]),
        run_root=root,
        frozen_output=root + "/analysis/frozen_candidate.yaml",
        expected_state=state,
        expected_exit_code=code,
        accounting_sha256=_sha(accounting),
        startup_sha256=_sha(startup),
        batch_record_sha256=_sha(batch),
        wrapper_path=wrapper_path,
        wrapper_sha256=_sha(sources[wrapper_path]),
        entrypoint_path=entrypoint,
        source_blob_sha256s={p: _sha(b) for p, b in sources.items()},
        outputs=outputs,
    )
    seal = dict(
        classification=freeze.CLASSIFICATION,
        run_root=root,
        freeze_job=int(job),
        protocol_path=protocol.removeprefix("mosaickv/"),
        protocol_sha256=spec["protocol_sha256"],
        frozen_candidate_created=False,
    )
    if version == 88:
        seal["freeze_state"] = "completed_failed_closed_no_eligible_candidate"
    return [accounting, startup, batch, seal, spec, sources, spec["repository_tree_oid"]]


def approve(monkeypatch, args):
    """Only synthetic tests replace the production registry authorization."""
    spec = args[4]
    monkeypatch.setitem(freeze.REGISTERED_SPEC_SHA256, spec["version"], freeze._digest(spec))


def repin(monkeypatch, args):
    spec, sources = args[4:6]
    for index, field in (
        (0, "accounting_sha256"),
        (1, "startup_sha256"),
        (2, "batch_record_sha256"),
    ):
        spec[field] = _sha(args[index])
    spec["source_blob_sha256s"] = {path: _sha(raw) for path, raw in sources.items()}
    spec["wrapper_sha256"] = _sha(sources[spec["wrapper_path"]])
    approve(monkeypatch, args)


@pytest.mark.parametrize("version", [76, 78, 79, 80, 81, 88])
def test_six_pure_guarded_negative_capsules_and_seven_distinct_writer_links(monkeypatch, version):
    args = capsule(version)
    approve(monkeypatch, args)
    before = deepcopy(args)
    facts = freeze.validate_freeze_runtime(*args)
    assert args == before
    assert facts["negative_result_preserved"] is True
    assert facts["quality_gate_pass"] is False
    assert facts["scientific_dependency_validation"] is False
    assert facts["canonical_eligible"] is False
    assert facts["outputs"][0]["writer_path"] == freeze.BASE_WRITER
    assert facts["expected_exit_code"] == ("0:0" if version == 88 else "2:0")
    if version == 88:
        assert facts["entrypoint_path"] == freeze.ROBUST_WRITER
        assert facts["outputs"][1]["writer_path"] == freeze.ROBUST_WRITER
    assert not any(word in str(facts) for word in ("sbatch ", "PYTHONPATH", "pipefail"))
    monkeypatch.setitem(freeze.REGISTERED_FACTS_SHA256, version, facts["runtime_evidence_sha256"])
    assert freeze.validate_runtime_facts(facts) == facts


def test_validator_and_fact_projection_perform_no_io(monkeypatch):
    args = capsule()
    approve(monkeypatch, args)

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure freeze validator attempted I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    for name in ("open", "stat", "lstat", "resolve", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, name, forbidden)
    facts = freeze.validate_freeze_runtime(*args)
    monkeypatch.setitem(freeze.REGISTERED_FACTS_SHA256, 76, facts["runtime_evidence_sha256"])
    assert freeze.validate_runtime_facts(facts) == facts


def test_unregistered_self_rehashed_specification_never_authorizes_a_writer():
    with pytest.raises(ValueError, match="six pinned"):
        freeze.validate_freeze_runtime(*capsule())


def test_registered_spec_preflight_returns_independent_data(monkeypatch):
    args = capsule()
    approve(monkeypatch, args)
    result = freeze.validate_registered_spec(args[4])
    assert result == args[4]
    result["source_blob_sha256s"].clear()
    assert args[4]["source_blob_sha256s"]


@pytest.mark.parametrize("version", [True, False, "76", 77, None, []])
def test_preflight_rejects_unregistered_or_boolean_version_types(version):
    spec = capsule()[4]
    spec["version"] = version
    with pytest.raises(ValueError, match="six pinned"):
        freeze.validate_registered_spec(spec)


@pytest.mark.parametrize(
    "field",
    [
        "accounting_sha256",
        "startup_sha256",
        "source_sha",
        "wrapper_sha256",
        "entrypoint_path",
        "seal_sha256",
    ],
)
def test_pinned_spec_cannot_be_changed_after_authorization(monkeypatch, field):
    args = capsule()
    approve(monkeypatch, args)
    args[4][field] = "0" * 64
    with pytest.raises(ValueError, match="six pinned"):
        freeze.validate_freeze_runtime(*args)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_raw_accounting_prefix_and_batch_hashes_are_independent(monkeypatch, index):
    args = capsule()
    approve(monkeypatch, args)
    args[index] = args[index].replace(b"a", b"b", 1) if index == 1 else args[index] + b"x"
    with pytest.raises(ValueError):
        freeze.validate_freeze_runtime(*args)


@pytest.mark.parametrize(
    "mutation",
    [
        "job",
        "user",
        "state",
        "exit",
        "workdir",
        "multiple",
        "truncated",
        "fields",
        "command",
        "unknown_option",
        "duplicate_option",
        "dependency",
        "export_key",
        "duplicate_export",
        "output_root",
        "protocol",
        "frozen",
        "wrapper",
        "shell",
    ],
)
def test_even_rehashed_accounting_must_match_exact_job_arguments_and_exports(monkeypatch, mutation):
    args = capsule()
    old = args[0]
    replacements = {
        "job": (b"18570076|", b"18570077|"),
        "user": (b"|djy8hg|", b"|someone|"),
        "state": (b"|FAILED|", b"|COMPLETED|"),
        "exit": (b"|2:0|", b"|1:0|"),
        "workdir": (b"|/scratch/djy8hg/workdir/MosaicKV/mosaickv|", b"|/scratch/djy8hg/other|"),
        "command": (b"sbatch ", b"echo "),
        "unknown_option": (b"--parsable ", b"--parsable --wrap=anything "),
        "duplicate_option": (b"--parsable ", b"--parsable --parsable "),
        "dependency": (b"afterok:12345", b"afterany:12345"),
        "export_key": (b"_FROZEN=", b"_UNKNOWN="),
        "duplicate_export": (b"_FROZEN=", b"_SPEC="),
        "output_root": (
            b"OUTPUT_ROOT=/scratch/djy8hg/runs/example-v76",
            b"OUTPUT_ROOT=/scratch/djy8hg/runs/other",
        ),
        "protocol": (b"example_v76.yaml", b"example_v78.yaml"),
        "frozen": (b"frozen_candidate.yaml", b"another.yaml"),
        "wrapper": (b"slurm/choice_c_v76_freeze.sbatch", b"slurm/choice_c_v78_freeze.sbatch"),
        "shell": (b"sbatch ", b"sbatch ; "),
    }
    if mutation == "multiple":
        args[0] += old
    elif mutation == "truncated":
        args[0] = old[:-1]
    elif mutation == "fields":
        args[0] = b"18570076|djy8hg|FAILED|2:0\n"
    else:
        args[0] = old.replace(*replacements[mutation], 1)
    repin(monkeypatch, args)
    with pytest.raises(ValueError):
        freeze.validate_freeze_runtime(*args)


def test_only_registered_scratch_aliases_and_original_protocol_relative_path_are_accepted(
    monkeypatch,
):
    args = capsule(78)
    args[0] = args[0].replace(b"/scratch/djy8hg/", b"/sfs/weka/scratch/djy8hg/")
    args[3]["run_root"] = "/sfs/weka" + args[3]["run_root"]
    args[3]["protocol_path"] = args[4]["protocol_path"]
    repin(monkeypatch, args)
    result = freeze.validate_freeze_runtime(*args)
    assert result["run_root"].startswith("/sfs/weka/scratch/")


@pytest.mark.parametrize("mutation", ["dirty", "source", "truncated", "extra", "space"])
def test_rehashed_startup_still_requires_exact_55_byte_clean_source(monkeypatch, mutation):
    args = capsule()
    replacements = {
        "dirty": (b"clean_git_sha", b"dirty_git_sha"),
        "source": (b"a" * 40, b"b" * 40),
        "space": (b"clean_git_sha", b" clean_git_sha"),
    }
    if mutation == "extra":
        args[1] += b"\n"
    elif mutation == "truncated":
        args[1] = args[1][:-1]
    else:
        args[1] = args[1].replace(*replacements[mutation])
    repin(monkeypatch, args)
    with pytest.raises(ValueError):
        freeze.validate_freeze_runtime(*args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("freeze_job", True),
        ("freeze_job", "99999"),
        ("classification", "positive"),
        ("frozen_candidate_created", 0),
        ("frozen_candidate_created", True),
        ("run_root", "/scratch/djy8hg/runs/another"),
        ("protocol_path", "configs/other.yaml"),
        ("protocol_sha256", "f" * 64),
        ("extra", False),
    ],
)
def test_seal_metadata_does_not_supply_unchecked_assertion_booleans(monkeypatch, field, value):
    args = capsule()
    approve(monkeypatch, args)
    args[3][field] = value
    with pytest.raises(ValueError):
        freeze.validate_freeze_runtime(*args)


def test_v88_zero_exit_must_remain_explicitly_negative(monkeypatch):
    args = capsule(88)
    approve(monkeypatch, args)
    args[3]["freeze_state"] = "completed_positive"
    with pytest.raises(ValueError, match="negative seal"):
        freeze.validate_freeze_runtime(*args)


@pytest.mark.parametrize("mutation", ["job", "header", "body"])
def test_rehashed_batch_record_requires_exact_job_header_and_source_body(monkeypatch, mutation):
    args = capsule()
    if mutation == "job":
        args[2] = args[2].replace(b"18570076", b"18570077", 1)
    elif mutation == "header":
        args[2] = args[2].replace(b"-" * 80, b"-" * 79, 1)
    else:
        args[2] += b"# later edit\n"
    repin(monkeypatch, args)
    with pytest.raises(ValueError, match="batch header/body"):
        freeze.validate_freeze_runtime(*args)


@pytest.mark.parametrize(
    "old,new",
    [
        (b'"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"', b"# guard merely mentioned"),
        (b"set -euo pipefail", b"set +e"),
        (b"--run-prefix v76", b"--run-prefix v78"),
        (b'--spec "${SPEC}"', b'--spec "${SPEC}" --spec other'),
        (b'--output-root "${OUTPUT_ROOT}"', b"--output-root other"),
        (b'"${PYTHON}" scripts/', b'"${PYTHON}" other/'),
        (b'"${PACKAGE_ROOT}/scripts/cache_env.sh"', b'"${PACKAGE_ROOT}/scripts/other.sh"'),
    ],
)
def test_even_reapproved_wrapper_must_match_reviewed_guard_and_exact_argv_template(
    monkeypatch, old, new
):
    args = capsule()
    path = args[4]["wrapper_path"]
    args[5][path] = args[5][path].replace(old, new, 1)
    header = (f"Batch Script for {args[4]['job_id']}\n" + "-" * 80 + "\n").encode()
    args[2] = header + args[5][path]
    repin(monkeypatch, args)
    with pytest.raises(ValueError, match="wrapper template"):
        freeze.validate_freeze_runtime(*args)


def test_v88_status_two_requires_absent_frozen_path_before_mapping_to_zero(monkeypatch):
    args = capsule(88)
    path = args[4]["wrapper_path"]
    args[5][path] = args[5][path].replace(b' && ! -e "${FROZEN}"', b"")
    args[2] = args[2][: args[2].index(b"#!/bin/bash")] + args[5][path]
    repin(monkeypatch, args)
    with pytest.raises(ValueError, match="wrapper template"):
        freeze.validate_freeze_runtime(*args)


def test_v88_robust_entrypoint_requires_explicit_base_writer_call(monkeypatch):
    args = capsule(88)
    args[5][freeze.ROBUST_WRITER] = args[5][freeze.ROBUST_WRITER].replace(
        b"base = analyze_base(", b"base = different_writer(", 1
    )
    repin(monkeypatch, args)
    with pytest.raises(ValueError, match="original base writer"):
        freeze.validate_freeze_runtime(*args)


@pytest.mark.parametrize(
    "old,new",
    [
        (b"else 2", b"else 0"),
        (
            b'parser.add_argument("--spec", required=True',
            b'parser.add_argument("--spec", required=False',
        ),
        (b'"label_blind_candidate_ranking.parquet"', b'"wrong.parquet"'),
        (b"raise SystemExit(main())", b"raise SystemExit(2)"),
        (
            b"    pd.DataFrame(candidate_ranking).to_parquet",
            b"    if selected:\n        pd.DataFrame(candidate_ranking).to_parquet",
        ),
    ],
)
def test_reviewed_writer_ast_must_keep_unconditional_target_write_and_negative_main(
    monkeypatch, old, new
):
    args = capsule()
    args[5][freeze.BASE_WRITER] = args[5][freeze.BASE_WRITER].replace(old, new, 1)
    repin(monkeypatch, args)
    with pytest.raises(ValueError):
        freeze.validate_freeze_runtime(*args)


@pytest.mark.parametrize("failure", ["tree", "missing_source", "extra_source", "source_bytes"])
def test_critical_committed_source_fieldmap_and_tree_are_required(monkeypatch, failure):
    args = capsule()
    approve(monkeypatch, args)
    if failure == "tree":
        args[6] = "f" * 40
    elif failure == "missing_source":
        del args[5][freeze.BASE_WRITER]
    elif failure == "extra_source":
        args[5]["mosaickv/scripts/another.py"] = b"# no\n"
    else:
        args[5][freeze.BASE_WRITER] += b"# modified\n"
    with pytest.raises(ValueError):
        freeze.validate_freeze_runtime(*args)


@pytest.mark.parametrize(
    "value", ["../x", "//scratch/djy8hg/x", "/home/private", "/scratch/djy8hg/.codex/auth.json"]
)
def test_unregistered_paths_are_rejected_without_resolving(value):
    with pytest.raises(ValueError):
        freeze._path(value)


def test_bounds_remain_finite_and_do_not_allow_oversized_source(monkeypatch):
    args = capsule()
    args[5][freeze.BASE_WRITER] = b"x" * freeze.MAXIMUM_LIVE_BYTES
    repin(monkeypatch, args)
    with pytest.raises(ValueError, match="live critical-source byte"):
        freeze.validate_freeze_runtime(*args)
    with pytest.raises(ValueError, match="node limit"):
        freeze._digest([None] * freeze.MAXIMUM_NODES)
    with pytest.raises(ValueError, match="scalar byte"):
        freeze._digest("x" * (freeze.MAXIMUM_SCALAR_BYTES + 1))
    nested = None
    for _ in range(freeze.MAXIMUM_DEPTH + 1):
        nested = [nested]
    with pytest.raises(ValueError, match="depth limit"):
        freeze._digest(nested)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha", "f" * 40),
        ("job_id", "99999"),
        ("quality_gate_pass", True),
        ("negative_result_preserved", False),
        ("scientific_dependency_validation", True),
        ("canonical_eligible", True),
        ("extra", False),
    ],
)
def test_self_rehashed_runtime_facts_cannot_broaden_the_pinned_projection(
    monkeypatch, field, value
):
    args = capsule()
    approve(monkeypatch, args)
    facts = freeze.validate_freeze_runtime(*args)
    monkeypatch.setitem(freeze.REGISTERED_FACTS_SHA256, 76, facts["runtime_evidence_sha256"])
    facts[field] = value
    facts["runtime_evidence_sha256"] = freeze._digest(
        {key: val for key, val in facts.items() if key != "runtime_evidence_sha256"}
    )
    with pytest.raises(ValueError, match="pinned projections"):
        freeze.validate_runtime_facts(facts)
