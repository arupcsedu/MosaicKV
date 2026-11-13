"""Synthetic original scheduler records; no scheduler or historical file I/O."""

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from mosaickv.evaluation import parquet_startup_bindings_v50 as startup


def fixture(*, quoted=False, stage="validation"):
    source = "c" * 40
    output = "/scratch/djy8hg/runs/example/analysis"
    body = (
        "set -euo pipefail; cd /scratch/djy8hg/workdir/source; "
        "source mosaickv/scripts/cache_env.sh; "
        'export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"; '
        "bash mosaickv/scripts/assert_clean_worktree.sh; "
        "/scratch/djy8hg/env/mosaickv/bin/python "
        "mosaickv/scripts/analyze_milebench_shared_bytes.py "
        "--root /scratch/djy8hg/runs/example --plan-sha "
        + "a" * 64
        + f" --stage {stage} --output {output}"
    )
    wrap = repr(body) if quoted else body
    accounting = (
        "21671354|djy8hg|COMPLETED|0:0|/scratch/djy8hg/workdir/source|"
        f"sbatch --cpus-per-task=2 --wrap={wrap}\n"
    ).encode()
    record = f"clean_git_sha={source}\n".encode()
    spec = dict(
        job_id="21671354",
        accounting_sha256=hashlib.sha256(accounting).hexdigest(),
        startup_sha256=hashlib.sha256(record).hexdigest(),
        source_sha=source,
        output_path=output,
        stage=stage,
    )
    return accounting, record, spec


def rehash(accounting, record, spec):
    return dict(
        spec,
        accounting_sha256=hashlib.sha256(accounting).hexdigest(),
        startup_sha256=hashlib.sha256(record).hexdigest(),
    )


@pytest.mark.parametrize("quoted", [False, True])
@pytest.mark.parametrize("stage", ["validation", "larger"])
def test_original_unquoted_or_quoted_sacct_wrap_is_valid(quoted, stage):
    args = fixture(quoted=quoted, stage=stage)
    original = deepcopy(args)
    result = startup.validate_startup(*args)
    assert args == original
    assert result == dict(args[2], output_path="/sfs/weka/scratch/djy8hg/runs/example/analysis")
    assert set(result) == startup.SPEC_FIELDS
    assert not any(word in str(result) for word in ("sbatch", "PYTHONPATH", "pipefail"))


def test_scratch_aliases_match_without_touching_the_filesystem(monkeypatch):
    accounting, record, spec = fixture()
    spec["output_path"] = "/sfs/weka/scratch/djy8hg/runs/example/analysis"

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure validator performed I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    for name in ("open", "stat", "lstat", "resolve", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, name, forbidden)
    assert startup.validate_startup(accounting, record, spec)["output_path"] == spec["output_path"]


@pytest.mark.parametrize(
    "failure",
    ["job", "user", "state", "exit", "fields", "multi_line", "truncated_line", "unknown_command"],
)
def test_scheduler_identity_completion_and_record_structure_are_exact(failure):
    accounting, record, spec = fixture()
    if failure == "job":
        accounting = accounting.replace(b"21671354|", b"21671355|", 1)
    elif failure == "user":
        accounting = accounting.replace(b"|djy8hg|", b"|someone|", 1)
    elif failure == "state":
        accounting = accounting.replace(b"|COMPLETED|", b"|FAILED|", 1)
    elif failure == "exit":
        accounting = accounting.replace(b"|0:0|", b"|1:0|", 1)
    elif failure == "fields":
        accounting = b"21671354|djy8hg|COMPLETED|0:0\n"
    elif failure == "multi_line":
        accounting += accounting
    elif failure == "truncated_line":
        accounting = accounting[:-1]
    else:
        accounting = accounting.replace(b"sbatch ", b"echo ", 1)
    with pytest.raises(ValueError):
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))


@pytest.mark.parametrize(
    "failure", ["source", "prefix", "no_newline", "extra_line", "extra_space", "short_sha"]
)
def test_startup_is_exactly_the_55_byte_clean_source_record(failure):
    accounting, record, spec = fixture()
    if failure == "source":
        record = record.replace(b"c" * 40, b"d" * 40)
    elif failure == "prefix":
        record = record.replace(b"clean_git_sha=", b"dirty_git_sha=")
    elif failure == "no_newline":
        record = record[:-1]
    elif failure == "extra_line":
        record += b"extra\n"
    elif failure == "extra_space":
        record = b" " + record
    else:
        record = record[:-2] + b"\n"
    with pytest.raises(ValueError, match="exact clean source"):
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))


@pytest.mark.parametrize("key", ["accounting_sha256", "startup_sha256"])
def test_every_raw_hash_is_bound_to_the_static_specification(key):
    accounting, record, spec = fixture()
    spec[key] = "f" * 64
    with pytest.raises(ValueError, match="hash differs"):
        startup.validate_startup(accounting, record, spec)


@pytest.mark.parametrize(
    "old,new",
    [
        (b"--stage validation", b"--stage larger"),
        (b"--stage validation", b"--stage unknown"),
        (b"--stage validation", b""),
        (b"--stage validation", b"--stage validation --stage larger"),
        (
            b"--output /scratch/djy8hg/runs/example/analysis",
            b"--output /scratch/djy8hg/runs/other/analysis",
        ),
        (b"--output /scratch/djy8hg/runs/example/analysis", b"--output"),
        (
            b"--output /scratch/djy8hg/runs/example/analysis",
            b"--output /scratch/djy8hg/runs/example/analysis --output /scratch/djy8hg/runs/other",
        ),
        (b"assert_clean_worktree.sh", b"not_the_guard.sh"),
        (b"analyze_milebench_shared_bytes.py", b"another_analysis.py"),
        (
            b"bash mosaickv/scripts/assert_clean_worktree.sh",
            b"echo mosaickv/scripts/assert_clean_worktree.sh",
        ),
        (
            b"/scratch/djy8hg/env/mosaickv/bin/python "
            b"mosaickv/scripts/analyze_milebench_shared_bytes.py",
            b"echo mosaickv/scripts/analyze_milebench_shared_bytes.py",
        ),
    ],
)
def test_target_stage_guard_and_actual_analyzer_invocation_are_required(old, new):
    accounting, record, spec = fixture()
    assert old in accounting
    accounting = accounting.replace(old, new)
    with pytest.raises(ValueError):
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))


def test_guard_after_the_analyzer_is_not_a_valid_startup_boundary():
    accounting, record, spec = fixture()
    guard = b"bash mosaickv/scripts/assert_clean_worktree.sh; "
    accounting = accounting.replace(guard, b"").replace(b"\n", b"; " + guard + b"\n")
    with pytest.raises(ValueError, match="precede"):
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))


@pytest.mark.parametrize("injection", [b"| cat", b"|| true", b"&", b"$(id)", b"`id`", b"# comment"])
def test_unreviewed_shell_control_or_substitution_is_rejected(injection):
    accounting, record, spec = fixture()
    accounting = accounting.replace(b"; cd ", b" " + injection + b"; cd ", 1)
    with pytest.raises(ValueError):
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))


@pytest.mark.parametrize("where", ["wrap", "export"])
def test_literal_token_assignment_is_rejected_without_echoing_its_value(where):
    accounting, record, spec = fixture()
    secret = b"SYNTHETIC_PRIVATE_VALUE"
    if where == "wrap":
        accounting = accounting.replace(b"set -euo", b"export HF_TOKEN=" + secret + b"; set -euo")
    else:
        accounting = accounting.replace(b"sbatch ", b"sbatch --export=HF_TOKEN=" + secret + b" ")
    with pytest.raises(ValueError) as caught:
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))
    assert secret.decode() not in str(caught.value)


@pytest.mark.parametrize(
    "path",
    [
        "relative",
        "/scratch/djy8hg/../other",
        "/scratch/djy8hg/.codex/auth.json",
        "/home/djy8hg/runs",
        "/scratch/djy8hg//runs",
        "/scratch/djy8hg/runs\x00",
    ],
)
def test_static_output_cannot_traverse_or_access_home_or_credentials(path):
    accounting, record, spec = fixture()
    spec["output_path"] = path
    with pytest.raises(ValueError):
        startup.validate_startup(accounting, record, spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "1; command"),
        ("source_sha", "HEAD"),
        ("source_sha", "c" * 39),
        ("stage", "unknown"),
        ("accounting_sha256", "short"),
        ("startup_sha256", None),
    ],
)
def test_static_scope_fields_are_strict(field, value):
    accounting, record, spec = fixture()
    spec[field] = value
    with pytest.raises(ValueError):
        startup.validate_startup(accounting, record, spec)


def test_unknown_static_fields_are_rejected():
    accounting, record, spec = fixture()
    spec["skip_guard"] = True
    with pytest.raises(ValueError, match="exact static"):
        startup.validate_startup(accounting, record, spec)


@pytest.mark.parametrize("which", [0, 1])
def test_large_or_empty_records_are_rejected_before_parsing(which):
    args = list(fixture())
    args[which] = b"x" * (startup.MAXIMUM_BYTES + 1)
    with pytest.raises(ValueError, match="bounded"):
        startup.validate_startup(*args)
    args[which] = b""
    with pytest.raises(ValueError, match="bounded"):
        startup.validate_startup(*args)


def test_equals_options_and_extra_registered_recovery_arguments_are_supported():
    accounting, record, spec = fixture()
    accounting = accounting.replace(b"--stage validation", b"--stage=validation")
    accounting = accounting.replace(b"--output /scratch", b"--output=/scratch")
    accounting = accounting.replace(
        b"\n",
        b" --recovery-root /scratch/djy8hg/runs/recovery --recovery-plan-sha " + b"f" * 64 + b"\n",
    )
    assert (
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))["stage"]
        == "validation"
    )


def test_final_exec_prefix_executes_the_exact_registered_analyzer():
    accounting, record, spec = fixture(stage="larger")
    accounting = accounting.replace(
        b"; /scratch/djy8hg/env/mosaickv/bin/python ",
        b"; exec /scratch/djy8hg/env/mosaickv/bin/python ",
    )
    assert (
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))["stage"]
        == "larger"
    )


@pytest.mark.parametrize(
    "wrapper",
    [b"exec echo", b"exec arbitrary_wrapper", b"echo exec", b"exec exec", b"exec -a fake"],
)
def test_exec_does_not_authorize_echo_unknown_wrappers_or_extra_interpreters(wrapper):
    accounting, record, spec = fixture()
    accounting = accounting.replace(
        b"; /scratch/djy8hg/env/mosaickv/bin/python ",
        b"; " + wrapper + b" /scratch/djy8hg/env/mosaickv/bin/python ",
    )
    with pytest.raises(ValueError):
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))


@pytest.mark.parametrize("ending", [b"'", b'"'])
def test_truncated_wrap_quotes_are_rejected(ending):
    accounting, record, spec = fixture()
    accounting = accounting.replace(b"--wrap=", b"--wrap=" + ending)
    with pytest.raises(ValueError, match="quot"):
        startup.validate_startup(accounting, record, rehash(accounting, record, spec))
