import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import evicastkv_async_constructor_profile as control
import pytest


def test_background_fit_records_only_existing_thread_and_preserves_results(tmp_path):
    profile = control.FitProfile()
    task = profile.wrap(lambda owner, index: index + 1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert list(pool.map(lambda i: task(None, i), range(32))) == list(range(1, 33))
    value = profile.finish(tmp_path, expected=32)
    assert value["calls"] == 32 and value["threads"] == 1
    assert not value["background_thread_CPU_not_attributed"]
    assert value["CPU_service_NOT_additive_to_request_wall"]
    assert value["CUDA_kernel_execution_NOT_measured"]
    assert control.sha(tmp_path / "background-fit.prof") == value["profile_sha256"]
    assert profile.thread != threading.get_ident()


def test_ineligible_baseline_does_not_invent_background_work(tmp_path):
    profile = control.FitProfile()
    assert profile.finish(tmp_path, expected=0) == dict(
        calls=0, threads=0, background_fit_ineligible=True
    )
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="count"):
        profile.finish(tmp_path, expected=32)


def test_cross_thread_rejected_and_exception_disables_profiler(tmp_path):
    profile = control.FitProfile()

    def failure(owner):
        raise ValueError("original task failure")

    wrapped = profile.wrap(failure)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match="original task failure"):
            pool.submit(wrapped, None).result()
        assert pool.submit(sys.getprofile).result() is None
    with pytest.raises(RuntimeError, match="private fit thread"):
        wrapped(None)
    assert profile.calls == 1
    assert profile.finish(tmp_path, expected=1)["calls"] == 1


def test_runtime_unchanged_and_profiles_bound_to_three_existing_arms():
    assert control.base.ARMS[:3] == control.ARMS
    assert len(control.APPROVED) == 4
    assert all("/src/" not in p for p in control.APPROVED)
    assert control.PREVIOUS.name.endswith("launch-retry1")


def test_complete_launcher_wires_worker_and_real_configuration_without_jobs():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.argv = ['controller']
import evicastkv_async_constructor_profile as c
def dispatch():
    engine, history = c.base.base, c.base.base.base
    assert engine.control.worker is history.worker
    assert history.ORIGINAL_WORKER is c.worker
    assert engine.control.analyze is c.analyze
    assert engine.control.prepare is c.prepare
    assert engine.ARMS == c.ARMS
    records = {
        'mosaickv_exact': {'graph-lt': {'packet_sha256': 'source'}},
        'prefixkv_reimpl': {a: {'packet_sha256': a} for a in ('graph', 'graph-original')},
    }
    configured = engine.control.configurations(records)
    assert tuple(configured) == c.ARMS
    record = configured['mosaickv_exact', 'joint-async-handoff']
    assert record['verified_constructor'] == 'async_handoff'
c.base.prior.OLD_MAIN = dispatch
c.main()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
