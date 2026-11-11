import evicastkv_allocator_transport_probe as probe
import pytest


def test_no_model_or_campaign_claims_and_single_bounded_gpu_request(tmp_path):
    policy = probe.POLICY
    assert policy["GPU_requests"] == 1
    assert policy["worker_A100s"] == 1
    assert policy["worker_CPUs"] == 12
    assert policy["worker_minutes"] <= 120
    assert not policy["exact_source_KV_peak_certified"]
    assert not policy["independent_quality_or_superiority_established"]
    assert not policy["paper_campaign_authorized"]
    cmd = probe.command(tmp_path, "probe", "worker")
    assert "--gres=gpu:a100:1" in cmd
    assert "--reservation=bi_fox_dgx" in cmd
    assert "--time=00:10:00" in cmd
    assert not any("array" in value for value in cmd)
    cpu = probe.command(tmp_path, "cpu", "tests")
    assert not any("--gres" in value or "reservation" in value for value in cpu)


@pytest.mark.parametrize(
    "queue, count",
    [
        ("", 0),
        ("N/A\n(null)\n", 0),
        ("gres/gpu:a100:2\ngres/gpu:1\n", 3),
        ("gpu:a100:4\ngpu:v100:1\n", 5),
        ("license:1,gres/gpu:a100:1", 1),
    ],
)
def test_concurrency_parser_is_conservative(queue, count):
    assert probe.conservative_gpu_count(queue) == count


@pytest.mark.parametrize("value", ["gpu:a100", "gpu:a100:two", "gpu:x:a100:2"])
def test_concurrency_parser_rejects_ambiguous_gpu_counts(value):
    with pytest.raises(ValueError, match="allocation syntax"):
        probe.conservative_gpu_count(value)


def test_unregistered_mode_is_rejected_before_submission(tmp_path):
    with pytest.raises(ValueError, match="unregistered"):
        probe.command(tmp_path, "bad", "paper-campaign")


def test_snapshot_keeps_requested_bytes_pool_identity_and_pending_storage():
    class Cuda:
        @staticmethod
        def memory_snapshot():
            return [
                dict(
                    device=0,
                    address=1000,
                    segment_pool_id=(7, 9),
                    blocks=[
                        dict(size=512, requested_size=3, state="active_allocated"),
                        dict(size=512, requested_size=7, state="active_awaiting_free"),
                        dict(size=512, requested_size=0, state="inactive"),
                    ],
                )
            ]

    class Torch:
        cuda = Cuda()

    assert probe.snapshot_blocks(Torch()) == [
        dict(device=0, addr=1000, size=3, pool=[7, 9], state="active_allocated"),
        dict(device=0, addr=1512, size=7, pool=[7, 9], state="active_awaiting_free"),
    ]


def test_failed_operator_keeps_native_capture_without_promoting_it(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from mosaickv.measurements import native_allocator_events

    monkeypatch.setattr(probe, "load", lambda root: dict(source_sha="synthetic"))
    monkeypatch.setattr(probe.shared, "cpu_gate", lambda *args: {})

    def fail():
        raise ValueError("pending-free control failed")

    monkeypatch.setattr(probe, "run_operator", fail)
    monkeypatch.setattr(
        native_allocator_events,
        "load_recorder",
        lambda: SimpleNamespace(
            stop=lambda: dict(events=[], complete=True),
            build_provenance=dict(testing=True),
        ),
    )
    result = probe.worker(tmp_path)
    assert result["status"] == "failed"
    assert "pending-free" in result["errors"][0]
    assert result["failed_capture_sha256"] == probe.sha(tmp_path / "failed_capture.json")
    saved = probe.read(tmp_path / "failed_capture.json")
    assert saved["failed_control_preserved"] and not saved["capability_pass"]


def test_failed_CPU_gate_emits_terminal_report_without_releasing_work(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "load", lambda root: dict(source_sha="synthetic"))

    def failed_gate(*args):
        raise ValueError("full suite failed")

    def forbidden_submit(*args, **kwargs):
        raise AssertionError("failed CPU gate must not release work")

    monkeypatch.setattr(probe.shared, "cpu_gate", failed_gate)
    monkeypatch.setattr(probe, "submit", forbidden_submit)
    result = probe.after_cpu(tmp_path)
    assert result["status"] == "finite_diagnostics_incomplete_research_gates_still_blocked"
    assert not result["paper_campaign_authorized"]
    assert "CPU-gate.json" in result["controller_errors"]
    assert (tmp_path / "FINAL_diagnostics.json").exists()
