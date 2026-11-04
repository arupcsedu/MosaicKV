import evicastkv_deferred_transport_latency as controller
import pytest


def test_text_and_whole_readers_require_their_matching_reference():
    frozen = {"prefixkv_reimpl": dict(packet_sha256="text")}
    controller.require_matching_packet(dict(digest="text"), "prefixkv_reimpl", {}, frozen)
    controller.require_matching_packet(
        dict(digest="whole"), "prefixkv_reimpl", dict(whole_policy={}), frozen, "whole"
    )
    with pytest.raises(RuntimeError, match="matching immediate"):
        controller.require_matching_packet(
            dict(digest="whole"), "prefixkv_reimpl", dict(whole_policy={}), frozen
        )


def test_wrong_reference_does_not_silently_pass():
    frozen = {"prefixkv_reimpl": dict(packet_sha256="text")}
    for signature, record, whole in (
        ("text", dict(whole_policy={}), "whole"),
        ("whole", {}, None),
        ("arbitrary", {}, None),
    ):
        with pytest.raises(RuntimeError, match="packet words"):
            controller.require_matching_packet(
                dict(digest=signature), "prefixkv_reimpl", record, frozen, whole
            )


@pytest.mark.parametrize("complete,passed", [(False, False), (True, True)])
def test_source_followup_cannot_skip_current_terminal_gate(tmp_path, monkeypatch, complete, passed):
    def read(path):
        if path.name == "plan.json":
            return {"bound_files": {}}
        if "construction-bookkeeping-consumed" in str(path):
            return dict(complete=complete, all_input_latency_gate=passed)
        return dict(complete=True, all_input_latency_gate=False)

    monkeypatch.setattr(controller, "clean_source", lambda *_: "immutable")
    monkeypatch.setattr(controller.driver, "read", read)
    monkeypatch.setattr(controller, "sha", lambda *_: "digest")
    monkeypatch.setattr(controller.direct, "submit", lambda *_: pytest.fail("must not submit"))
    root = tmp_path / "not-authorized"
    with pytest.raises(ValueError, match="bookkeeping strict negative"):
        controller.prepare(root, "immutable")
    assert not root.exists()
