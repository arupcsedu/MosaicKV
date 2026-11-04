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
