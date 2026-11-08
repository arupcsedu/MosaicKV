from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.history_source_precision import HistoryFormat
from mosaickv.evaluation.legacy_dispatch_fingerprint import verify
from mosaickv.evaluation.sequential_kv_lease import fill_component


def test_explicit_history_dispatch_reaches_checked_method_and_other_formats_stay_rejected(
    monkeypatch,
):
    calls = []
    packet = HistoryFormat(None, None, None, (1, 32, 16, 128))
    target = torch.empty(1, 32, 16, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(
        HistoryFormat, "fill_component", lambda self, a, c, t: calls.append((self, a, c, t))
    )
    fill_component(packet, "attention", 1, target)
    assert calls[0][0] is packet and calls[0][1:3] == ("attention", 1)
    assert calls[0][3] is target
    with pytest.raises(NotImplementedError, match="unaudited format"):
        fill_component(SimpleNamespace(shape=target.shape), None, 1, target)
    with pytest.raises(ValueError):
        fill_component(packet, None, 2, target)


def test_legacy_fingerprint_rejects_any_unrelated_change():
    old = "def fill_component(packet, attention, component, destination):\n    return 7\n"
    branch = (
        "    from mosaickv.evaluation.history_source_precision import HistoryFormat\n"
        "    if isinstance(packet, HistoryFormat):\n"
        "        packet.fill_component(attention, component, destination)\n"
        "        return\n"
    )
    new = old.replace("    return 7", branch + "    return 7")
    verify(old, new)
    with pytest.raises(ValueError):
        verify(old, new.replace("return 7", "return 8"))
