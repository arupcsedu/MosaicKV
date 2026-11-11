"""Closed native allocator snapshot-state vocabulary.

PyTorch 2.8's pinned C++ serializer emits ``active_pending_free``; its Python
memory.py documentation still uses ``active_awaiting_free``. Both spellings
describe backing that must remain live until a completed-free event.
"""

PENDING_STATES = frozenset({"active_pending_free", "active_awaiting_free"})
ACTIVE_STATES = PENDING_STATES | {"active_allocated"}
KNOWN_STATES = ACTIVE_STATES | {"inactive"}


def is_active_state(state):
    if not isinstance(state, str) or state not in KNOWN_STATES:
        raise ValueError(f"unreviewed allocator snapshot state: {state!r}")
    return state in ACTIVE_STATES


def is_pending_state(state):
    is_active_state(state)
    return state in PENDING_STATES
