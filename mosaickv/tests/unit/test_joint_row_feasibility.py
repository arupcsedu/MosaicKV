import pytest
import torch

from mosaickv.evaluation.joint_row_feasibility import WINDOWS, admissible, comparison, geometry


@pytest.mark.parametrize(
    "length,first,tail", [(600, 512, 88), (604, 512, 92), (1, 0, 1), (127, 0, 127)]
)
def test_source_and_one_incoming_row_fit_the_existing_bounded_workspace(length, first, tail):
    assert geometry(length) == dict(first=first, prompt_rows=tail, joint_rows=tail + 1, width=4096)


@pytest.mark.parametrize("length", [0, -1, True, 128, 256])
def test_nonfeasible_geometry_is_not_silently_grown(length):
    with pytest.raises((ValueError, NotImplementedError)):
        geometry(length)


def test_numerically_equal_signed_zeros_are_still_a_failed_word_gate():
    left = torch.tensor([-0.0], dtype=torch.bfloat16)
    right = torch.tensor([0.0], dtype=torch.bfloat16)
    row = comparison(left, right)
    assert row["changed_words"] == 1 and row["maximum_absolute_difference"] == 0
    assert not row["bitwise_equal"]


def test_every_window_and_both_components_must_match_exactly():
    rows = [
        dict(
            step=s,
            component=c,
            **{
                backend: dict(
                    incoming_row=dict(bitwise_equal=True), prompt_rows=dict(bitwise_equal=True)
                )
                for backend in ("torch_joint", "Lt_joint")
            },
        )
        for s in WINDOWS
        for c in (0, 1)
    ]
    assert all(admissible(rows).values())
    rows[-1]["Lt_joint"]["incoming_row"]["bitwise_equal"] = False
    assert not admissible(rows)["Lt_joint"]
    with pytest.raises(ValueError, match="every"):
        admissible(rows[:-1])


def test_hooks_observe_but_never_replace_model_outputs_and_restore_on_failure(monkeypatch):
    from types import SimpleNamespace

    from mosaickv.evaluation.joint_row_feasibility import Probe, current

    attention = SimpleNamespace(k_proj=torch.nn.Linear(4, 4), v_proj=torch.nn.Linear(4, 4))
    hidden = torch.zeros(1, 1, 4)

    def step():
        return attention.k_proj(hidden), attention.v_proj(hidden)

    adapter = SimpleNamespace(
        _language_layers=lambda: [SimpleNamespace(self_attn=attention)], decode_one_token=step
    )
    expected = step()
    state = Probe(adapter)
    seen = []
    monkeypatch.setattr(state, "observe", lambda component, module, x, y: seen.append(component))
    with pytest.raises(RuntimeError, match="stop"), state.installed():
        assert current() is state
        actual = adapter.decode_one_token()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        raise RuntimeError("stop")
    assert adapter.decode_one_token is step and current() is None
    assert seen == [0, 1]
    assert not attention.k_proj._forward_hooks and not attention.v_proj._forward_hooks
