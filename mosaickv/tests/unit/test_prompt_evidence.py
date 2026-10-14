"""CPU synthetic processor-offset mapping, including expanded image placeholders."""

from types import SimpleNamespace

import pytest

from mosaickv.adapters.huggingface.prompt_evidence import align_text_positions, map_prompt_evidence
from mosaickv.cache_state import Modality, ModalitySpan


def test_alignment_after_images_and_boundary_overlapping_subwords():
    assert align_text_positions(
        [1, 99, 20, 21, 22],
        [(0, 0), (0, 7), (7, 11), (11, 14), (14, 20)],
        [1, 99, 99, 99, 99, 20, 21, 22],
        99,
        ((9, 14),),
    ) == (5, 6)
    assert align_text_positions(
        [1, 99, 99, 20],
        [(0, 0), (0, 1), (1, 2), (2, 4)],
        [1, 99, 99, 99, 99, 99, 99, 20],
        99,
        ((2, 4),),
    ) == (7,)
    with pytest.raises(ValueError, match="differ"):
        align_text_positions(
            [1, 99, 20], [(0, 0), (0, 7), (7, 10)], [1, 99, 99, 21], 99, ((7, 10),)
        )
    with pytest.raises(ValueError, match="unmapped"):
        align_text_positions([1, 20], [(0, 0), (0, 2)], [1, 20], 99, ((4, 6),))


class Tokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        assert kwargs == {"return_offsets_mapping": True, "add_special_tokens": True}
        return {
            "input_ids": [1] + [ord(c) + 100 for c in text],
            "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))],
        }


class IDs:
    def __init__(self, ids):
        self.ids = ids

    def tolist(self):
        return self.ids


class Adapter:
    capabilities = SimpleNamespace(model_family="llava_1_5")
    processor = SimpleNamespace(tokenizer=Tokenizer())

    def _normalize_messages(self, messages, media):
        return messages, [], []

    def _image_token_id(self):
        return 99

    def _render_prompt(self, chat):
        return (
            "\n".join(m["role"] + ": " + "".join(p["text"] for p in m["content"]) for m in chat)
            + "\nASSISTANT:"
        )


def fixture():
    adapter = Adapter()
    chat = [
        dict(role="system", content=[dict(type="text", text="Repeated question")]),
        dict(
            role="user", content=[dict(type="text", text="Repeated question; answer briefly. café")]
        ),
    ]
    prompt = adapter._render_prompt(chat)
    ids = adapter.processor.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)[
        "input_ids"
    ]
    prepared = SimpleNamespace(
        model_inputs={"input_ids": [IDs(ids)]},
        logical_sequence_length=len(ids),
        modality_map=(ModalitySpan(0, len(ids), Modality.TEXT),),
    )
    return adapter, chat, prepared


def test_mapping_uses_role_template_not_first_matching_question_string():
    adapter, chat, prepared = fixture()
    mapped = map_prompt_evidence(adapter, prepared, chat)
    prompt = adapter._render_prompt(chat)
    start = prompt.index("user: ") + len("user: ")
    assert mapped.positions == tuple(
        range(start + 1, start + 1 + len(chat[1]["content"][0]["text"]))
    )
    assert len(mapped.trace()["mapping_sha256"]) == 64
    assert mapped == map_prompt_evidence(adapter, prepared, chat)


def test_mapping_fails_closed_on_unknown_models_multiturn_or_wrong_ids():
    adapter, chat, prepared = fixture()
    adapter.capabilities = SimpleNamespace(model_family="qwen2_5_vl")
    with pytest.raises(ValueError, match="LLaVA"):
        map_prompt_evidence(adapter, prepared, chat)
    adapter.capabilities = SimpleNamespace(model_family="llava_1_5")
    with pytest.raises(ValueError, match="one user"):
        map_prompt_evidence(adapter, prepared, chat + chat[-1:])
    prepared.model_inputs["input_ids"][0].ids[-1] += 1
    with pytest.raises(ValueError, match="differ"):
        map_prompt_evidence(adapter, prepared, chat)
