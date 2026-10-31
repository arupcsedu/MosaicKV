import inspect
from types import SimpleNamespace

import evicastkv_funded_window_diagnostic as diagnostic
import torch


def test_protected_window_uses_actual_text_and_preserves_full_marker(monkeypatch):
    import mosaickv.evaluation.funded_text_completion as completion

    monkeypatch.setattr(completion, "generation_delimiter", lambda *args: (18, 19))
    prepared = SimpleNamespace(
        model_inputs={"input_ids": torch.ones(1, 20, dtype=torch.long)},
        modality_map=[
            SimpleNamespace(modality="text", start=0, end=2),
            SimpleNamespace(modality="image", start=2, end=6),
            SimpleNamespace(modality="text", start=6, end=20),
        ],
    )
    assert diagnostic.protected_window(prepared, None) == tuple(range(6, 20))


def test_only_ten_new_rows_and_same_algorithms_no_scope_sweep():
    prepare = inspect.getsource(diagnostic.prepare)
    assert "reused_rows_expected=20" in prepare and "new_inference_rows_expected=10" in prepare
    assert 'get("representation") in ("anchor", "source")' in prepare
    assert "protected_prompt_window=16" in prepare
    assert "last16_actual_text_positions" in prepare
    assert len(diagnostic.VARIANTS) == 15
    worker = inspect.getsource(diagnostic.worker)
    assert "delimiter = protected_window(prepared, adapter.processor.tokenizer)" in worker
    assert "only_storage_precision_and_protection_changes=False" in worker
    assert 'decode_wall_seconds=storage["decode_wall_seconds"]' in worker
