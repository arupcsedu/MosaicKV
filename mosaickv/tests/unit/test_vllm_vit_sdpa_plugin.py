from __future__ import annotations

from mosaickv.backends.vllm_native_plugin import _patch_vit_attention_to_sdpa


def test_vit_sdpa_patch_changes_only_test_platform_class() -> None:
    class FakePlatform:
        @classmethod
        def get_vit_attn_backend(cls, support_fa: bool = False) -> str:
            del cls, support_fa
            return "XFORMERS"

    platform = FakePlatform()

    _patch_vit_attention_to_sdpa(platform, "TORCH_SDPA")
    _patch_vit_attention_to_sdpa(platform, "SHOULD_NOT_REPLACE")

    assert platform.get_vit_attn_backend() == "TORCH_SDPA"
    assert platform.get_vit_attn_backend(support_fa=True) == "TORCH_SDPA"
