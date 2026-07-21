# SGLang validation artifacts

The JSON files in this directory are controlled, dirty-worktree development
comparisons between SGLang FullKV job `17160103` and HF eager FullKV job
`17160104`. Both preserve `status: failed` because generated token IDs and
decoded text did not match, despite identical manifest input and generation
hashes. They are not paper-eligible result rows and must not be converted into
a parity claim.

SGLang Stage A execution itself passed its backend-specific verifier for the
pinned Qwen2.5-VL-3B checkpoint in job `17160103` and the pinned 7B checkpoint
in job `17160299`. Native MosaicKV remains unsupported; see
[`docs/sglang_native_blocker.md`](../../docs/sglang_native_blocker.md).
