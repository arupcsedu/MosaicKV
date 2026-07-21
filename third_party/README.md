# Third-party source registry

External source is preserved at an immutable revision and is not imported into
MosaicKV's primary Python environment.

| Component | Upstream | Pin | License | Local use |
|---|---|---|---|---|
| LOOK-M | <https://github.com/SUSTechBruce/LOOK-M> | `ecf0f51a9c416c2d85e47faf2638502f01a6d748` | MIT, copyright 2024 Bruce.wan | Official baseline source inspection and isolated execution only |
| PrefixKV | <https://github.com/THU-MIG/PrefixKV> | `597f1ab032704951550f93bcc8a23f1454b80aa4` | MIT, copyright 2024 Zuyan Liu | Official baseline source inspection, profiles, and isolated execution only |

Initialize and verify the exact checkout with:

```bash
git submodule update --init --recursive third_party/LOOK-M
test "$(git -C third_party/LOOK-M rev-parse HEAD)" = \
  ecf0f51a9c416c2d85e47faf2638502f01a6d748
git -C third_party/LOOK-M status --porcelain

git submodule update --init --recursive third_party/PrefixKV
test "$(git -C third_party/PrefixKV rev-parse HEAD)" = \
  597f1ab032704951550f93bcc8a23f1454b80aa4
git -C third_party/PrefixKV status --porcelain
```

The submodule remained unmodified during the controlled LLaVA-1.5-7B parity
run. `mosaickv/scripts/run_prefixkv_llava_parity.py` imports the pinned
`PrefixKV` class and its eager-attention patch directly, and writes all
generated profiles and measurements under `/scratch`. See
`mosaickv/docs/baselines/prefixkv_parity_report.md` for the exact non-canonical
run and artifact hashes.

The last command must be empty. Do not edit the submodule for an experiment.
Put any necessary change in the component's `third_party/patches/` directory, apply it to a
disposable checkout, and record the patch digest in the run manifest. A
patched execution remains an official-source execution only when the patch is
limited to portability or measurement instrumentation and does not change the
algorithm; otherwise label it as a reimplementation.
