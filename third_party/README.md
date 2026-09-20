# Third-party source registry

External source is preserved at an immutable revision and is not imported into
MosaicKV's primary Python environment.

| Component | Upstream | Pin | License | Local use |
|---|---|---|---|---|
| LOOK-M | <https://github.com/SUSTechBruce/LOOK-M> | `ecf0f51a9c416c2d85e47faf2638502f01a6d748` | MIT, copyright 2024 Bruce.wan | Official baseline source inspection and isolated execution only |
| PrefixKV | <https://github.com/THU-MIG/PrefixKV> | `597f1ab032704951550f93bcc8a23f1454b80aa4` | MIT, copyright 2024 Zuyan Liu | Official baseline source inspection, profiles, and isolated execution only |
| Elastic Cache | <https://github.com/liuzuyan/ElasticCache> | `3138e03fc8894231d98a2e5353f27a59af854cc3` | MIT, copyright 2024 Zuyan Liu | Official source audit; local code is always `elastic_cache_reimpl` |
| StreamingLLM | <https://github.com/mit-han-lab/streaming-llm> | `2e5042606d69933d88fbf909bd77907456b9b4dd` | MIT | Official source audit; local code is always `streamingllm_reimpl` |
| PyramidKV | <https://github.com/IsaacRe/PyramidKV> | `1ca16d6ea0bc3e48fc4cc8623bc4901bc131d511` | Apache-2.0 | Official source audit; local code is always `pyramidkv_reimpl` |
| SnapKV | <https://github.com/FasterDecoding/SnapKV> | `e216ddc84c5bd210378cbdbbba12ba02102aa640` | Apache-2.0 | Official source audit; local code is always `snapkv_reimpl` |

H2O (`ac75c2a8a9e76832b2a4139b9363373b56336bfb`) and EasyKV/RoCo
(`a1d71cae3b562d9a709dda3741bd63e46a09ad31`) were audited from their public
repositories, but neither audited tree contains a top-level license file.
Their code is therefore not redistributed.  Only clean-room, paper-derived
`h2o_reimpl` and `roco_reimpl` implementations live in MosaicKV.  The complete
machine-readable audit is in `third_party/baseline_sources.yaml`.

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

git submodule update --init --recursive \
  third_party/ElasticCache third_party/StreamingLLM third_party/PyramidKV \
  third_party/SnapKV
test "$(git -C third_party/ElasticCache rev-parse HEAD)" = \
  3138e03fc8894231d98a2e5353f27a59af854cc3
test "$(git -C third_party/StreamingLLM rev-parse HEAD)" = \
  2e5042606d69933d88fbf909bd77907456b9b4dd
test "$(git -C third_party/PyramidKV rev-parse HEAD)" = \
  1ca16d6ea0bc3e48fc4cc8623bc4901bc131d511
test "$(git -C third_party/SnapKV rev-parse HEAD)" = \
  e216ddc84c5bd210378cbdbbba12ba02102aa640
git -C third_party/StreamingLLM status --porcelain
git -C third_party/PyramidKV status --porcelain
git -C third_party/SnapKV status --porcelain
git -C third_party/ElasticCache status --porcelain
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

Official PrefixKV, LOOK-M, and Elastic Cache are launched only through the
fail-closed wrapper:

```bash
source mosaickv/scripts/cache_env.sh
python mosaickv/scripts/run_official_baseline.py prefixkv --check-only \
  --repository-root "$PWD" \
  --python /scratch/djy8hg/env/mosaickv_prefixkv_parity/bin/python \
  --output-directory /scratch/djy8hg/runs/mosaickv/prefixkv/check \
  --model-path /scratch/djy8hg/models/llava-v1.5-7b \
  --data-path /scratch/djy8hg/data/prefixkv/llava_description.json \
  --media-root /scratch/djy8hg/data/prefixkv/images
```

The wrapper verifies the Git SHA, clean source tree, preserved license,
baseline-specific package versions, scratch-only paths, and writes an immutable
official launch manifest. LOOK-M uses the wrapper name `lookm`; Elastic Cache
uses `elastic_cache` and its audited Torch 2.1.2/Transformers 4.31.0 author
stack. A
MosaicKV common-environment row is never labeled official merely because the
reimplementation follows the paper.
