# LOOK-M isolated patches

`0001-portable-llava-import.patch` removes two author-machine absolute Python
paths and replaces the LLaVA worker path with the checkout-relative path or
the `LOOKM_LLAVA_PATH` environment variable. It changes no scoring, selection,
merging, model, tokenization, or generation behavior.

Audited SHA-256:
`25daea5b4ed3b51f990ed14106e8400d447060438a2bdf6d196bd3a609d355e4`.

Never apply it in the pinned submodule. Use a disposable checkout:

```bash
git clone --no-local third_party/LOOK-M /scratch/djy8hg/workdir/lookm-official-run
git -C /scratch/djy8hg/workdir/lookm-official-run checkout \
  ecf0f51a9c416c2d85e47faf2638502f01a6d748
git -C /scratch/djy8hg/workdir/lookm-official-run apply \
  "$PWD/third_party/patches/LOOK-M/0001-portable-llava-import.patch"
```

Record both the upstream SHA and SHA-256 of every applied patch in the run
manifest. Instrumentation for selected-position export must be a separate,
reviewed patch; it has not been invented or silently inserted into the
official algorithm here.
