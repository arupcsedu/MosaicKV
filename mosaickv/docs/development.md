# Development environment

The audited development environment is isolated at
`/scratch/djy8hg/env/mosaickv_dev`. It was created from Python 3.11.15 without
system site packages. The runtime dependency is NumPy; PyTorch, Transformers,
vLLM, SGLang, FlashAttention, and lmms-eval are optional diagnostics and are not
installed or imported during ordinary smoke tests.

From the MosaicKV repository root:

```bash
/scratch/djy8hg/env/mosaickv_dev/bin/python -m pip install -e 'mosaickv[dev]'
/scratch/djy8hg/env/mosaickv_dev/bin/mosaickv doctor
/scratch/djy8hg/env/mosaickv_dev/bin/mosaickv smoke
cd mosaickv
/scratch/djy8hg/env/mosaickv_dev/bin/python -m pytest
PYTHON_BIN=/scratch/djy8hg/env/mosaickv_dev/bin/python ./scripts/check.sh
```

`evaluate` and `benchmark` are strict configuration preflights. They return
`status: not_run` until the corresponding research milestones and correctness
gates are implemented; they do not emit fabricated measurements.
