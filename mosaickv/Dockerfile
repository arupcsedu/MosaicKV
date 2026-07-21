FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG MOSAICKV_GIT_SHA=unknown
LABEL org.opencontainers.image.title="MosaicKV common runtime"
LABEL org.opencontainers.image.revision="${MOSAICKV_GIT_SHA}"
LABEL org.mosaickv.python="3.11"
LABEL org.mosaickv.cuda.runtime="12.4"
LABEL org.mosaickv.cuda.minimum-driver="525"
LABEL org.mosaickv.backends="hf,vllm,sglang"
LABEL org.mosaickv.support-status="unverified-until-clean-gpu-smoke"

ENV PYTHONNOUSERSITE=1 \
    PIP_CONFIG_FILE=/dev/null \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv \
    PIP_CACHE_DIR=/scratch/djy8hg/cache/mosaickv/pip \
    UV_CACHE_DIR=/scratch/djy8hg/cache/mosaickv/uv \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=10 \
    UV_CONCURRENT_DOWNLOADS=4 \
    XDG_CACHE_HOME=/scratch/djy8hg/cache/mosaickv/xdg \
    HF_HOME=/scratch/djy8hg/cache/mosaickv/huggingface \
    HF_HUB_CACHE=/scratch/djy8hg/cache/mosaickv/huggingface/hub \
    HF_ASSETS_CACHE=/scratch/djy8hg/cache/mosaickv/huggingface/assets \
    HF_DATASETS_CACHE=/scratch/djy8hg/cache/mosaickv/datasets \
    TRANSFORMERS_CACHE=/scratch/djy8hg/cache/mosaickv/transformers \
    TORCH_HOME=/scratch/djy8hg/cache/mosaickv/torch \
    TORCHINDUCTOR_CACHE_DIR=/scratch/djy8hg/cache/mosaickv/torchinductor \
    TRITON_CACHE_DIR=/scratch/djy8hg/cache/mosaickv/triton \
    NUMBA_CACHE_DIR=/scratch/djy8hg/cache/mosaickv/numba \
    CUDA_CACHE_PATH=/scratch/djy8hg/cache/mosaickv/cuda \
    FLASHINFER_WORKSPACE_BASE=/scratch/djy8hg/cache/mosaickv/flashinfer \
    VLLM_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv/vllm \
    SGLANG_CACHE_DIR=/scratch/djy8hg/cache/mosaickv/sglang \
    PRE_COMMIT_HOME=/scratch/djy8hg/cache/mosaickv/pre-commit \
    MPLCONFIGDIR=/scratch/djy8hg/cache/mosaickv/matplotlib \
    WANDB_CACHE_DIR=/scratch/djy8hg/cache/mosaickv/wandb/cache \
    WANDB_DATA_DIR=/scratch/djy8hg/cache/mosaickv/wandb/data \
    RAY_TMPDIR=/scratch/djy8hg/cache/mosaickv/ray \
    TMPDIR=/scratch/djy8hg/cache/mosaickv/tmp \
    PATH=/opt/mosaickv-venv/bin:${PATH}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       build-essential ffmpeg git libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/mosaickv-venv \
    && mkdir -p /scratch/djy8hg/cache/mosaickv/tmp

WORKDIR /opt/MosaicKV
COPY mosaickv/env/common/requirements.lock /tmp/mosaickv-requirements.lock
RUN python -m pip install 'uv==0.11.29' \
    && uv pip sync --python /opt/mosaickv-venv/bin/python \
       --cache-dir /scratch/djy8hg/cache/mosaickv/uv \
       /tmp/mosaickv-requirements.lock

COPY mosaickv /opt/MosaicKV/mosaickv
WORKDIR /opt/MosaicKV/mosaickv
RUN uv pip install --python /opt/mosaickv-venv/bin/python \
      --cache-dir /scratch/djy8hg/cache/mosaickv/uv --no-deps --editable . \
    && uv pip check --python /opt/mosaickv-venv/bin/python \
    && useradd --create-home --uid 10001 mosaickv \
    && chown -R mosaickv:mosaickv /scratch/djy8hg/cache/mosaickv /opt/MosaicKV

VOLUME ["/scratch/djy8hg/cache/mosaickv"]
USER mosaickv
CMD ["bash", "scripts/docker_smoke.sh"]
