FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    curl \
    git \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY docker_assets/Miniconda3-py312_25.7.0-2-Linux-x86_64.sh /tmp/miniconda.sh

RUN bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh

RUN python --version

SHELL ["/bin/bash", "-lc"]

COPY docker_assets/pip_wheels /tmp/pip_wheels

RUN pip install --no-cache-dir /tmp/pip_wheels/* \
    && rm -rf /tmp/pip_wheels

COPY docker_assets/wheels /tmp/wheels

RUN pip install --no-cache-dir /tmp/wheels/* \
    && rm -rf /tmp/wheels

RUN pip install --no-cache-dir numpy==2.1.3 pandas==2.2.3

WORKDIR /workspace

CMD python -c "import torch, numpy, scipy, matplotlib; \
print('torch=', torch.__version__); \
print('cuda available=', torch.cuda.is_available()); \
print('device count=', torch.cuda.device_count()); \
print('numpy=', numpy.__version__); \
print('scipy=', scipy.__version__); \
print('matplotlib=', matplotlib.__version__)"