FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /Radar_Vital_Signs
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .
CMD ["python", "-m", "src.inference.dar_pipeline", "--help"]
