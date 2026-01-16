FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bzip2 git \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniforge（conda-forge 生态）
ARG MINIFORGE_VERSION=24.11.3-0
RUN curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh \
    && /opt/conda/bin/conda config --system --set auto_activate_base false \
    && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# 把默认用户 vscode 改名为 sunny（避免 UID/GID=1000 冲突）
RUN set -eux; \
    usermod -l sunny vscode; \
    groupmod -n sunny vscode; \
    usermod -d /home/sunny -m sunny; \
    chown -R sunny:sunny /opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/*

RUN echo "sunny ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/sunny \
    && chmod 0440 /etc/sudoers.d/sunny
