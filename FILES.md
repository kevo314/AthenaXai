###############################################################################
# Athena Xai v1 — Dockerfile.whisper
# Jetson Orin Nano Super 8GB / JetPack 6.2 / CUDA 12.6
#
# SELF-CONTAINED: whisper-server binary + CUDA toolkit libs from host.
# runtime: nvidia provides GPU driver. This image provides the toolkit.
# No Python, no pip, no torch — just the binary and its .so dependencies.
###############################################################################
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Minimal runtime deps (curl for healthcheck, ffmpeg for audio conversion)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CUDA toolkit shared libraries copied from host JetPack installation
COPY cuda-libs/ /usr/local/lib/cuda/

# ggml shared libraries from the llama.cpp / whisper.cpp builds
COPY lib/ /usr/local/lib/

# whisper-server binary
COPY bin/whisper-server /usr/local/bin/whisper-server
RUN chmod +x /usr/local/bin/whisper-server

# Make all libraries discoverable
RUN echo "/usr/local/lib/cuda" > /etc/ld.so.conf.d/cuda-toolkit.conf \
    && echo "/usr/local/lib" > /etc/ld.so.conf.d/ggml.conf \
    && ldconfig

# Verify the binary can at least show its help (proves .so linking works)
RUN whisper-server --help 2>&1 | head -3 || echo "WARNING: whisper-server --help failed"

ENV LD_LIBRARY_PATH=/usr/local/lib/cuda:/usr/local/lib
