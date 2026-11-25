#!/usr/bin/env bash
# Baseline (step 0) as measured: Unsloth llama.cpp b12588-mix-c0c3de8 CUDA 13 bundle unpacked in $cache/fast,
# run inside any image that carries the CUDA 13 runtime (LLAMA_IMAGE). ub 64 is the largest that fits 250K + q8_0 KV, same UD-Q4_K_XL file as Maya, 250K window.
set -euo pipefail
gpu=${1:?gpu}; ub=${2:-64}; b=${3:-64}; kv=${4:-q8_0}
cache=${INKLING_CACHE:-$HOME/.cache/inkling-250k}
exec docker run --rm --init --name "ink-llamacpp-gpu$gpu" --network host --gpus "device=$gpu" \
  -u "$(id -u):$(id -g)" -e LD_LIBRARY_PATH=/bench/fast:/opt/venv/lib/python3.13/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64 \
  -v "$cache:/bench:ro" --entrypoint /bench/fast/llama-server "${LLAMA_IMAGE:?an image with the CUDA 13 runtime; see README}" \
  -m /bench/inkling-q4xl.gguf --alias inkling-2-mini -ngl all -np 1 -c 250000 \
  -ctk "$kv" -ctv "$kv" -fa on -b "$b" -ub "$ub" --port $((18100 + gpu)) --host 127.0.0.1 --timeout 86400 --jinja --no-warmup
