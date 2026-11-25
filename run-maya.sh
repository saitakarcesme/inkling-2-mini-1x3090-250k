#!/usr/bin/env bash
# Maya ik2 OpenAI/Anthropic-compatible server on one card.
# usage: run-maya.sh <gpu> <config.json in configs/> <port>
set -euo pipefail
gpu=${1:?gpu}; cfg=${2:?config}; port=${3:?port}
cache=${INKLING_CACHE:-$HOME/.cache/inkling-250k}
here=$(cd "$(dirname "$0")" && pwd)
exec docker run --rm --init --name "ink-maya-$gpu" --network host --gpus "\"device=$gpu\"" --ipc=host \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$cache/inkling-q4xl.gguf:/maya/models/Inkling-2-Mini-UD-Q4_K_XL.gguf:ro" \
  -v "$cache/mtp.safetensors:/maya/models/mtp.safetensors:ro" \
  -v "$cache/maya-pack:/maya/ik2:ro" \
  -v "$here/configs:/maya/cfg:ro" \
  inkling-maya:27daec5 /opt/venv/bin/python serve/server.py --engine maya --config "/maya/cfg/$cfg" \
  --host "${BIND:-127.0.0.1}" --port "$port"
