#!/usr/bin/env bash
# One-time setup: pinned Maya engine image, pinned weights (sha256-checked), tokenizer pack.
# Needs: Linux, Docker with the NVIDIA Container Toolkit, ~40 GB free disk, curl.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cache=${INKLING_CACHE:-$HOME/.cache/inkling-250k}
MAYA_COMMIT=27daec5e   # alesha-pro/maya-inkling-mini, 2026-11-23
mkdir -p "$cache"
if [ ! -d "$cache/maya/.git" ]; then git clone https://github.com/alesha-pro/maya-inkling-mini.git "$cache/maya"; fi
git -C "$cache/maya" fetch -q origin && git -C "$cache/maya" checkout -q "$MAYA_COMMIT"
docker build --network host -f "$here/docker/Dockerfile" -t inkling-maya:27daec5 "$cache/maya"

[ -f "$cache/inkling-q4xl.gguf" ] || curl -fL --retry 3 -o "$cache/inkling-q4xl.gguf" \
  https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/45fd08764f2b0bf16b03a5483a80e5c7aaec8772/Inkling-2-Mini-UD-Q4_K_XL.gguf
[ -f "$cache/mtp.safetensors" ] || curl -fL --retry 3 -o "$cache/mtp.safetensors" \
  https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/b85c3cfa615fc413cc9af3379f491b963f3b586a/mtp.safetensors
(cd "$cache" && sha256sum -c <<'HASHES'
e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15  inkling-q4xl.gguf
7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2  mtp.safetensors
HASHES
)
mkdir -p "$cache/maya-pack"
docker run --rm --network none -u "$(id -u):$(id -g)" -v "$cache:/w" inkling-maya:27daec5 \
  /opt/venv/bin/python tools/strata_tokenizer.py --gguf /w/inkling-q4xl.gguf --out /w/maya-pack \
  --chat-template tools/ik2/chat_template.jinja --check
echo "ready: $cache"
