# Inkling 2 Mini on one RTX 3090, 250K context

A measured serving recipe for the original [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines, 26B looped MoE) on **one 24 GB RTX 3090** with a **250,000-token window**. It starts from a stock llama.cpp server and adds one change at a time. Each step is measured on the same card with the same weight file and the same real workloads.

| On one RTX 3090, 250K window | llama.cpp (start) | Maya engine | Maya + DSpark drafter (this recipe) |
|---|---:|---:|---:|
| Writing new code, tok/s | 52.5 | 130.9 | **203.4** |
| Editing a 2.4K-token file, tok/s | 51.8 | 128.4 | **254.7** |
| Turkish prose, tok/s | 52.6 | **131.9** | 116.2 |
| Reasoning, thinking on, tok/s | 52.5 | 130.0 | **157.8** |
| 245K tokens of real code: wait for first token | 941 s | **173 s** | 174 s |
| 245K tokens of real code: decode, tok/s | 29.3 | 75.1 | **80.2** |
| Next turn in the same 245K conversation: first token | 3.3 s | **0.2 s** | **0.2 s** |
| GSM8K, first 100, greedy | 97 | 95 | 95 |

Bold marks the best value in each row. The drafter makes free prose about 12% slower; everything else gets faster or stays the same. The table below shows how much each step adds. The GSM8K differences are within run-to-run noise: llama.cpp kept thinking even with `reasoning_effort: none` (see *Findings*), and Maya answered directly.

## What runs

* **Weights:** Unsloth `UD-Q4_K_XL` GGUF ([`45fd087`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/45fd08764f2b0bf16b03a5483a80e5c7aaec8772)). The experts of layers 0–19 keep Thinking Machines' own NVFP4 bits; Maya decodes them in registers on SM86. The official checkpoint itself (NVFP4/MXFP8) needs SM90 in vLLM: the `vllm/vllm-openai:inkling-2-mini` image stops on a 3090 at its compute-capability check, before loading weights.
* **Drafter:** Thinking Machines' DSpark `mtp.safetensors` ([`b85c3cf`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a)). It is lossless: the model verifies every drafted token.
* **Engine:** [`maya-ik2`](https://github.com/alesha-pro/maya-inkling-mini) at `27daec5`. All 3,072 experts, the dense weights, the FP8 KV cache for 250K and the drafter fit in VRAM: 23.35 GiB in use.
* **API:** OpenAI chat/responses and Anthropic messages on one port, with streaming, tool calls and `reasoning_effort` (`none` … `max`). Pi, OpenCode, Claude-compatible clients and T3 Code connect directly.

## Quick start

```bash
git clone https://github.com/saitakarcesme/inkling-2-mini-1x3090-250k && cd inkling-2-mini-1x3090-250k
./setup.sh                                  # pinned engine image, sha256-checked weights, tokenizer (~40 GB disk)
./run-maya.sh 0 maya-dspark.json 18200      # GPU 0, port 18200
curl -s http://127.0.0.1:18200/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini", "reasoning_effort": "none", "max_tokens": 200,
  "messages": [{"role": "user", "content": "Write a haiku about a lighthouse."}]}'
```

The engine needs the whole card. If a desktop or another program uses ~0.8 GB on it, the startup log says `resident path off ... decode runs at about half speed`. Use `maya-dspark.json` with `"--chunk","1024"` added to `args`, as the [2×3090 recipe](https://github.com/saitakarcesme/inkling-2-mini-2x3090-250k) does for its desktop card, or close that program. `BIND=0.0.0.0` exposes the port, for example on a tailnet.

## Step by step

All steps use one RTX 3090 (GPU1 of this machine, 420 W limit, driver 615.78.08), the same `UD-Q4_K_XL` file and a 250,000-token window. Each step was run once through the HTTP API, the way an agent calls it. The first request after every start was discarded.

| Step | Change | New code | Edit | 245K first token | 245K decode | What it buys |
|---|---|---:|---:|---:|---:|---|
| 0 | llama.cpp `b12588` (Unsloth CUDA 13 build), q8_0 KV, `-ub 64` | 52.5 | 51.8 | 941 s | 29.3 | starting point; `-ub 128` and up do not fit 250K |
| 1 | Maya `maya-ik2`, FP8 KV, every expert resident, one CUDA graph per token | 130.9 | 128.4 | 173 s | 75.1 | **2.5× decode, 5.4× prefill** |
| 2 | + DSpark drafter (`--draft mtp.safetensors`, drafter window 64K) | 203.4 | 254.7 | 174 s | 80.2 | **+55% new code, +98% edits**; prose −12% |

Tried and not kept, each measured against step 2 at temperature 1.0 ([`results/sw-*.json`](results/)):

| Setting | New code | Edit | Prose | Verdict |
|---|---:|---:|---:|---|
| `IK2_DRAFT_WIDTH=6` | 192.8 | 280.3 | 114.4 | edits +10%, new code −5%: within sampling noise, default kept |
| `IK2_DRAFT_WIDTH=4` | 185.9 | 217.6 | 116.2 | slower on edits |
| `IK2_DRAFT_PROBE=16` | 181.9 | 254.5 | 120.9 | no clear gain |
| `--chunk 2048` | | | | 245K first token 172.7 s vs 173.8 s: the automatic chunk already does this |

Sampling at temperature 1.0 moves a single run by about ±5%. Treat a difference smaller than that as noise.

## The workloads

These were fixed before the first measurement. There are no repeated-token or synthetic prompts. A suffix drafter scores near-perfectly on repeated tokens, which tells you nothing about real use.

* **Short tasks** ([`bench/inkbench.py`](bench/inkbench.py)): a new Python CLI tool (1,536 tokens), adding a `--retries` option to a real 146-line Python file and returning it whole (2,048), a ~500-word Turkish explainer (1,024), and a scheduling word problem with thinking on at `medium` (answer 3,957; every step got it right).
* **Long context** ([`bench/build_corpus.py`](bench/build_corpus.py)): real C++/CUDA/Python source (the Maya tree at `27daec5`), cut on file boundaries to 64K, 128K and 245K tokens. The question asks about a function defined only in the *first* file, so the answer comes from the far end of the window. A unique first line keeps each request cold. Both Maya steps found the file and the function at every size. llama.cpp found them at 128K and 245K; at 64K it used up its 1,536 tokens thinking and returned no answer.
* **Agent follow-up:** after the 245K answer, a second question in the same conversation. This measures prefix reuse.
* **Tool call:** a coding-agent system prompt with `read_file` / `run_shell` tools. Every step produced a valid call.
* **Quality guard:** GSM8K test questions 1–100, greedy, `reasoning_effort: none`.

Raw per-request results, with server timings and drafter counters, are in [`results/`](results/). Reproduce one row:

```bash
python3 bench/build_corpus.py "$INKLING_CACHE/maya/src" <tokenizer.json> bench/corpus
python3 bench/inkbench.py http://127.0.0.1:18200/v1 my-run results
python3 bench/report.py results s0-llamacpp-ub64 s1-maya-plain s2-maya-dspark my-run
```

## Findings

* **llama.cpp ignores `reasoning_effort: none` for this model.** The template's "effort 0" still produced long thinking: the 64K answer spent all 1,536 tokens thinking and returned no visible text. A zero `reasoning_budget_tokens` made it loop on empty think blocks. Maya opens the answer segment directly.
* **Decode slows with depth for a physical reason.** 8 of the 48 layer visits attend over the full prefix, so every token reads 16 KB of FP8 cache per token of context: 130 tok/s near empty, 110 at 64K, 95 at 128K, 75–80 at 245K.
* **The drafter barely helps deep in the context.** The drafter only sees the last 65,536 positions (`IK2_DRAFT_CTX`). A full-context drafter needs about 1 GiB more, which one card does not have at 250K.
* **Power limit matters for prefill.** The same recipe on the second card (350 W, with 0.8 GB used by the desktop and therefore `--chunk 1024`) took 210 s for the 245K first token, against 174 s at 420 W. Decode differed by under 7%.

## How this was approached

Two rules came from earlier work on this machine: [Elpis](https://github.com/gildrb/elpis) (`gildrb/elpis`) and [HyperQwen](https://github.com/syv-ai/HyperQwen) (`syv-ai/HyperQwen`). First, fix the quality check and the workloads before measuring, and keep a speedup only if the check still passes. Second, report copy-heavy work (edits) separately from free text, because speculation flatters it. HyperQwen's practice of discarding the first request after a start and stating the power limit with every number is followed here too.

## Credits

The model and the DSpark drafter are by [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini), the GGUF by [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), and the engine is [maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini) by alesha-pro and the [Project Maya](https://github.com/mw00/project-maya) contributors. This repo adds no kernels. It contains the measured configuration, the scripts and the benchmark. Weights are Apache-2.0; the scripts here are MIT.
