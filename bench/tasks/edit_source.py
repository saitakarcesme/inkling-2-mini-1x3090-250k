"""Send the same prompts to one local model, one at a time, and record outputs and speed.

usage: run_race.py <glm|qwen> <out_dir>
Streams each answer so time-to-first-token and decode speed are measured per request.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

PROFILES = {
    # GLM-5.3-Flash 2.05bpw EXL3, faithful profile. Model-card sampling, thinking on, a larger thinking budget than the
    # server default (6144) because these are open-ended creative builds.
    "glm": {
        "url": "http://127.0.0.1:30270/v1/chat/completions",
        "key": "local",
        "model": "glm-5.3-flash",
        "extra": {"temperature": 1.0, "top_p": 0.95, "thinking_budget": 10240},
        "max_tokens": 32768,
    },
    # Qwen3.8-27B EXL3 on Elpis: greedy only (sampling is rejected with HTTP 400), thinking on, high effort.
    "qwen": {
        "url": "http://127.0.0.1:11445/v1/chat/completions",
        "key": "local",
        "model": "qwen3.8:27b",
        "extra": {"temperature": 0, "reasoning_effort": "high"},
        "max_tokens": 32768,
    },
    "qwen_med": {
        "url": "http://127.0.0.1:11445/v1/chat/completions",
        "key": "local",
        "model": "qwen3.8:27b",
        "extra": {"temperature": 0, "reasoning_effort": "medium"},
        "max_tokens": 32768,
    },
}

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(profile, prompt_text):
    body = {"model": profile["model"], "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": profile["max_tokens"], "stream": True,
            "stream_options": {"include_usage": True}, **profile["extra"]}
    req = urllib.request.Request(profile["url"], json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + profile["key"]})
    start = time.time()
    first = first_answer = None
    reasoning, content, usage, finish = [], [], None, None
    with urllib.request.urlopen(req, timeout=7200) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                now = time.time()
                if delta.get("reasoning_content"):
                    first = first or now
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    first = first or now
                    first_answer = first_answer or now
                    content.append(delta["content"])
                finish = choice.get("finish_reason") or finish
    end = time.time()
    text = "".join(content)
    # Some servers leave the think block inline instead of in reasoning_content.
    if "</think>" in text and not reasoning:
        head, _, text = text.partition("</think>")
        reasoning = [head.replace("<think>", "")]
    completion = (usage or {}).get("completion_tokens")
    decode_start = first or start
    return {
        "wall_s": round(end - start, 1),
        "ttft_s": round((first or end) - start, 2),
        "answer_start_s": round((first_answer or end) - start, 1),
        "completion_tokens": completion,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "decode_tok_s": round(completion / max(end - decode_start, 1e-6), 1) if completion else None,
        "e2e_tok_s": round(completion / (end - start), 1) if completion else None,
        "finish": finish,
        "reasoning_chars": len("".join(reasoning)),
        "answer_chars": len(text),
        "reasoning": "".join(reasoning),
        "content": text,
    }


def extract_html(text):
    blocks = re.findall(r"```(?:html|HTML)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    start = text.find("<!DOCTYPE")
    if start < 0:
        start = text.lower().find("<html")
    return text[start:] if start >= 0 else None


def hardware():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,power.limit,clocks.max.sm", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return smi


def main():
    name, out_dir = sys.argv[1], sys.argv[2]
    profile = PROFILES[name]
    os.makedirs(out_dir, exist_ok=True)
    spec = json.load(open(os.path.join(HERE, "prompts.json")))
    meta = {"profile": name, "model": profile["model"], "sampling": profile["extra"], "max_tokens": profile["max_tokens"],
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "gpus": hardware()}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    for item in spec["prompts"]:
        path = os.path.join(out_dir, item["id"] + ".json")
        if os.path.exists(path):
            continue
        print(time.strftime("%H:%M:%S"), "start", item["id"], flush=True)
        try:
            result = run_one(profile, item["prompt"] + "\n\n" + spec["suffix_html"])
        except Exception as error:  # keep the race going; record the failure
            result = {"error": repr(error)}
        result.update({"id": item["id"], "title": item["title"]})
        html = extract_html(result.get("content", ""))
        if html:
            open(os.path.join(out_dir, item["id"] + ".html"), "w").write(html)
        result["html_chars"] = len(html) if html else 0
        json.dump(result, open(path, "w"), indent=1)
        print(time.strftime("%H:%M:%S"), "done", item["id"],
              {k: result.get(k) for k in ("wall_s", "completion_tokens", "decode_tok_s", "finish", "html_chars")}, flush=True)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
