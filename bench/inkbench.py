"""Realistic speed + sanity benchmark for an OpenAI-compatible Inkling 2 Mini server.

usage: inkbench.py <base_url> <label> <out_dir> [--suite short,long,followup,gsm,tools] [--model NAME]
       [--temperature 1.0] [--gsm-n 100] [--long 64k,128k,245k] [--concurrency N]

Everything goes through the HTTP chat API with streaming, the way Pi / T3 / coding agents call it.
No synthetic token streams: the prompts are real tasks, the long inputs are real source code
(frozen by build_corpus.py). Each request records wall time, time to first token, decode tok/s
(completion tokens / time after the first token) and the server's own timings when it reports them.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import statistics
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

EDIT_FILE = open(os.path.join(HERE, "tasks", "edit_source.py")).read()

SHORT = [
    {"id": "code-new", "effort": "none", "max_tokens": 1536,
     "prompt": "Write a complete, production-quality Python 3 command-line tool `dupfind.py` that finds duplicate files "
               "under a directory. Group candidates by size first, then by a fast partial hash of the first 64 KiB, "
               "then confirm with a full SHA-256. Support --min-size, --json output, --delete-newer with a --dry-run "
               "default, and a progress line on stderr. Include type hints and a short usage docstring."},
    {"id": "edit", "effort": "none", "max_tokens": 2048,
     "prompt": "Here is a Python file:\n\n```python\n" + EDIT_FILE + "\n```\n\nAdd a `--retries N` command-line option "
               "(default 2). When a request in run_one raises an exception, retry it up to N times with a 5 second "
               "pause, and record the number of attempts in the result as \"attempts\". Keep everything else exactly "
               "as it is. Return the complete updated file in one ```python block."},
    {"id": "prose-tr", "effort": "none", "max_tokens": 1024,
     "prompt": "Bir lise öğrencisine, büyük dil modellerinde 'KV önbelleği'nin ne olduğunu ve uzun bir sohbette "
               "cevapların neden yavaşladığını anlatan, örnekli ve akıcı bir Türkçe yazı yaz. Yaklaşık 500 kelime olsun."},
    {"id": "reason", "effort": "medium", "max_tokens": 4096,
     "prompt": "A warehouse has three conveyor belts. Belt A moves 12 boxes per minute, belt B 18, belt C 25. "
               "Belt C breaks down for 7 minutes every 30 minutes, starting at minute 10. Belt B only runs while "
               "belt A has run for at least 5 minutes since the last time B stopped, and B stops whenever C is down. "
               "Starting at minute 0 with all belts able to run (B waits for A's first 5 minutes), how many boxes "
               "are moved in the first 90 minutes? Show your reasoning briefly and give the final number on the last "
               "line as 'ANSWER: <n>'."},
]

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a UTF-8 text file from the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command in the workspace and return stdout/stderr.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def post(base, body, timeout=10800):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("API_KEY", "local")})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(base, model, messages, max_tokens, effort, temperature, tools=None, stream=True, extra=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": stream, "stream_options": {"include_usage": True}}
    if effort is not None:
        body["reasoning_effort"] = effort
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    if tools:
        body["tools"] = tools
    body.update(extra or {})
    start = time.time()
    first = None
    content, reasoning, tool_calls, usage, finish, timings = [], [], {}, None, None, None
    with post(base, body) as response:
        if not stream:
            data = json.load(response)
            end = time.time()
            msg = data["choices"][0]["message"]
            return {"wall_s": end - start, "ttft_s": None, "content": msg.get("content") or "",
                    "reasoning": msg.get("reasoning_content") or "", "tool_calls": msg.get("tool_calls") or [],
                    "usage": data.get("usage"), "finish": data["choices"][0].get("finish_reason"),
                    "timings": data.get("timings")}
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                got = False
                for key, sink in (("content", content), ("reasoning_content", reasoning), ("reasoning", reasoning)):
                    if delta.get(key):
                        sink.append(delta[key]); got = True
                for call in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                    fn = call.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                    got = True
                if got and first is None:
                    first = time.time()
                finish = choice.get("finish_reason") or finish
    end = time.time()
    return {"wall_s": end - start, "ttft_s": (first or end) - start, "decode_s": end - (first or end),
            "content": "".join(content), "reasoning": "".join(reasoning),
            "tool_calls": list(tool_calls.values()), "usage": usage, "finish": finish, "timings": timings}


def summarize(r):
    usage = r.get("usage") or {}
    comp = usage.get("completion_tokens")
    out = {"wall_s": round(r["wall_s"], 2), "ttft_s": round(r["ttft_s"], 2) if r.get("ttft_s") is not None else None,
           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp,
           "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "finish": r.get("finish")}
    if comp and r.get("decode_s"):
        # first token arrives at ttft; the remaining comp-1 tokens take decode_s
        out["decode_tok_s"] = round((comp - 1) / max(r["decode_s"], 1e-6), 1)
    if comp:
        out["e2e_tok_s"] = round(comp / r["wall_s"], 1)
    if out["prompt_tokens"] and r.get("ttft_s"):
        out["prefill_tok_s"] = round(out["prompt_tokens"] / r["ttft_s"], 1)
    t = r.get("timings") or {}
    for k in ("prompt_per_second", "predicted_per_second", "prompt_n", "predicted_n", "cache_n",
              "draft_n", "draft_n_accepted"):
        if k in t:
            out["server_" + k] = t[k]
    return out


def gpu_info():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,name,power.limit,memory.used,clocks.max.sm,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # remote client
        return repr(e)


def gsm_answer(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("label"); ap.add_argument("out")
    ap.add_argument("--suite", default="short,long,followup,tools,gsm")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gsm-n", type=int, default=100)
    ap.add_argument("--long", default="64k,128k,245k")
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus"))
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request body")
    a = ap.parse_args()
    extra = json.loads(a.extra)
    suites = a.suite.split(",")
    os.makedirs(a.out, exist_ok=True)
    report = {"label": a.label, "base": a.base, "model": a.model, "temperature": a.temperature, "extra": extra,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "gpus": gpu_info(), "results": {}}
    path = os.path.join(a.out, a.label + ".json")
    if os.path.exists(path):  # a later partial run adds to the same label instead of replacing it
        report["results"] = json.load(open(path)).get("results", {})

    def save():
        json.dump(report, open(path, "w"), indent=1, ensure_ascii=False)

    def log(name, s):
        print(time.strftime("%H:%M:%S"), a.label, name, json.dumps(s, ensure_ascii=False), flush=True)

    # warm-up: first request after a start reads low (HyperQwen's rule); discard it
    chat(a.base, a.model, [{"role": "user", "content": "Say OK."}], 16, "none", a.temperature, extra=extra)

    if "short" in suites:
        for task in SHORT:
            r = chat(a.base, a.model, [{"role": "user", "content": task["prompt"]}], task["max_tokens"],
                     task["effort"], a.temperature, extra=extra)
            s = summarize(r)
            s["answer_head"] = r["content"][:300]
            if task["id"] == "reason":
                s["answer_line"] = (re.findall(r"ANSWER:\s*(\S+)", r["content"]) or [None])[-1]
            if task["id"] == "edit":
                s["has_retries"] = "retries" in r["content"] and "attempts" in r["content"]
            report["results"]["short/" + task["id"]] = s
            open(os.path.join(a.out, f"{a.label}-{task['id']}.txt"), "w").write(r["reasoning"] + "\n=====\n" + r["content"])
            log(task["id"], {k: v for k, v in s.items() if k != "answer_head"}); save()

    if "tools" in suites:
        msgs = [{"role": "system", "content": "You are a coding agent. Use the tools to inspect the workspace before answering."},
                {"role": "user", "content": "What Python version does this repo require? Check pyproject.toml."}]
        r = chat(a.base, a.model, msgs, 512, "none", a.temperature, tools=TOOLS, extra=extra)
        s = summarize(r)
        s["tool_calls"] = r["tool_calls"]
        ok = False
        for c in r["tool_calls"]:
            try:
                ok = ok or (c["name"] in ("read_file", "run_shell") and "pyproject" in json.loads(c["arguments"] or "{}").get(
                    "path", json.loads(c["arguments"] or "{}").get("command", "")))
            except Exception:
                pass
        s["tool_call_ok"] = ok
        report["results"]["tools/read_file"] = s
        log("tools", s); save()

    if "long" in suites or "followup" in suites:
        corpus_meta = json.load(open(os.path.join(a.corpus, "corpus.json")))
        fn = corpus_meta["function"]
        for label in a.long.split(","):
            # a unique first line keeps each size cold: the dumps share their leading files, and every engine here
            # has a prefix cache that would otherwise serve the smaller dump's tokens to the larger request
            text = f"[session {a.label} {label} {time.time():.0f}]\n" + open(os.path.join(a.corpus, f"long-{label}.txt")).read()
            msgs = [{"role": "user", "content": text}]
            r = chat(a.base, a.model, msgs, 1536, "none", a.temperature, extra=extra)
            s = summarize(r)
            s["found_file"] = corpus_meta["first_file"] in r["content"]
            s["found_fn"] = fn in r["content"]
            s["answer_head"] = r["content"][:400]
            report["results"][f"long/{label}"] = s
            open(os.path.join(a.out, f"{a.label}-long-{label}.txt"), "w").write(r["content"])
            log("long-" + label, {k: v for k, v in s.items() if k != "answer_head"}); save()
            if "followup" in suites and label == a.long.split(",")[-1]:
                msgs2 = msgs + [{"role": "assistant", "content": r["content"]},
                                {"role": "user", "content": "Now list every file in the dump whose path contains "
                                 "'attn' or 'attention', one per line, nothing else."}]
                r2 = chat(a.base, a.model, msgs2, 1024, "none", a.temperature, extra=extra)
                s2 = summarize(r2)
                s2["answer_head"] = r2["content"][:400]
                report["results"][f"followup/{label}"] = s2
                log("followup-" + label, {k: v for k, v in s2.items() if k != "answer_head"}); save()

    if "gsm" in suites:
        rows = [json.loads(l) for l in open(os.path.join(HERE, "tasks", "gsm8k_test.jsonl"))][: a.gsm_n]

        def one(row):
            q = row["question"] + "\n\nSolve it step by step, briefly. End with a line 'ANSWER: <number>'."
            r = chat(a.base, a.model, [{"role": "user", "content": q}], 2048, "none", 0.0, extra=extra)
            gold = row["answer"].split("####")[-1].strip().replace(",", "")
            got = (re.findall(r"ANSWER:\s*\$?(-?[\d,\.]+)", r["content"]) or [gsm_answer(r["content"])])[-1]
            got = (got or "").replace(",", "").rstrip(".")
            try:
                correct = float(got) == float(gold)
            except ValueError:
                correct = False
            return correct, summarize(r)

        t0 = time.time()
        with cf.ThreadPoolExecutor(a.concurrency) as pool:
            outs = list(pool.map(one, rows))
        wall = time.time() - t0
        toks = sum((s.get("completion_tokens") or 0) for _, s in outs)
        report["results"]["gsm8k"] = {"n": len(rows), "correct": sum(c for c, _ in outs), "wall_s": round(wall, 1),
                                      "completion_tokens": toks, "aggregate_tok_s": round(toks / wall, 1),
                                      "concurrency": a.concurrency,
                                      "median_decode_tok_s": statistics.median([s["decode_tok_s"] for _, s in outs if s.get("decode_tok_s")] or [0])}
        log("gsm8k", report["results"]["gsm8k"]); save()

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    print("DONE", path)


if __name__ == "__main__":
    main()
