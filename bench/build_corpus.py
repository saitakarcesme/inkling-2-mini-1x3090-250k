"""Freeze the long-context inputs used by every engine step.

usage: build_corpus.py <source_root> <tokenizer.json> <out_dir>

Real source code, files in sorted path order, each under a "=== path ===" header. The dump is cut on file
boundaries to three sizes (target prompt tokens incl. the question). A question asks about a function that sits in
the FIRST file of the dump, so the answer has to come from the far end of the context. The texts are written once
and then reused unchanged by llama.cpp, Maya and every later step.
"""
import json
import os
import re
import sys

from tokenizers import Tokenizer

SRC, TOK, OUT = sys.argv[1:4]
TARGETS = {"64k": 64_000, "128k": 128_000, "245k": 245_000}
EXTS = (".cu", ".cuh", ".cpp", ".h", ".hpp", ".c", ".py", ".md")

tok = Tokenizer.from_file(TOK)
files = []
for root, dirs, names in os.walk(SRC):
    dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("third_party", "build", "assets"))
    for name in sorted(names):
        if name.endswith(EXTS):
            path = os.path.join(root, name)
            try:
                text = open(path, encoding="utf-8").read()
            except UnicodeDecodeError:
                continue
            if 200 < len(text) < 400_000:
                files.append((os.path.relpath(path, SRC), text))

# A function from the first file that is defined nowhere else: the far-end fact the question asks about.
DEF = re.compile(r"^(?:static\s+|inline\s+|__global__\s+|template<[^>]*>\s*)*[\w:<>\*]+[\s\*&]+(\w{6,})\s*\([^;]*$", re.M)
everything = "\n".join(t for _, t in files[:60])
first_rel = fn = None
for rel, text in files:
    for name in DEF.findall(text):
        if name not in ("main", "return") and everything.count(name + "(") <= 3:
            first_rel, fn = rel, name
            break
    if fn:
        break
files.sort(key=lambda f: f[0] != first_rel)  # that file goes first
question = (f"\n\n---\nYou were given a dump of a C++/CUDA/Python code base above. Answer from the dump only.\n"
            f"1. Which file defines the function `{fn}`, and what does it do? Quote its signature.\n"
            f"2. In five bullet points, describe how the code base is organised.\n")
q_tokens = len(tok.encode(question).ids)

os.makedirs(OUT, exist_ok=True)
meta = {"source_root": SRC, "first_file": first_rel, "function": fn, "files_available": len(files), "sizes": {}}
for label, target in TARGETS.items():
    parts, used = [], q_tokens
    for rel, text in files:
        block = f"=== {rel} ===\n{text}\n"
        n = len(tok.encode(block).ids)
        if used + n > target:
            continue  # skip a file that would overshoot; smaller later files may still fit
        parts.append(block)
        used += n
        if used > target - 400:
            break
    prompt = "".join(parts) + question
    n_total = len(tok.encode(prompt).ids)
    open(os.path.join(OUT, f"long-{label}.txt"), "w").write(prompt)
    meta["sizes"][label] = {"tokens_raw": n_total, "files": len(parts), "chars": len(prompt)}
    print(label, n_total, "tokens", len(parts), "files")
json.dump(meta, open(os.path.join(OUT, "corpus.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
