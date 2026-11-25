"""Print the speed ladder from results/*.json as a markdown table.

usage: report.py <results_dir> label1 label2 ...   (labels in ladder order)
"""
import json
import os
import sys

d, labels = sys.argv[1], sys.argv[2:]
rows = [("short/code-new", "decode_tok_s", "New code, decode tok/s"),
        ("short/edit", "decode_tok_s", "Edit a 1.7K-token file, decode tok/s"),
        ("short/prose-tr", "decode_tok_s", "Turkish prose, decode tok/s"),
        ("short/reason", "decode_tok_s", "Reasoning (thinking on), decode tok/s"),
        ("long/64k", "ttft_s", "64K real code, time to first token (s)"),
        ("long/64k", "decode_tok_s", "64K real code, decode tok/s"),
        ("long/128k", "ttft_s", "128K real code, time to first token (s)"),
        ("long/128k", "decode_tok_s", "128K real code, decode tok/s"),
        ("long/245k", "ttft_s", "245K real code, time to first token (s)"),
        ("long/245k", "prefill_tok_s", "245K real code, prefill tok/s"),
        ("long/245k", "decode_tok_s", "245K real code, decode tok/s"),
        ("followup/245k", "ttft_s", "Follow-up turn at 245K, time to first token (s)"),
        ("gsm8k", "correct", "GSM8K first 100, solved"),
        ("gsm8k", "aggregate_tok_s", "GSM8K batch, aggregate tok/s")]
data = {l: json.load(open(os.path.join(d, l + ".json")))["results"] for l in labels}
print("| | " + " | ".join(labels) + " |")
print("|---|" + "---:|" * len(labels))
for key, field, name in rows:
    vals = []
    for l in labels:
        v = data[l].get(key, {}).get(field)
        vals.append("—" if v is None else (f"{v:,.1f}" if isinstance(v, float) else f"{v:,}"))
    print(f"| {name} | " + " | ".join(vals) + " |")
checks = []
for l in labels:
    r = data[l]
    checks.append(f"{l}: reason={r.get('short/reason', {}).get('answer_line')} edit={r.get('short/edit', {}).get('has_retries')} "
                  f"tool={r.get('tools/read_file', {}).get('tool_call_ok')} "
                  + " ".join(f"{k}:{r.get('long/' + k, {}).get('found_file')}/{r.get('long/' + k, {}).get('found_fn')}" for k in ("64k", "128k", "245k")))
print("\n" + "\n".join(checks))
