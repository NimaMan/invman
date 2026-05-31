"""
Render the streamed Rust-heuristic results (/tmp/missing_heuristics.jsonl) into
paste-ready Python config literals for invman/problems/lost_sales/problem_info.py:

  - MMPP2 positive/negative tables (M1/M2/SVBS per lead time), repo-computed.
  - The Geometric and Poisson-L10 M1/SVBS cells that were missing.
  - A crosscheck table (computed myopic2 vs literature) for trust.

Usage:
  OUT_JSONL=/tmp/missing_heuristics.jsonl \
  PYTHONPATH=/home/nima/code/ml/invman env -u VIRTUAL_ENV \
    /home/nima/miniconda3/bin/python scripts/lost_sales/render_computed_heuristics_config.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

OUT_JSONL = os.environ.get("OUT_JSONL", "/tmp/missing_heuristics.jsonl")

# demand token -> shortage_cost -> lead_time -> {heuristic: value}
fill = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
crosscheck = []

with open(OUT_JSONL) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("error"):
            print(f"# ERROR {rec['instance']} {rec['heuristic']}: {rec['error']}")
            continue
        if rec["role"] == "crosscheck":
            crosscheck.append(rec)
            continue
        token = {"Geometric": "Geometric", "Poisson": "Poisson",
                 "MarkovModulatedPoisson2": None}[rec["demand"]]
        # MMPP2 tokens depend on pos/neg, encoded in instance name
        if rec["demand"] == "MarkovModulatedPoisson2":
            token = "MMPP2Positive" if "mmpp2_pos" in rec["instance"] else "MMPP2Negative"
        key = {"myopic1": "M1", "myopic2": "M2", "svbs": "SVBS"}[rec["heuristic"]]
        fill[token][rec["shortage_cost"]][rec["lead_time"]][key] = rec["value"]

print("=" * 70)
print("CROSSCHECK (computed myopic2 vs literature)")
print("=" * 70)
for rec in sorted(crosscheck, key=lambda r: r["instance"]):
    lit = rec["literature"]
    diff = None if lit is None else round(rec["value"] - lit, 3)
    print(f"  {rec['instance']:24s} computed={rec['value']:.4f}  lit={lit}  diff={diff}")

print()
print("=" * 70)
print("FILL VALUES (repo-computed)")
print("=" * 70)
for token in sorted(fill):
    for p in sorted(fill[token]):
        print(f"\n# {token}  shortage_cost={p}")
        for L in sorted(fill[token][p]):
            cells = fill[token][p][L]
            print(f"  L{L}: {json.dumps(cells)}")

# Emit the MMPP2 table literals directly (since those tables don't exist yet)
for sign in ("MMPP2Positive", "MMPP2Negative"):
    for p in (4, 19):
        if p not in fill.get(sign, {}):
            continue
        print(f"\n{sign}_demand_shortage_cost_{p} = {{")
        for L in sorted(fill[sign][p]):
            c = fill[sign][p][L]
            print(f'    {L}: {{"optimal": None, "M2": {c.get("M2")}, "M1": {c.get("M1")}, '
                  f'"SVBS": {c.get("SVBS")}, "CappedBS": None}},')
        print("}")
