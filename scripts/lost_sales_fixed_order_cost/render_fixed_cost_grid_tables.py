"""
Render fixed-order-cost lost-sales full-grid results as LaTeX tables in the
style of invman_lostsales.tex (Table perf6810): costs grouped by lead time with
Geometric/Poisson sub-columns, rows = shortage cost x policy, one table per
setup cost K.

Objective
=========
Turn the completed fixed-cost instance summaries under
  outputs/benchmarks/<RUN_TAG>/instances/*.json
into paper-ready mean-cost tables for paper/inventory_control_policy_benchmarks.tex.

The fixed-cost suite compares three classical heuristics -- (s,S), (s,nQ), and
the modified (s,S,q) -- against six learned CMA-ES policy families. A learned
run is treated as DIVERGED (CMA-ES failure, not true performance) and rendered
as "---" when its mean cost exceeds DIVERGENCE_FACTOR x the instance's best
heuristic cost; diverged cells are listed in the footnote so they can be re-run.

Usage:
  PYTHONPATH=/home/nima/code/ml/invman env -u VIRTUAL_ENV \
    /home/nima/storage/samsung8tb/miniconda3/bin/python3.12 \
    scripts/lost_sales_fixed_order_cost/render_fixed_cost_grid_tables.py \
    [RUN_TAG]
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

RUN_TAG = sys.argv[1] if len(sys.argv) > 1 else "fixed_cost_paper_suite_2k_scale20_seed42"
INSTANCES_DIR = f"outputs/benchmarks/{RUN_TAG}/instances"
DIVERGENCE_FACTOR = 5.0
LEAD_TIMES = [4, 6, 8, 10]
DEMANDS = [("Geometric", "geom"), ("Poisson", "pois")]
SHORTAGES = [4, 19]
SETUP_COSTS = [5, 25]

# (json heuristic key, LaTeX row label)
HEURISTIC_ROWS = [
    ("s_s", r"$(s,S)$"),
    ("s_nq", r"$(s,nQ)$"),
    ("modified_s_s_q", r"modified $(s,S,q)$"),
]
# (json policy id, LaTeX row label)
POLICY_ROWS = [
    ("linear_soft_gated_direct_quantity", "Linear soft-gated direct"),
    ("nn_soft_gated_direct_quantity_h8_selu", "NN soft-gated direct"),
    ("linear_soft_gated_ordinal_quantity", "Linear soft-gated ordinal"),
    ("nn_soft_gated_ordinal_quantity_h8_selu", "NN soft-gated ordinal"),
    ("soft_tree_depth1_linear_leaf", "Soft tree, depth-1"),
    ("soft_tree_depth2_linear_leaf", "Soft tree, depth-2"),
]


def _load():
    """cells[(K,p,L,demand_token)] = {'heur': {key:cost}, 'pol': {pid:cost}, 'best_heur': cost}."""
    cells = {}
    diverged = []
    for f in sorted(glob.glob(INSTANCES_DIR + "/*.json")):
        d = json.load(open(f))
        p = d["params"]
        demand = str(p["demand_dist_name"])
        token = {"Poisson": "pois", "Geometric": "geom", "MarkovModulatedPoisson2": "mmpp2"}.get(demand)
        key = (int(p["fixed_order_cost"]), int(round(float(p["shortage_cost"]))), int(p["lead_time"]), token)
        heur = {k: v["mean_cost"] for k, v in d["heuristics"]["evaluation"].items()}
        best_heur = d.get("comparative_summary", {}).get("best_heuristic_cost")
        pol = {}
        for pid, v in d.get("learned_policies", {}).items():
            c = v["evaluation"]["learned_policy"]["mean_cost"]
            if c is None or (isinstance(c, float) and (math.isnan(c) or math.isinf(c))) or (
                best_heur and c > DIVERGENCE_FACTOR * best_heur
            ):
                diverged.append((d["reference_instance"], pid, c))
                pol[pid] = None
            else:
                pol[pid] = c
        cells[key] = {"heur": heur, "pol": pol, "best_heur": best_heur}
    return cells, diverged


def _fmt(v):
    return "---" if v is None else f"{v:.3f}"


def _col_values(cells, K, p, source_kind, source_key):
    """Return the 8 cell strings (L x demand) for one policy/heuristic row; also
    the raw values so the caller can bold the best learned policy per column."""
    out = []
    for L in LEAD_TIMES:
        for _demand_name, token in DEMANDS:
            cell = cells.get((K, p, L, token))
            if cell is None:
                out.append(None)
                continue
            if source_kind == "heur":
                out.append(cell["heur"].get(source_key))
            else:
                out.append(cell["pol"].get(source_key))
    return out


def _render_table(cells, K):
    col_spec = "|l|l|" + "|".join(["c|c"] * len(LEAD_TIMES)) + "|"
    head_groups = " & ".join(
        rf"\multicolumn{{2}}{{c{'||' if i < len(LEAD_TIMES) - 1 else '|'}}}{{Lead Time = {L}}}"
        for i, L in enumerate(LEAD_TIMES)
    )
    demand_hdr = " & ".join("Geometric & Poisson" for _ in LEAD_TIMES)
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{Average cost of policy function approximators and classical heuristics on the "
        rf"fixed-order-cost lost-sales grid with setup cost $K={K}$, lead times $L\in\{{4,6,8,10\}}$, "
        rf"penalty $p\in\{{4,19\}}$, and Geometric/Poisson demand with mean $5$. Lower is better; "
        rf"diverged CMA-ES runs are shown as ``---'' (see footnote).}}",
        rf"\label{{tab:fixed-cost-grid-K{K}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\centering",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        rf" & & {head_groups} \\",
        rf"Shortage & Policy & {demand_hdr} \\",
        r" Cost & & " + " & ".join([" "] * (2 * len(LEAD_TIMES))) + r" \\ \hline",
    ]
    for p in SHORTAGES:
        # precompute per-column best learned-policy value (for bolding)
        col_best = []
        for ci in range(2 * len(LEAD_TIMES)):
            vals = []
            for pid, _ in POLICY_ROWS:
                v = _col_values(cells, K, p, "pol", pid)[ci]
                if v is not None:
                    vals.append(v)
            col_best.append(min(vals) if vals else None)

        rows = []
        for key, label in HEURISTIC_ROWS:
            cells_str = [_fmt(v) for v in _col_values(cells, K, p, "heur", key)]
            rows.append((label, cells_str))
        for pid, label in POLICY_ROWS:
            raw = _col_values(cells, K, p, "pol", pid)
            cells_str = []
            for ci, v in enumerate(raw):
                if v is not None and col_best[ci] is not None and abs(v - col_best[ci]) < 1e-9:
                    cells_str.append(rf"\textbf{{{v:.3f}}}")
                else:
                    cells_str.append(_fmt(v))
            rows.append((label, cells_str))

        first = True
        for label, cells_str in rows:
            shortage_label = str(p) if first else ""
            first = False
            lines.append(f"{shortage_label} & {label} & " + " & ".join(cells_str) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _aggregate(cells, diverged):
    """Per-policy: mean relative gap to best heuristic (excluding diverged) and win count."""
    gaps = defaultdict(list)
    wins = defaultdict(int)
    n_inst = 0
    for key, cell in cells.items():
        bh = cell["best_heur"]
        if bh is None:
            continue
        n_inst += 1
        for pid, _ in POLICY_ROWS:
            c = cell["pol"].get(pid)
            if c is not None:
                g = 100.0 * (c - bh) / bh
                gaps[pid].append(g)
                if g < 0:
                    wins[pid] += 1
    return gaps, wins, n_inst


def main():
    cells, diverged = _load()
    gaps, wins, n_inst = _aggregate(cells, diverged)

    print(f"% Auto-generated by scripts/lost_sales_fixed_order_cost/render_fixed_cost_grid_tables.py")
    print(f"% Run tag: {RUN_TAG}; completed instances: {len(cells)}")
    print()
    for K in SETUP_COSTS:
        print(_render_table(cells, K))
        print()

    print("% ---- aggregate summary (for prose / appendix) ----")
    print(f"% completed instances: {n_inst}")
    for pid, label in POLICY_ROWS:
        g = gaps.get(pid, [])
        mean_gap = sum(g) / len(g) if g else float("nan")
        print(f"%   {label:28s} mean_gap%={mean_gap:7.2f}  beats_best_heur={wins.get(pid,0)}/{len(g)}")
    print(f"% diverged learned runs ({len(diverged)}):")
    for name, pid, c in diverged:
        print(f"%   {name}  {pid}  cost={c}")


if __name__ == "__main__":
    main()
