"""
Generate LaTeX results tables for the inventory-control policy benchmarks paper.

Algorithmic description
=======================
For each benchmark problem (vanilla lost-sales, fixed-order-cost lost-sales,
dual-sourcing) this script:
  1. Enumerates the FULL instance grid from the Rust bindings (invman_rust), so
     every paper-grid instance gets a row even if it has not finished training.
  2. Reads each instance summary JSON under
     outputs/benchmarks/<run_tag>/instances/<name>.json (written by the suite
     runners) and pulls the classical-heuristic mean costs, the learned-policy
     mean costs, and the optimal/reference cost.
  3. Emits one LaTeX cost table per problem: rows = instances, columns =
     classical heuristics + learned policy approximators (+ optimal / published
     DRL gap where available). Cells with no data yet are left EMPTY; obviously
     diverged CMA-ES runs (cost > DIVERGENCE_FACTOR x best heuristic) are shown
     as a dagger so they don't pollute the table. The best (lowest) cost in each
     row is bolded.

Re-run any time as training progresses to refill the tables. Output goes to
paper/generated/results_<problem>.tex plus a combined all_results_tables.tex,
ready to \\input into paper/inventory_control_policy_benchmarks.tex.

Usage:
  PYTHONPATH=/home/nima/code/ml/invman \\
  /home/nima/storage/samsung8tb/miniconda3/bin/python3.12 paper/generate_results_tables.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import invman_rust

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "paper" / "generated"
DIVERGENCE_FACTOR = 5.0

# ---- short labels --------------------------------------------------------------
POLICY_LABELS = {
    "linear_categorical_quantity_q20": "L-Cat",
    "linear_sigmoid_direct_quantity": "L-Sig",
    "linear_soft_gated_direct_quantity": "L-SGD",
    "nn_soft_gated_direct_quantity_h8_selu": "NN-SGD",
    "linear_hard_gated_direct_quantity": "L-HGD",
    "linear_soft_gated_ordinal_quantity": "L-SGO",
    "nn_soft_gated_ordinal_quantity_h8_selu": "NN-SGO",
    "soft_tree_depth1_linear_leaf": "Tree-1",
    "soft_tree_depth2_linear_leaf": "Tree-2",
}
HEURISTIC_LABELS = {
    "myopic1": "M1", "myopic2": "M2", "svbs": "SVBS",
    "s_s": "$(s,S)$", "s_nq": "$(s,nQ)$", "modified_s_s_q": "$(s,S,q)$",
    "single_index": "SI", "dual_index": "DI", "capped_dual_index": "CDI",
    "tailored_base_surge": "TBS",
}


def _short_policy(pid: str) -> str:
    if pid in POLICY_LABELS:
        return POLICY_LABELS[pid]
    # dual-sourcing soft-tree specs: compress to a readable token
    return (pid.replace("soft_tree_", "ST-").replace("_quantity", "")
            .replace("capped_dual_index", "cdi").replace("dual_index", "di")
            .replace("delta_smallcap_targets", "sc").replace("_targets", "")
            .replace("axis_constant", "axc").replace("oblique_linear", "obl")
            .replace("__", "_").strip("_"))[:16]


def _fmt(v, *, best: bool = False, diverged: bool = False) -> str:
    if diverged:
        return r"$\dagger$"
    if v is None:
        return ""
    s = f"{v:.3f}"
    return rf"\textbf{{{s}}}" if best else s


def _instance_data(run_tag: str, name: str) -> dict | None:
    path = REPO / "outputs" / "benchmarks" / run_tag / "instances" / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _learned_cost(summary: dict, pid: str):
    lp = summary.get("learned_policies", {}).get(pid)
    if not lp:
        return None
    return lp.get("evaluation", {}).get("learned_policy", {}).get("mean_cost")


def _heur_cost(summary: dict, key: str):
    # vanilla/fixed-cost nest under heuristics.evaluation; dual-sourcing is flat.
    h = summary.get("heuristics", {})
    table = h.get("evaluation") if isinstance(h, dict) and isinstance(h.get("evaluation"), dict) else h
    cell = table.get(key) if isinstance(table, dict) else None
    return cell.get("mean_cost") if isinstance(cell, dict) else None


def _render_table(*, label, caption, col_groups, rows):
    """col_groups: list of (header, [subcolumn labels]). rows: list of (row_label, [cell strings])."""
    ncol = 1 + sum(len(sub) for _, sub in col_groups)
    colspec = "l" + "".join("r" * len(sub) for _, sub in col_groups)
    group_hdr = " & ".join([""] + [rf"\multicolumn{{{len(sub)}}}{{c}}{{{hd}}}" for hd, sub in col_groups])
    sub_hdr = " & ".join(["Instance"] + [s for _, sub in col_groups for s in sub])
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
        group_hdr + r" \\", sub_hdr + r" \\", r"\midrule",
    ]
    for row_label, cells in rows:
        lines.append(" & ".join([row_label] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\par\medskip\footnotesize Empty cells: not yet trained. "
              r"$\dagger$: diverged CMA-ES run (excluded). Bold: best (lowest) cost in the row.",
              r"\end{table}"]
    return "\n".join(lines)


def _build_problem(*, run_tag, instances, heuristics, policies, optimal_getter, extra_cols=None):
    """Return (n_done, n_total, latex). `instances` = [(name, row_label)]."""
    heur_keys = [k for k, _ in heuristics]
    pol_ids = [p for p, _ in policies]
    rows = []
    n_done = 0
    for name, row_label in instances:
        summary = _instance_data(run_tag, name)
        if summary is not None:
            n_done += 1
        # gather raw costs for best-in-row bolding (heuristics + policies)
        heur_vals = {k: (_heur_cost(summary, k) if summary else None) for k in heur_keys}
        best_heur = min([v for v in heur_vals.values() if v is not None], default=None)
        pol_vals = {}
        for pid in pol_ids:
            c = _learned_cost(summary, pid) if summary else None
            diverged = c is not None and best_heur is not None and c > DIVERGENCE_FACTOR * best_heur
            pol_vals[pid] = (None if diverged else c, diverged)
        all_costs = [v for v in heur_vals.values() if v is not None] + \
                    [c for (c, dv) in pol_vals.values() if c is not None]
        row_best = min(all_costs) if all_costs else None
        cells = []
        for k in heur_keys:
            v = heur_vals[k]
            cells.append(_fmt(v, best=(v is not None and row_best is not None and abs(v - row_best) < 1e-6)))
        for pid in pol_ids:
            c, dv = pol_vals[pid]
            cells.append(_fmt(c, best=(c is not None and row_best is not None and abs(c - row_best) < 1e-6), diverged=dv))
        if optimal_getter is not None:
            cells.append(_fmt(optimal_getter(name, summary)))
        if extra_cols is not None:
            cells.extend(extra_cols(name, summary))
        rows.append((row_label, cells))
    col_groups = [("Heuristics", [lbl for _, lbl in heuristics]),
                  ("Learned policy approximators", [lbl for _, lbl in policies])]
    if optimal_getter is not None:
        col_groups.append(("", ["Opt."]))
    return n_done, len(instances), rows, col_groups


# ---- problem configs -----------------------------------------------------------
def vanilla():
    names = [n for n in invman_rust.lost_sales_reference_instance_names() if n != "vanilla_l4_p4_poisson5"]
    def label(n):
        r = invman_rust.lost_sales_reference_costs(n)
        dem = {"Poisson": "Pois", "Geometric": "Geom", "MarkovModulatedPoisson2": "MMPP2"}[r["demand_kind"]]
        tag = "+" if (n.endswith("pos_" ) or "mmpp2_pos" in n) else ("-" if "mmpp2_neg" in n else "")
        return f"{dem}{tag} $L{int(r['lead_time'])}$ $p{int(r['shortage_cost'])}$"
    instances = sorted(((n, label(n)) for n in names),
                       key=lambda x: x[0])
    heur = [("myopic1", "M1"), ("myopic2", "M2"), ("svbs", "SVBS")]
    pol = [(p, POLICY_LABELS[p]) for p in (
        "linear_soft_gated_direct_quantity", "nn_soft_gated_direct_quantity_h8_selu",
        "linear_soft_gated_ordinal_quantity", "nn_soft_gated_ordinal_quantity_h8_selu",
        "soft_tree_depth1_linear_leaf", "soft_tree_depth2_linear_leaf")]
    def opt(n, s):
        return invman_rust.lost_sales_reference_costs(n)["costs"].get("optimal")
    return dict(run_tag="lost_sales_paper_suite_2k_scale20_seed42", instances=instances,
                heuristics=heur, policies=pol, optimal_getter=opt,
                caption="Vanilla lost-sales benchmark: mean cost of learned policy approximators vs.\\ classical heuristics across the 32-instance paper grid. Optimal is the literature value where reported.",
                label="tab:results-vanilla-lost-sales")


def fixed_cost():
    grid = invman_rust.lost_sales_fixed_order_cost_expand_experiment_grid("lost_sales_style_full_grid_mu5")
    grid = [g for g in grid if int(g["params"]["lead_time"]) in (4, 6, 8, 10)]
    def label(g):
        p = g["params"]
        dem = {"Poisson": "Pois", "Geometric": "Geom", "MarkovModulatedPoisson2": "MMPP2"}.get(p["demand_dist_name"], p["demand_dist_name"])
        tag = "+" if "mmpp2_pos" in g["name"] else ("-" if "mmpp2_neg" in g["name"] else "")
        return f"{dem}{tag} $L{int(p['lead_time'])}$ $p{int(p['shortage_cost'])}$ $K{int(p['fixed_order_cost'])}$"
    instances = sorted(((g["name"], label(g)) for g in grid), key=lambda x: x[0])
    heur = [("s_s", "$(s,S)$"), ("s_nq", "$(s,nQ)$"), ("modified_s_s_q", "$(s,S,q)$")]
    pol = [(p, POLICY_LABELS[p]) for p in (
        "linear_soft_gated_direct_quantity", "nn_soft_gated_direct_quantity_h8_selu",
        "linear_soft_gated_ordinal_quantity", "nn_soft_gated_ordinal_quantity_h8_selu",
        "soft_tree_depth1_linear_leaf", "soft_tree_depth2_linear_leaf")]
    def opt(n, s):
        return (s or {}).get("optimal_reference", {}).get("mean_cost") if s else None
    return dict(run_tag="fixed_cost_paper_suite_2k_scale20_seed42", instances=instances,
                heuristics=heur, policies=pol, optimal_getter=opt,
                caption="Fixed-order-cost lost-sales benchmark: mean cost of learned policy approximators vs.\\ the $(s,S)$, $(s,nQ)$ and modified $(s,S,q)$ heuristics across the 64-instance grid.",
                label="tab:results-fixed-cost-lost-sales")


def dual_sourcing():
    refs = invman_rust.dual_sourcing_list_reference_instances()
    def label(r):
        return f"$l_r{int(r['regular_lead_time'])}$ $c_e{int(r['expedited_order_cost'])}$"
    instances = [(r["name"], label(r)) for r in refs]
    heur = [("single_index", "SI"), ("dual_index", "DI"), ("capped_dual_index", "CDI"), ("tailored_base_surge", "TBS")]
    refmap = {r["name"]: r for r in refs}
    done = {n: _instance_data("dual_sourcing_paper_suite", n) for n, _ in instances}
    pol_ids = []
    for s in done.values():
        if s:
            for pid in s.get("learned_policies", {}):
                if pid not in pol_ids:
                    pol_ids.append(pid)
    pol = [(p, _short_policy(p)) for p in pol_ids] or [("soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets", "ST-axc-cdi-sc")]
    def opt(n, s):
        if s and s.get("comparative_summary", {}).get("optimal_cost") is not None:
            return s["comparative_summary"]["optimal_cost"]
        return None
    def a3c_gap(n, s):
        g = refmap[n].get("published_optimality_gap_pct", {}).get("a3c")
        return [f"{g:.2f}\\%" if g is not None else ""]
    return dict(run_tag="dual_sourcing_paper_suite", instances=instances,
                heuristics=heur, policies=pol, optimal_getter=opt, extra_cols=a3c_gap,
                caption="Dual-sourcing benchmark (Gijsbrechts et al.\\ 2022, Fig.~9): mean cost of learned soft-tree policies vs.\\ the single-/dual-/capped-dual-index and tailored-base-surge heuristics, the DP optimal, and the published A3C optimality gap.",
                label="tab:results-dual-sourcing")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = []
    for name, cfg in (("vanilla_lost_sales", vanilla()),
                      ("fixed_cost_lost_sales", fixed_cost()),
                      ("dual_sourcing", dual_sourcing())):
        extra = cfg.pop("extra_cols", None)
        caption = cfg.pop("caption"); label = cfg.pop("label")
        n_done, n_total, rows, col_groups = _build_problem(extra_cols=extra, **cfg)
        if extra is not None:  # dual-sourcing published A3C gap column
            col_groups.append(("", ["A3C gap"]))
        latex = _render_table(label=label,
                              caption=caption + f" ({n_done}/{n_total} instances trained so far.)",
                              col_groups=col_groups, rows=rows)
        (OUT_DIR / f"results_{name}.tex").write_text(latex + "\n")
        combined.append(latex)
        print(f"  {name}: {n_done}/{n_total} instances -> generated/results_{name}.tex")
    (OUT_DIR / "all_results_tables.tex").write_text("\n\n".join(combined) + "\n")
    print(f"  combined -> {OUT_DIR / 'all_results_tables.tex'}")


if __name__ == "__main__":
    main()
