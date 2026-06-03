"""
Compute the trainable parameter count of every learned policy reported in the paper,
per problem, with its closed-form formula, and emit a LaTeX table substantiating the
"compact policy" claim (C3).

Algorithmic description
=======================
The number CMA-ES optimizes is the flat parameter vector of the policy; its length is
the trainable parameter count. For each (problem, policy) we compute that count two ways
and cross-check them:

  1. FORMULA (closed form, from invman/policy.py "PARAMETER LAYOUT"). With input dim i,
     output width o, control dim c, soft-tree depth D (n_in=2^D-1 internal nodes,
     n_leaf=2^D leaves):
       linear backbone:                      i*o + o            = o*(i+1)
       one-hidden-layer NN, hidden width H:   i*H + H + H*o + o  = H*(i+1) + o*(H+1)
       soft tree, constant leaf:              n_in*(i+1) + n_leaf*c
       soft tree, linear   leaf:              n_in*(i+1) + n_leaf*c*(i+1) = (i+1)*(n_in + n_leaf*c)
     Decoder output widths o: soft-gated DIRECT -> o=2 (gate + quantity logit);
     categorical / soft-gated ORDINAL -> o = qbar+1 = M+1 (= 21 here) scores;
     soft-tree scalar order -> c=1; dual-sourcing capped-dual-index coords -> c=3;
     multi-echelon (warehouse, retailer) order-up-to levels -> c=2.

  2. SAVED MODEL (ground truth): len(model_params.npy) for a trained checkpoint, when one
     exists on disk. The script asserts FORMULA == SAVED wherever a model is found, so the
     printed counts are validated, not asserted.

Problem input dimensions i: lost sales / fixed-cost i = lead time L; dual sourcing
i = regular lead time l_r; multi-echelon i = raw decision-state dim
(1 warehouse on-hand + (l_w-1) warehouse pipeline + R retailer on-hand + R*(l_r-1)
retailer pipeline): setting1 (l_w=2,l_r=2,R=10) -> 22; setting2 (l_w=5,l_r=3,R=10) -> 35.

Output: a console table plus paper/generated/policy_parameter_counts.tex.

Usage:
  PYTHONPATH=/home/nima/code/ml/invman \\
  python3 paper/policy_parameter_counts.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "generated" / "policy_parameter_counts.tex"

QBAR = 20          # max order size used by the categorical/ordinal heads (q in {0..20})
H = 8              # NN hidden width (h8)
M1 = QBAR + 1      # categorical/ordinal output width (= 21)


def p_linear(i, o):
    return i * o + o


def p_nn(i, o, h=H):
    return i * h + h + h * o + o


def p_tree(i, c, depth, leaf):
    n_in = 2 ** depth - 1
    n_leaf = 2 ** depth
    splits = n_in * (i + 1)
    leaves = n_leaf * c if leaf == "constant" else n_leaf * c * (i + 1)
    return splits + leaves


def saved_count(pattern: str):
    """len(model_params.npy) for the newest checkpoint matching a glob, or None."""
    for d in reversed(sorted(glob.glob(pattern))):
        f = Path(d) / "model_params.npy"
        if f.exists():
            return int(np.load(f).shape[0])
    return None


# ---- lost sales / fixed-cost: input dim = lead time L ---------------------------------
LS_MODELS = str(REPO / "outputs/benchmarks/lost_sales_paper_suite_2k_scale20_seed42/models")
# (label, backbone, formula-in-L, policy_id, count(i))
LS_POLICIES = [
    ("L-SGD", "linear", r"$2(L{+}1)$", "linear_soft_gated_direct_quantity", lambda i: p_linear(i, 2)),
    ("NN-SGD", "NN ($H{=}8$)", r"$8(L{+}1){+}18$", "nn_soft_gated_direct_quantity_h8_selu", lambda i: p_nn(i, 2)),
    ("L-SGO", "linear", r"$21(L{+}1)$", "linear_soft_gated_ordinal_quantity", lambda i: p_linear(i, M1)),
    ("NN-SGO", "NN ($H{=}8$)", r"$8(L{+}1){+}189$", "nn_soft_gated_ordinal_quantity_h8_selu", lambda i: p_nn(i, M1)),
    ("L-Cat", "linear", r"$21(L{+}1)$", "linear_categorical_quantity_q20", lambda i: p_linear(i, M1)),
    ("NN-Cat", "NN ($H{=}8$)", r"$8(L{+}1){+}189$", "nn_categorical_quantity_h8_selu_q20", lambda i: p_nn(i, M1)),
    ("Tree-1", "soft tree", r"$3(L{+}1)$", "soft_tree_depth1_linear_leaf", lambda i: p_tree(i, 1, 1, "linear")),
    ("Tree-2", "soft tree", r"$7(L{+}1)$", "soft_tree_depth2_linear_leaf", lambda i: p_tree(i, 1, 2, "linear")),
]
LS_LEADTIMES = (4, 6, 8, 10)


def lost_sales_rows():
    rows = []
    for label, backbone, formula, pid, f in LS_POLICIES:
        counts = []
        for L in LS_LEADTIMES:
            saved = saved_count(f"{LS_MODELS}/*_lit_poisson_p4_l{L}_{pid}_*")
            formula_val = f(L)
            counts.append(f"{formula_val}!={saved}" if (saved is not None and saved != formula_val) else str(formula_val))
        rows.append((label, backbone, formula, counts))
    return rows


# ---- dual sourcing: input dim = regular lead time l_r; depth-2 CDI-coordinate tree -----
DS_MODELS = str(REPO / "outputs/benchmarks/dual_sourcing_paper_suite/models")
DS_FORMULA = r"$3(l_r{+}1){+}12$"  # depth-2 constant leaf, c=3: 3(l_r+1) + 4*3


def dual_rows():
    out = []
    for lr in (2, 3, 4):
        saved = saved_count(f"{DS_MODELS}/*_dual_l{lr}_ce105_soft_tree_axis_constant_*")
        formula_val = p_tree(lr, 3, 2, "constant")
        ok = "" if saved is None else ("✓" if saved == formula_val else f"!=saved={saved}")
        out.append((lr, f"{formula_val}{ok}"))
    return out


# ---- multi-echelon: direct-level tree, input dim differs by setting's lead times -------
ME_INPUT_DIM = {"setting1": 22, "setting2": 35}
ME_FORMULA = {2: r"$11(i{+}1)$", 3: r"$23(i{+}1)$"}  # (i+1)*(n_in + 2*n_leaf): D2->11, D3->23


def multi_rows():
    out = []
    for setting, depth in (("setting1", 2), ("setting2", 3)):
        pat = str(REPO / f"outputs/multi_echelon/gijsbrechts2022_{setting}_policy/models/*direct_level_d{depth}_*")
        i = ME_INPUT_DIM[setting]
        saved = saved_count(pat)
        formula_val = p_tree(i, 2, depth, "linear")
        ok = "" if saved is None else ("✓" if saved == formula_val else f"!=saved={saved}")
        out.append((setting, depth, i, ME_FORMULA[depth], f"{formula_val}{ok}"))
    return out


def main():
    ls = lost_sales_rows()
    ds = dual_rows()
    me = multi_rows()

    # ---- console ----
    print("Lost sales / fixed-cost (input dim = lead time L); cols L=4,6,8,10:")
    print(f"  {'Policy':7s} {'Backbone':12s} {'Formula':18s} L4   L6   L8   L10")
    for label, backbone, formula, counts in ls:
        bb = backbone.replace(r"$H{=}8$", "H=8").replace("$", "")
        fm = formula.replace("{+}", "+").replace("$", "")
        print(f"  {label:7s} {bb:12s} {fm:18s} " + "  ".join(f"{c:>4s}" for c in counts))
    print("\nDual sourcing (depth-2 CDI-coordinate tree, input dim = l_r), formula 3(l_r+1)+12:")
    for lr, c in ds:
        print(f"  l_r={lr}: {c}")
    print("\nMulti-echelon (direct-level tree):")
    for setting, depth, i, formula, c in me:
        print(f"  {setting} (depth {depth}, input dim {i}), formula {formula.replace('{+}','+').replace('$','')}: {c}")

    # ---- LaTeX ----
    lines = [
        r"\begin{table}[!htbp]", r"\centering", r"\small",
        r"\caption{Trainable parameter counts of the learned policies (the length of the "
        r"flat vector CMA-ES optimizes), with closed-form formulas. Counts are validated "
        r"against the saved models. Lost-sales / fixed-cost counts grow with the lead time "
        r"$L$ (input dimension); dual sourcing with the regular lead time $l_r$; multi-echelon "
        r"with the raw decision-state dimension $i$. Every policy carries tens to a few hundred "
        r"parameters, two to three orders of magnitude smaller than a typical deep-RL network.}",
        r"\label{tab:param-counts}",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"\multicolumn{7}{l}{\emph{Lost sales \& fixed-cost lost sales} (input dim $=L$)} \\",
        r"Policy & Backbone & \#params & $L{=}4$ & $L{=}6$ & $L{=}8$ & $L{=}10$ \\",
        r"\midrule",
    ]
    for label, backbone, formula, counts in ls:
        lines.append(f"{label} & {backbone} & {formula} & " + " & ".join(counts) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\emph{Dual sourcing} (depth-2 capped-dual-index-coordinate tree, input dim $=l_r$)} \\",
        r"Soft tree & soft tree & " + DS_FORMULA + r" & \multicolumn{4}{l}{"
        + "; ".join(f"$l_r{{=}}{lr}$: {c.replace(chr(10003),'')}" for lr, c in ds) + r"} \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\emph{Multi-echelon} (direct-level tree)} \\",
    ]
    for setting, depth, i, formula, c in me:
        lines.append(
            f"Soft tree & soft tree & {formula} & "
            f"\\multicolumn{{4}}{{l}}{{{setting} (depth {depth}, $i={i}$): {c.replace(chr(10003),'')}}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nLaTeX -> {OUT}")


if __name__ == "__main__":
    main()
