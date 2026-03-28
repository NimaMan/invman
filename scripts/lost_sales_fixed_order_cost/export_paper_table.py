import argparse
import json
from pathlib import Path


DEFAULT_SUITE_JSON = Path(
    "outputs/benchmarks/fixed_cost_l4_canonical_suite_5k_paperlike/fixed_cost_canonical_suite.json"
)
DEFAULT_OUTPUT = Path("paper/generated/fixed_cost_canonical_table.tex")


DISPLAY_NAMES = {
    "linear_categorical_quantity": "Linear, categorical quantity",
    "linear_gated_ordinal_quantity": "Linear, gated ordinal quantity",
    "nn_categorical_quantity": "NN, categorical quantity$^{\\dagger}$",
    "nn_gated_ordinal_quantity": "NN, gated ordinal quantity",
    "soft_tree_depth2_linear_leaf": "Soft tree, oblique depth-2 linear leaf",
    "soft_tree_depth1_linear_leaf": "Soft tree, oblique depth-1 linear leaf",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the canonical fixed-cost benchmark summary to a TeX table for the paper."
    )
    parser.add_argument("--suite_json", default=str(DEFAULT_SUITE_JSON))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.4f}"


def build_table(summary: dict) -> str:
    heuristic_eval = summary["heuristics"]["evaluation"]
    best_heuristic = min(
        heuristic_eval[name]["mean_cost"] for name in ("s_s", "s_nq", "modified_s_s_q")
    )

    heuristic_rows = [
        ("$(s,S)$", heuristic_eval["s_s"]["mean_cost"]),
        ("$(s,nQ)$", heuristic_eval["s_nq"]["mean_cost"]),
        ("modified $(s,S,q)$", heuristic_eval["modified_s_s_q"]["mean_cost"]),
    ]

    policy_rows = []
    for item in summary["learned_policies"]:
        label = DISPLAY_NAMES[item["id"]]
        cost = item["evaluation"]["learned_policy"]["mean_cost"]
        gap = cost - best_heuristic
        policy_rows.append((label, cost, gap))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Canonical fixed-order-cost lost-sales benchmark for $L=4$, $p=4$, $K=5$, and Poisson demand with mean $5$. All learned policies use $5000$ CMA-ES iterations, population size $50$, training horizon $2000$, and long-run evaluation over $10$ seeds with horizon $10^6$.}",
        r"\label{tab:fixed-cost-canonical}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Policy family & Mean cost & Gap vs.\ best heuristic \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Heuristic baselines}} \\",
    ]

    for label, cost in heuristic_rows:
        lines.append(f"{label} & {fmt(cost)} & {fmt(cost - best_heuristic)} \\\\")

    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{3}{l}{\textit{Learned policy families}} \\",
        ]
    )

    for label, cost, gap in policy_rows:
        lines.append(f"{label} & {fmt(cost)} & {fmt(gap)} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip\footnotesize{$^{\dagger}$ The current NN categorical-quantity baseline matches the linear categorical baseline exactly and should be re-verified before publication claims rely on it.}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    suite_json = Path(args.suite_json)
    output = Path(args.output)
    summary = json.loads(suite_json.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_table(summary), encoding="utf-8")
    print(json.dumps({"suite_json": str(suite_json.resolve()), "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
