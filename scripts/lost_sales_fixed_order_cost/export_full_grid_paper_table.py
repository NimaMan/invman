import argparse
import json
from pathlib import Path


DEFAULT_SUITE_JSON = Path(
    "outputs/benchmarks/fixed_cost_full_grid_suite_5k_paperlike/fixed_cost_full_suite.json"
)
DEFAULT_OUTPUT = Path("paper/generated/fixed_cost_full_grid_table.tex")


DISPLAY_NAMES = {
    "linear_categorical_quantity": "Linear, categorical quantity",
    "linear_gated_ordinal_quantity": "Linear, gated ordinal quantity",
    "nn_categorical_quantity": "NN, categorical quantity$^{\\dagger}$",
    "nn_gated_ordinal_quantity": "NN, gated ordinal quantity",
    "soft_tree_depth2_linear_leaf": "Soft tree, oblique depth-2 linear leaf",
    "soft_tree_depth1_linear_leaf": "Soft tree, oblique depth-1 linear leaf",
}

ROW_ORDER = tuple(DISPLAY_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the fixed-cost full-grid benchmark summary to a TeX table for the paper."
    )
    parser.add_argument("--suite_json", default=str(DEFAULT_SUITE_JSON))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def fmt_pct(value: float) -> str:
    return f"{value:.3f}"


def _trusted_policy_summary(policies: dict) -> tuple[str, dict]:
    trusted_items = [(policy_id, item) for policy_id, item in policies.items() if item["status"] == "trusted"]
    if not trusted_items:
        raise ValueError("Expected at least one trusted policy row in the full-grid summary.")
    return min(trusted_items, key=lambda kv: kv[1]["mean_relative_gap_pct_vs_best_heuristic"])


def build_table(summary: dict) -> str:
    num_instances = int(summary["num_instances"])
    policies = summary["aggregate"]["policies"]
    best_trusted_id, best_trusted_item = _trusted_policy_summary(policies)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Aggregate fixed-order-cost lost-sales results over the literature-aligned "
            r"subset grid with $L \in \{1,2,3,4\}$, $p \in \{4,19\}$, $K \in \{5,25\}$, and "
            r"Poisson demand with mean $5$. Each row reports the mean relative gap to the best "
            r"heuristic found on each instance and the number of instances on which the learned "
            r"policy beats that best heuristic. Lower is better.}"
        ),
        r"\label{tab:fixed-cost-full-grid}",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Policy family & Mean relative gap (\%) & Better than best heuristic & Status \\",
        r"\midrule",
    ]

    for policy_id in ROW_ORDER:
        item = policies.get(policy_id)
        if item is None:
            continue
        lines.append(
            f"{DISPLAY_NAMES[policy_id]} & "
            f"{fmt_pct(item['mean_relative_gap_pct_vs_best_heuristic'])} & "
            f"{item['better_than_best_heuristic_count']}/{num_instances} & "
            f"{item['status']} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\par\medskip\footnotesize{Best trusted aggregate row: "
                f"{DISPLAY_NAMES[best_trusted_id]} with mean relative gap "
                f"{fmt_pct(best_trusted_item['mean_relative_gap_pct_vs_best_heuristic'])}\\% "
                f"and wins on {best_trusted_item['better_than_best_heuristic_count']}/{num_instances} instances. "
                r"$^{\dagger}$ The current NN categorical-quantity baseline remains provisional "
                r"and should be re-verified before publication claims rely on it.}"
            ),
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
