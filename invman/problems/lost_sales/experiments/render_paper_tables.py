import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SUITE_ROOT = (
    REPO_ROOT / "outputs" / "benchmarks" / "lost_sales_selected_paper_suite_scale20_rust_seed42"
)
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEFAULT_MD_PATH = REPORTS_DIR / "selected_policy_tables.md"
DEFAULT_TEX_PATH = REPORTS_DIR / "selected_policy_tables.tex"

EXPECTED_DEMANDS = ["Poisson", "Geometric"]
EXPECTED_SHORTAGE_COSTS = [4, 19]
EXPECTED_LEAD_TIMES = [2, 4, 6, 8, 10]

HEURISTIC_ROWS = [
    ("myopic1", "myopic1"),
    ("myopic2", "myopic2"),
    ("svbs", "SVBS"),
]
POLICY_ROWS = [
    ("linear_soft_gated_direct_quantity", "Linear soft-gated direct quantity"),
    ("linear_soft_gated_ordinal_quantity", "Linear soft-gated ordinal quantity"),
    ("soft_tree_depth1_linear_leaf", "Soft tree, depth-1 linear leaf"),
    ("soft_tree_depth2_linear_leaf", "Soft tree, depth-2 linear leaf"),
]
ROW_ORDER = HEURISTIC_ROWS + POLICY_ROWS


def parse_args():
    parser = argparse.ArgumentParser(description="Render paper-style vanilla lost-sales tables.")
    parser.add_argument("--suite_root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--md_out", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--tex_out", type=Path, default=DEFAULT_TEX_PATH)
    return parser.parse_args()


def _load_instance_payloads(instances_dir: Path):
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(instances_dir.glob("*.json"))]


def _collect_tables(payloads: list[dict]):
    tables = {}
    for payload in payloads:
        params = payload["params"]
        demand = params["demand_dist_name"]
        shortage_cost = int(round(float(params["shortage_cost"])))
        lead_time = int(params["lead_time"])
        block_key = (demand, shortage_cost)
        bucket = tables.setdefault(block_key, {})

        column = {}
        for key, _ in HEURISTIC_ROWS:
            entry = payload["heuristics"]["evaluation"].get(key)
            if entry is not None:
                column[key] = float(entry["mean_cost"])
        for key, _ in POLICY_ROWS:
            entry = payload["learned_policies"].get(key)
            if entry is not None:
                column[key] = float(entry["evaluation"]["learned_policy"]["mean_cost"])
        bucket[lead_time] = column
    return tables


def _format_value(value: float, *, bold: bool, latex: bool) -> str:
    text = f"{value:.4f}"
    if not bold:
        return text
    return f"\\textbf{{{text}}}" if latex else f"**{text}**"


def _best_rows_by_lead_time(lead_time_map: dict):
    best = {}
    for lead_time, row_values in lead_time_map.items():
        if not row_values:
            continue
        best_cost = min(row_values.values())
        best[lead_time] = {row_id for row_id, value in row_values.items() if abs(value - best_cost) < 1e-12}
    return best


def _block_markdown(demand: str, shortage_cost: int, lead_time_map: dict):
    available_columns = len(lead_time_map)
    best_by_lead_time = _best_rows_by_lead_time(lead_time_map)
    lines = [
        f"## {demand} Demand, p={shortage_cost}",
        "",
        f"Completed columns: `{available_columns}/{len(EXPECTED_LEAD_TIMES)}`",
        "",
        "| Policy / heuristic | " + " | ".join(f"L={lead_time}" for lead_time in EXPECTED_LEAD_TIMES) + " |",
        "| --- | " + " | ".join("---:" for _ in EXPECTED_LEAD_TIMES) + " |",
    ]
    for row_id, label in ROW_ORDER:
        row = [label]
        for lead_time in EXPECTED_LEAD_TIMES:
            value = lead_time_map.get(lead_time, {}).get(row_id)
            if value is None:
                row.append("---")
            else:
                row.append(
                    _format_value(
                        value,
                        bold=row_id in best_by_lead_time.get(lead_time, set()),
                        latex=False,
                    )
                )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _latex_table(demand: str, shortage_cost: int, lead_time_map: dict):
    best_by_lead_time = _best_rows_by_lead_time(lead_time_map)
    incomplete = len(lead_time_map) < len(EXPECTED_LEAD_TIMES)
    column_spec = "|l||" + "|".join("c" for _ in EXPECTED_LEAD_TIMES) + "|"
    caption = (
        f"Vanilla lost-sales mean costs for {demand.lower()} demand with shortage cost $p={shortage_cost}$ "
        f"across lead times $L \\in \\{{2,4,6,8,10\\}}$ under the selected $2000$-iteration / population-$64$ "
        "CMA-ES protocol with policy-side state scaling $20$."
    )
    if incomplete:
        caption += " Cells marked --- denote runs not yet completed."
    caption += " Bold entries mark the best value in each lead-time column."
    label = f"tab:lost-sales-{demand.lower()}-p{shortage_cost}-selected-scale20"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        "Policy / heuristic & " + " & ".join(f"$L={lead_time}$" for lead_time in EXPECTED_LEAD_TIMES) + " \\\\",
        "\\hline",
    ]
    for idx, (row_id, label_text) in enumerate(ROW_ORDER):
        if idx == len(HEURISTIC_ROWS):
            lines.append("\\hline")
        cells = [label_text]
        for lead_time in EXPECTED_LEAD_TIMES:
            value = lead_time_map.get(lead_time, {}).get(row_id)
            if value is None:
                cells.append("---")
            else:
                cells.append(
                    _format_value(
                        value,
                        bold=row_id in best_by_lead_time.get(lead_time, set()),
                        latex=True,
                    )
                )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def render_reports(payloads: list[dict]):
    tables = _collect_tables(payloads)
    md_sections = [
        "# Vanilla Lost-Sales Selected-Policy Tables",
        "",
        f"Suite root: `{DEFAULT_SUITE_ROOT}`",
        "",
        "These tables are generated from per-instance JSON summaries. Missing cells denote instances that have not finished yet.",
        "",
    ]
    tex_sections = []
    for demand in EXPECTED_DEMANDS:
        for shortage_cost in EXPECTED_SHORTAGE_COSTS:
            block = tables.get((demand, shortage_cost), {})
            md_sections.append(_block_markdown(demand, shortage_cost, block))
            md_sections.append("")
            tex_sections.append(_latex_table(demand, shortage_cost, block))
            tex_sections.append("")
    return "\n".join(md_sections).rstrip() + "\n", "\n".join(tex_sections).rstrip() + "\n"


def main():
    args = parse_args()
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    payloads = _load_instance_payloads(args.suite_root / "instances")
    markdown, latex = render_reports(payloads)
    args.md_out.write_text(markdown, encoding="utf-8")
    args.tex_out.write_text(latex, encoding="utf-8")
    print(json.dumps({"md_out": str(args.md_out), "tex_out": str(args.tex_out), "instances": len(payloads)}, indent=2))


if __name__ == "__main__":
    main()
