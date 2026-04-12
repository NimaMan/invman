import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from invman.problems.dual_sourcing.experiment_spec import EXPERIMENT_SPECS


DEFAULT_SUITE_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "dual_sourcing_gijs_structured_screening"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEFAULT_MD_PATH = REPORTS_DIR / "selected_policy_tables.md"
DEFAULT_TEX_PATH = REPORTS_DIR / "selected_policy_tables.tex"

EXPECTED_EXPEDITED_ORDER_COSTS = [105, 110]
EXPECTED_REGULAR_LEAD_TIMES = [2, 3, 4]

HEURISTIC_ROWS = [
    ("single_index", "single-index"),
    ("dual_index", "dual-index"),
    ("capped_dual_index", "capped dual-index"),
    ("tailored_base_surge", "tailored base-surge"),
]
POLICY_ROWS = [(spec["id"], spec["label"]) for spec in EXPERIMENT_SPECS]
ROW_ORDER = HEURISTIC_ROWS + POLICY_ROWS


def parse_args():
    parser = argparse.ArgumentParser(description="Render paper-style dual-sourcing tables.")
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
        expedited_cost = int(round(float(params["expedited_order_cost"])))
        regular_lead_time = int(params["regular_lead_time"])
        bucket = tables.setdefault(expedited_cost, {})

        column = {}
        heuristics = payload["heuristics"]["heuristics"]
        for key, _ in HEURISTIC_ROWS:
            entry = heuristics.get(key)
            if entry is not None:
                column[key] = float(entry["mean_cost"])
        for key, _ in POLICY_ROWS:
            entry = payload["learned_policies"].get(key)
            if entry is not None:
                column[key] = float(entry["evaluation"]["learned_policy"]["mean_cost"])
        bucket[regular_lead_time] = column
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


def _block_markdown(expedited_cost: int, lead_time_map: dict):
    available_columns = len(lead_time_map)
    best_by_lead_time = _best_rows_by_lead_time(lead_time_map)
    lines = [
        f"## c_e={expedited_cost}",
        "",
        f"Completed columns: `{available_columns}/{len(EXPECTED_REGULAR_LEAD_TIMES)}`",
        "",
        "| Policy / heuristic | " + " | ".join(f"l_r={lead_time}" for lead_time in EXPECTED_REGULAR_LEAD_TIMES) + " |",
        "| --- | " + " | ".join("---:" for _ in EXPECTED_REGULAR_LEAD_TIMES) + " |",
    ]
    for row_id, label in ROW_ORDER:
        row = [label]
        for lead_time in EXPECTED_REGULAR_LEAD_TIMES:
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


def _latex_table(expedited_cost: int, lead_time_map: dict):
    best_by_lead_time = _best_rows_by_lead_time(lead_time_map)
    incomplete = len(lead_time_map) < len(EXPECTED_REGULAR_LEAD_TIMES)
    column_spec = "|l||" + "|".join("c" for _ in EXPECTED_REGULAR_LEAD_TIMES) + "|"
    caption = (
        f"Dual-sourcing mean costs for the Gijsbrechts Figure 9 family with expedited order cost $c_e={expedited_cost}$ "
        f"across regular lead times $l_r \\in \\{{2,3,4\\}}$ under the structured soft-tree screening protocol."
    )
    if incomplete:
        caption += " Cells marked --- denote runs not yet completed."
    caption += " Bold entries mark the best value in each lead-time column."
    label = f"tab:dual-sourcing-ce{expedited_cost}"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        "Policy / heuristic & " + " & ".join(f"$l_r={lead_time}$" for lead_time in EXPECTED_REGULAR_LEAD_TIMES) + " \\\\",
        "\\hline",
    ]
    for idx, (row_id, label_text) in enumerate(ROW_ORDER):
        if idx == len(HEURISTIC_ROWS):
            lines.append("\\hline")
        cells = [label_text]
        for lead_time in EXPECTED_REGULAR_LEAD_TIMES:
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


def render_reports(payloads: list[dict], *, suite_root: Path | None = None):
    tables = _collect_tables(payloads)
    md_sections = [
        "# Dual-Sourcing Structured-Policy Tables",
        "",
        f"Suite root: `{suite_root or DEFAULT_SUITE_ROOT}`",
        "",
        "These tables are generated from per-instance JSON summaries. Missing cells denote instances that have not finished yet.",
        "",
    ]
    tex_sections = []
    for expedited_cost in EXPECTED_EXPEDITED_ORDER_COSTS:
        block = tables.get(expedited_cost, {})
        md_sections.append(_block_markdown(expedited_cost, block))
        md_sections.append("")
        tex_sections.append(_latex_table(expedited_cost, block))
        tex_sections.append("")
    return "\n".join(md_sections).rstrip() + "\n", "\n".join(tex_sections).rstrip() + "\n"


def main():
    args = parse_args()
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    payloads = _load_instance_payloads(args.suite_root / "instances")
    markdown, latex = render_reports(payloads, suite_root=args.suite_root)
    args.md_out.write_text(markdown, encoding="utf-8")
    args.tex_out.write_text(latex, encoding="utf-8")
    print(json.dumps({"md_out": str(args.md_out), "tex_out": str(args.tex_out), "instances": len(payloads)}, indent=2))


if __name__ == "__main__":
    main()
