import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    dumps_json,
    ensure_parent,
    get_kranenburg_exact_summary,
    get_kranenburg_reference_instances,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the Kranenburg (2006) Chapter 5 lateral-transshipment exact benchmark carried under spare_parts_inventory."
    )
    parser.add_argument(
        "--instance_name",
        default=None,
        help="Optional named Table 5.2 row. Defaults to the published base case.",
    )
    parser.add_argument(
        "--all_rows",
        action="store_true",
        help="Validate every carried Table 5.2 row instead of only the selected instance.",
    )
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _markdown(summary_rows: list[dict]) -> str:
    lines = [
        "| Instance | Varied Parameter | Value | `R1*` | `C1(R1*)` | `R3*` | `C3(R3*)` | Ratio | Match |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        reference = row["reference_instance"]
        evaluation = row["evaluation"]
        comparison = row["published_table_comparison"]
        lines.append(
            "| "
            f"`{reference['name']}` | "
            f"`{reference['varied_parameter']}` | "
            f"`{reference['varied_value_label']}` | "
            f"`{evaluation['situation1']['optimal_r']:.3f}` | "
            f"`{evaluation['situation1']['total_cost']:.3f}` | "
            f"`{evaluation['situation3']['optimal_r']:.3f}` | "
            f"`{evaluation['situation3']['total_cost']:.3f}` | "
            f"`{evaluation['cost_ratio_situation1_over_situation3']:.3f}` | "
            f"`{comparison['all_within_tolerance']}` |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    if parsed.all_rows:
        rows = [
            get_kranenburg_exact_summary(reference["name"])
            for reference in get_kranenburg_reference_instances()
        ]
    else:
        rows = [get_kranenburg_exact_summary(parsed.instance_name)]

    payload = {
        "validated_rows": rows,
        "all_rows_match_published_table": all(
            row["published_table_comparison"]["all_within_tolerance"] for row in rows
        ),
        "num_rows": len(rows),
    }
    payload["markdown"] = _markdown(rows)

    if parsed.output_json:
        output_path = Path(parsed.output_json)
        ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(dumps_json(payload))
    print()
    print(payload["markdown"])


if __name__ == "__main__":
    main()
