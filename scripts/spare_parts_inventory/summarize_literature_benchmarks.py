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

from common import (
    dumps_json,
    ensure_parent,
    get_kranenburg_reference_instances,
    get_literature_benchmark_catalog,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize literature-verified spare-parts benchmark scenarios."
    )
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _markdown(catalog: list[dict]) -> str:
    lines = [
        "## van Oers 2024",
        "",
        "| Scenario | AM Location | Policy | Base-Stock Levels | Reported Cost | Reported Readiness |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for scenario in catalog:
        for row in scenario["published_policy_results"]:
            lines.append(
                f"| `{scenario['name']}` | `{scenario['am_location']}` | "
                f"`{row['policy_name']}` | `{row['base_stock_levels']}` | "
                f"`{row['reported_cost_value']:.2f} ± {row['reported_cost_half_width']:.2f}` | "
                f"`{row['reported_readiness_percent']:.2f} ± {row['reported_readiness_half_width']:.3f}` |"
            )
    lines.extend(
        [
            "",
            "## Kranenburg 2006 Table 5.2",
            "",
            "| Instance | Varied Parameter | Value | Published `R1*` | Published `C1(R1*)` | Published `R3*` | Published `C3(R3*)` | Published Ratio |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for reference in get_kranenburg_reference_instances():
        lines.append(
            f"| `{reference['name']}` | `{reference['varied_parameter']}` | "
            f"`{reference['varied_value_label']}` | "
            f"`{reference['published_situation1_optimal_r']:.2f}` | "
            f"`{reference['published_situation1_cost']:.2f}` | "
            f"`{reference['published_situation3_optimal_r']:.2f}` | "
            f"`{reference['published_situation3_cost']:.2f}` | "
            f"`{reference['published_cost_ratio_situation1_over_situation3']:.2f}` |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    catalog = get_literature_benchmark_catalog()
    kranenburg_rows = get_kranenburg_reference_instances()
    payload = {
        "literature_benchmark_catalog": catalog,
        "kranenburg_table_5_2_rows": kranenburg_rows,
        "markdown": _markdown(catalog),
    }

    if parsed.output_json:
        output_path = Path(parsed.output_json)
        ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(dumps_json(payload))
    print()
    print(payload["markdown"])


if __name__ == "__main__":
    main()
