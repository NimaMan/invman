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

from common import dumps_json, ensure_parent, get_literature_benchmark_families


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize the literature-backed benchmark families tracked for random_yield_inventory."
    )
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _fmt(values):
    if not values:
        return "-"
    return ", ".join(str(value) for value in values)


def _markdown(families: list[dict]) -> str:
    lines = [
        "| Name | Match | Access | Yield Model | Lead Times | Success p | Yield (mean, cv) pairs |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for family in families:
        lines.append(
            f"| `{family['name']}` | `{family['model_match']}` | `{family['access_level']}` | "
            f"`{family['yield_model']}` | `{_fmt(family['lead_times'])}` | "
            f"`{_fmt(family['success_probabilities'])}` | `{_fmt(family['yield_rate_mean_cv_pairs'])}` |"
        )
    return "\n".join(lines)


def main():
    parsed = parse_args()
    families = get_literature_benchmark_families()
    payload = {
        "num_families": len(families),
        "families": families,
        "markdown": _markdown(families),
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
