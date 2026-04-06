#!/usr/bin/env python3
"""Collect fixed-cost ordinal scale-sweep results into a markdown table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RUN_RE = re.compile(
    r"fixed_cost_exact_ordinal_scale_(?:sweep|proxy|local)_s(?P<scale>\d+)_p(?P<pop>\d+)_seed\d+"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs" / "benchmarks",
    )
    parser.add_argument(
        "--write-markdown",
        type=Path,
        default=None,
        help="Optional path to write the markdown table.",
    )
    return parser.parse_args()


def iter_records(outputs_root: Path):
    for run_dir in sorted(outputs_root.glob("fixed_cost_exact_ordinal_scale_*_s*_p*_seed42")):
        match = RUN_RE.fullmatch(run_dir.name)
        if not match:
            continue
        result_dir = run_dir / "results"
        if not result_dir.exists():
            continue
        for path in sorted(result_dir.glob("*.json")):
            if path.name.startswith("status_"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            learned = payload["evaluation"]["learned_policy"]
            yield {
                "run_tag": run_dir.name,
                "scale": int(match.group("scale")),
                "population": int(match.group("pop")),
                "mean_cost": float(learned["mean_cost"]),
                "std_cost": float(learned["std_cost"]),
                "num_seeds": int(learned["num_seeds"]),
                "result_path": path,
            }


def build_markdown(records: list[dict]) -> str:
    if not records:
        return "No completed sweep results found.\n"

    lines = [
        "# Fixed-Cost Ordinal Scale Sweep",
        "",
        "| Population | Scale | Mean cost | Std. dev. | Seeds | Result |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in sorted(records, key=lambda item: (item["population"], item["scale"])):
        lines.append(
            "| {population} | {scale} | {mean_cost:.4f} | {std_cost:.4f} | {num_seeds} | `{result_path}` |".format(
                **record
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = list(iter_records(args.outputs_root))
    markdown = build_markdown(records)
    if args.write_markdown is not None:
        args.write_markdown.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
