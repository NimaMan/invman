#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

import invman_rust


PROBLEM_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROBLEM_ROOT / "results"
DEFAULT_DAYS = 30
DEFAULT_PATHS = 4000
DEFAULT_SEED = 20260406


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Hormuz Strait Month-Ahead Brent Scenario Simulation")
    lines.append("")
    lines.append(f"Analysis date: `{report['analysis_date']}`")
    lines.append(
        f"Latest observed Brent close used as the starting point: `{report['latest_observed_brent_price_usd_per_bbl']:.2f}` on `{report['latest_observed_close_date']}`"
    )
    lines.append(
        f"EIA month-ahead anchor: Brent stays above `{report['eia_next_two_month_floor_brent_usd_per_bbl']:.2f}` per barrel over the next two months."
    )
    lines.append(
        f"EIA second-quarter average anchor: `{report['eia_q2_2026_average_brent_usd_per_bbl']:.2f}` per barrel."
    )
    lines.append("")
    lines.append("## Scenario Summary")
    lines.append("")
    lines.append(
        "| Scenario | Day 30 Mean | P10 | P50 | P90 | Monthly Mean | Avg Tightness (mb/d) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for scenario in report["scenarios"]:
        lines.append(
            "| "
            + scenario["label"]
            + f" | {scenario['day_30_mean_brent_price_usd_per_bbl']:.2f}"
            + f" | {scenario['day_30_p10_brent_price_usd_per_bbl']:.2f}"
            + f" | {scenario['day_30_p50_brent_price_usd_per_bbl']:.2f}"
            + f" | {scenario['day_30_p90_brent_price_usd_per_bbl']:.2f}"
            + f" | {scenario['monthly_average_mean_brent_price_usd_per_bbl']:.2f}"
            + f" | {scenario['mean_effective_tightness_million_bpd']:.2f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for scenario in report["scenarios"]:
        lines.append(f"### {scenario['label']}")
        lines.append("")
        lines.append(scenario["description"])
        lines.append("")
        lines.append(
            f"Day 30 mean Brent: `{scenario['day_30_mean_brent_price_usd_per_bbl']:.2f}`"
            f" with an 80% band of `{scenario['day_30_p10_brent_price_usd_per_bbl']:.2f}` to"
            f" `{scenario['day_30_p90_brent_price_usd_per_bbl']:.2f}`."
        )
        lines.append(
            f"Peak mean Brent in the simulation month: `{scenario['peak_mean_brent_price_usd_per_bbl']:.2f}`"
            f" on day `{scenario['peak_mean_price_day']}`."
        )
        lines.append(
            f"Average effective tightness: `{scenario['mean_effective_tightness_million_bpd']:.2f}` million b/d."
        )
        lines.append("")
    lines.append("## Model Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        "The scenario engine uses the checked-in 2024 Hormuz flow weights, the EIA daily prices page dated April 6, 2026, the March 2026 EIA STEO, and the OPEC+ April 5, 2026 release."
    )
    lines.append("")
    return "\n".join(lines)


def write_daily_csv(report: dict, path: Path) -> None:
    fieldnames = [
        "scenario_id",
        "scenario_label",
        "day_index",
        "mean_brent_price_usd_per_bbl",
        "p10_brent_price_usd_per_bbl",
        "p50_brent_price_usd_per_bbl",
        "p90_brent_price_usd_per_bbl",
        "closure_fraction",
        "blocked_flow_million_bpd",
        "rerouted_flow_million_bpd",
        "reserve_release_million_bpd",
        "floating_storage_release_million_bpd",
        "non_hormuz_supply_response_million_bpd",
        "inventory_buffer_draw_million_bpd",
        "effective_tightness_million_bpd",
        "target_price_usd_per_bbl",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in report["scenarios"]:
            for day in scenario["daily"]:
                writer.writerow(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "scenario_label": scenario["label"],
                        **day,
                    }
                )


def verify_monotone_severity(report: dict) -> None:
    day_30_means = [
        scenario["day_30_mean_brent_price_usd_per_bbl"] for scenario in report["scenarios"]
    ]
    if any(
        later <= earlier
        for earlier, later in zip(day_30_means, day_30_means[1:])
    ):
        raise RuntimeError(
            "Scenario severity ordering check failed: day-30 means are not strictly increasing."
        )


def main() -> None:
    ensure_results_dir()
    report = invman_rust.hormuz_strait_month_ahead_price_scenarios(
        days=DEFAULT_DAYS,
        paths=DEFAULT_PATHS,
        seed=DEFAULT_SEED,
    )
    verify_monotone_severity(report)

    analysis_date = report["analysis_date"]
    json_path = RESULTS_DIR / f"month_ahead_price_scenarios_{analysis_date}.json"
    markdown_path = RESULTS_DIR / f"month_ahead_price_scenarios_{analysis_date}.md"
    csv_path = RESULTS_DIR / f"month_ahead_price_paths_{analysis_date}.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    write_daily_csv(report, csv_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
