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
    benchmark_initial_state,
    dumps_json,
    evaluate_echelon_base_stock_policy,
    ensure_parent,
    get_primary_reference,
    get_reference,
    published_cost,
    search_best_echelon_base_stock,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a literature-backed one_warehouse_multi_retailer reference instance against the published Kaynov et al. heuristic benchmark rows."
    )
    parser.add_argument("--reference_name", default="primary")
    parser.add_argument("--search_replications", type=int, default=256)
    parser.add_argument("--benchmark_replications", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _resolve_reference(name: str) -> dict:
    if name == "primary":
        return get_primary_reference()
    return get_reference(name)


def _published_vs_repo_row(published: dict, repo_eval: dict) -> dict:
    target_cost = published_cost(published)
    repo_cost = float(repo_eval["mean_cost"])
    return {
        "published_cost": float(target_cost),
        "published_standard_error": float(published["standard_error"]),
        "repo_cost": repo_cost,
        "absolute_gap_cost": float(repo_cost - target_cost),
        "relative_gap_percent": float(100.0 * (repo_cost - target_cost) / target_cost),
    }


def _markdown(payload: dict) -> str:
    comparisons = payload["comparisons"]
    lines = [
        "| Reference Metric | Value |",
        "| --- | --- |",
        f"| `reference_name` | `{payload['reference']['name']}` |",
        f"| `literature_verified` | `{payload['reference']['literature_verified']}` |",
        f"| `benchmark_periods` | `{payload['reference']['benchmark_periods']}` |",
        f"| `benchmark_replications` | `{payload['benchmark_replications']}` |",
        f"| `initial_state_rule` | `{payload['initial_state']['initial_state_rule']}` |",
        "",
        "| Policy | Published Cost | Repo Cost | Absolute Gap | Relative Gap % | Search Params |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| `echelon_base_stock_proportional` | `{comparisons['proportional']['published_cost']:.3f}` | `{comparisons['proportional']['repo_cost']:.3f}` | `{comparisons['proportional']['absolute_gap_cost']:.3f}` | `{comparisons['proportional']['relative_gap_percent']:.3f}` | `[{payload['heuristics']['proportional']['warehouse_base_stock_level']}, {', '.join(str(v) for v in payload['heuristics']['proportional']['retailer_base_stock_levels'])}]` |",
        f"| `echelon_base_stock_min_shortage` | `{comparisons['min_shortage']['published_cost']:.3f}` | `{comparisons['min_shortage']['repo_cost']:.3f}` | `{comparisons['min_shortage']['absolute_gap_cost']:.3f}` | `{comparisons['min_shortage']['relative_gap_percent']:.3f}` | `[{payload['heuristics']['min_shortage']['warehouse_base_stock_level']}, {', '.join(str(v) for v in payload['heuristics']['min_shortage']['retailer_base_stock_levels'])}]` |",
    ]
    return "\n".join(lines)


def main():
    parsed = parse_args()
    reference = _resolve_reference(parsed.reference_name)
    benchmark_replications = (
        int(reference["benchmark_replications"])
        if parsed.benchmark_replications is None
        else int(parsed.benchmark_replications)
    )
    proportional = search_best_echelon_base_stock(
        reference,
        allocation_policy="proportional",
        replications=int(parsed.search_replications),
        seed=int(parsed.seed),
    )
    min_shortage = search_best_echelon_base_stock(
        reference,
        allocation_policy="min_shortage",
        replications=int(parsed.search_replications),
        seed=int(parsed.seed),
    )

    proportional_eval = evaluate_echelon_base_stock_policy(
        reference,
        warehouse_base_stock_level=proportional["warehouse_base_stock_level"],
        retailer_base_stock_levels=proportional["retailer_base_stock_levels"],
        allocation_policy="proportional",
        replications=benchmark_replications,
        seed=int(parsed.seed),
    )
    min_shortage_eval = evaluate_echelon_base_stock_policy(
        reference,
        warehouse_base_stock_level=min_shortage["warehouse_base_stock_level"],
        retailer_base_stock_levels=min_shortage["retailer_base_stock_levels"],
        allocation_policy="min_shortage",
        replications=benchmark_replications,
        seed=int(parsed.seed),
    )

    payload = {
        "reference": reference,
        "initial_state": benchmark_initial_state(reference),
        "search_replications": int(parsed.search_replications),
        "benchmark_replications": benchmark_replications,
        "seed": int(parsed.seed),
        "heuristics": {
            "proportional": proportional_eval,
            "min_shortage": min_shortage_eval,
        },
        "search_results": {
            "proportional": proportional,
            "min_shortage": min_shortage,
        },
        "comparisons": {
            "proportional": _published_vs_repo_row(
                reference["published_proportional_benchmark"],
                proportional_eval,
            ),
            "min_shortage": _published_vs_repo_row(
                reference["published_min_shortage_benchmark"],
                min_shortage_eval,
            ),
        },
    }
    payload["markdown"] = _markdown(payload)

    if parsed.output_json:
        output_path = Path(parsed.output_json)
        ensure_parent(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(dumps_json(payload))
    print()
    print(payload["markdown"])


if __name__ == "__main__":
    main()
