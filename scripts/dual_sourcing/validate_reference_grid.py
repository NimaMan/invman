"""
Validate the dual-sourcing reference grid against the published Figure-9 gaps.

OBJECTIVE
    Reproduce, from the Rust backend, the heuristic ranking and (optionally) the
    optimality gaps that Gijsbrechts et al. (2022) Figure 9 reports for the six
    dual-sourcing rows, to confirm the migrated problem is faithful. The published
    metric is the relative optimality gap; the strongest heuristic is
    capped_dual_index (~0-0.11%).

ALGORITHM
    For each reference instance:
      1. Compute single/dual/capped-dual-index + tailored-base-surge costs on a
         fixed demand path via the Rust search bindings.
      2. (opt-in, --with_optimal_dp) Solve the bounded average-cost DP optimum.
         This is slow on l_r in {3,4}, so it is OFF by default; when off, the best
         heuristic (capped_dual_index ~ 0% published gap) is the optimal proxy.
      3. Report the repo heuristic ranking, repo optimality gaps vs the DP optimum
         (or vs the best-heuristic proxy), and the published Figure-9 gaps for
         side-by-side comparison.
"""

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dual_sourcing_benchmark_lib as lib


def _heuristic_ranking(gap_by_policy):
    return [
        policy
        for policy, _ in sorted(
            gap_by_policy.items(),
            key=lambda item: (float(item[1]), item[0]),
        )
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the dual-sourcing reference grid against the published Figure-9 gaps.")
    parser.add_argument("--references", nargs="+", default=None, help="Subset of reference instances (default: all six).")
    parser.add_argument("--search_horizon", type=int, default=6000)
    parser.add_argument(
        "--with_optimal_dp",
        action="store_true",
        help="Also solve the bounded average-cost DP optimum (slow on l_r=3,4).",
    )
    return parser.parse_args()


def main():
    parsed = parse_args()
    names = parsed.references or lib.build_grid_instances()
    if parsed.references is None:
        names = [i["name"] for i in lib.build_grid_instances()]

    results = {}
    for name in names:
        args = lib.build_reference_args(name)
        args.rollout_backend = "rust"
        inst = lib._reference_instance(name)
        published_gaps = dict(inst.get("published_optimality_gap_pct", {}))
        published_gaps.pop("source", None)
        published_gaps.pop("url", None)

        heuristics = lib.evaluate_default_heuristics(args, horizon=parsed.search_horizon)
        best_heuristic_name, best_heuristic_cost = lib.best_heuristic(heuristics)

        optimal = lib.bounded_dp_optimal(args) if parsed.with_optimal_dp else {
            "mean_cost": None, "available": False, "source": "skipped (use --with_optimal_dp)"
        }
        # Optimal proxy for the gap denominator: DP optimum if available, else the
        # best heuristic (capped_dual_index, ~0% published gap).
        denom = optimal["mean_cost"] if optimal.get("mean_cost") is not None else best_heuristic_cost
        denom_source = "bounded_dp" if optimal.get("mean_cost") is not None else "best_heuristic_proxy"

        repo_gaps = {}
        if denom is not None:
            for policy_name, summary in heuristics.items():
                if summary.get("mean_cost") is not None:
                    repo_gaps[policy_name] = 100.0 * (float(summary["mean_cost"]) / denom - 1.0)

        results[name] = {
            "reference": name,
            "source": inst["source"],
            "url": inst["url"],
            "heuristics": heuristics,
            "best_heuristic_name": best_heuristic_name,
            "best_heuristic_cost": best_heuristic_cost,
            "optimal": optimal,
            "repo_gap_denominator": denom,
            "repo_gap_denominator_source": denom_source,
            "repo_optimality_gap_pct": repo_gaps,
            "repo_heuristic_ranking": _heuristic_ranking(repo_gaps) if repo_gaps else [],
            "published_optimality_gap_pct": published_gaps,
            "repo_gap_minus_paper_pct": {
                policy_name: float(repo_gaps[policy_name]) - float(published_gaps[policy_name])
                for policy_name in published_gaps
                if policy_name in repo_gaps
            },
        }
        print(
            f"{name}: best_heuristic={best_heuristic_name} ({best_heuristic_cost:.3f}); "
            f"repo_ranking={results[name]['repo_heuristic_ranking']}"
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
