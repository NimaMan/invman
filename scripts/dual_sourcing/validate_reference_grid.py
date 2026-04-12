import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.problems.dual_sourcing.benchmark import benchmark_reference_instance
from invman.problems.dual_sourcing.reference_instances import (
    build_reference_args,
    get_benchmark_reference,
    get_reference_instance,
    list_reference_instances,
)


def _heuristic_ranking(gap_by_policy):
    return [
        policy
        for policy, _ in sorted(
            gap_by_policy.items(),
            key=lambda item: (float(item[1]), item[0]),
        )
    ]


def main():
    benchmark_reference = get_benchmark_reference()
    results = {}
    for name in list_reference_instances():
        args = build_reference_args(name)
        args.rollout_backend = "rust"
        args.eval_seeds = 2
        payload = benchmark_reference_instance(args)
        reference_instance = get_reference_instance(name)
        published_gaps = reference_instance.literature_values.get("published_optimality_gap_pct", {})
        bounded_dp_cost = float(payload["bounded_dp"]["average_cost"])
        repo_gaps = {
            policy_name: 100.0 * (float(summary["mean_cost"]) / bounded_dp_cost - 1.0)
            for policy_name, summary in payload["heuristics"].items()
        }
        payload["reference"] = {
            "name": name,
            "source": benchmark_reference.source,
            "url": benchmark_reference.url,
            "published_optimality_gap_pct": published_gaps,
            "published_heuristic_ranking": list(reference_instance.expected_ranking),
        }
        payload["repo_optimality_gap_pct"] = repo_gaps
        payload["repo_heuristic_ranking"] = _heuristic_ranking(repo_gaps)
        payload["repo_gap_minus_paper_pct"] = {
            policy_name: float(repo_gaps[policy_name]) - float(published_gaps[policy_name])
            for policy_name in published_gaps
            if policy_name in repo_gaps
        }
        results[name] = payload
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
