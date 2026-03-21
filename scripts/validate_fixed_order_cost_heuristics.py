import argparse
import json
import sys
from copy import copy
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.problems.lost_sales_fixed_order_cost import (
    build_reference_args,
    evaluate_policy_across_seeds,
    get_reference_instance,
    search_best_modified_s_s_q_policy,
    search_best_s_nq_policy,
    search_best_s_s_policy,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Validate fixed-order-cost heuristics on a starter instance.")
    parser.add_argument(
        "--reference_instance",
        default="starter_l4_p4_k5_poisson5",
        help="Named reference instance from the fixed-order-cost problem package.",
    )
    parser.add_argument("--search_horizon", default=None, type=int, help="Override the search horizon.")
    parser.add_argument("--eval_horizon", default=None, type=int, help="Override the evaluation horizon.")
    parser.add_argument("--eval_seeds", default=None, type=int, help="Override the number of evaluation seeds.")
    parser.add_argument("--position_upper_bound", default=None, type=int, help="Override the search upper bound for s and S.")
    parser.add_argument("--search_seed", default=None, type=int, help="Override the search seed.")
    parser.add_argument("--top_k_s_s_pairs", default=None, type=int, help="Override the number of s,S pairs used for the modified search.")
    parser.add_argument("--q_window", default=None, type=int, help="Override the q search window around the paper heuristic.")
    return parser


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    reference = get_reference_instance(cli_args.reference_instance)
    args = build_reference_args(cli_args.reference_instance)

    search_horizon = reference["search"]["search_horizon"] if cli_args.search_horizon is None else cli_args.search_horizon
    eval_horizon = reference["evaluation"]["eval_horizon"] if cli_args.eval_horizon is None else cli_args.eval_horizon
    eval_seeds = reference["evaluation"]["eval_seeds"] if cli_args.eval_seeds is None else cli_args.eval_seeds
    position_upper_bound = (
        reference["search"]["position_upper_bound"]
        if cli_args.position_upper_bound is None
        else cli_args.position_upper_bound
    )
    search_seed = reference["search"]["search_seed"] if cli_args.search_seed is None else cli_args.search_seed
    top_k_s_s_pairs = (
        reference["search"]["top_k_s_s_pairs"]
        if cli_args.top_k_s_s_pairs is None
        else cli_args.top_k_s_s_pairs
    )
    q_window = reference["search"]["q_window"] if cli_args.q_window is None else cli_args.q_window

    search_args = copy(args)
    search_args.horizon = search_horizon
    eval_args = copy(args)
    eval_args.horizon = eval_horizon

    s_s_summary = search_best_s_s_policy(
        args=search_args,
        seed=search_seed,
        horizon=search_horizon,
        position_upper_bound=position_upper_bound,
        top_k=top_k_s_s_pairs,
    )
    s_nq_summary = search_best_s_nq_policy(
        args=search_args,
        seed=search_seed,
        horizon=search_horizon,
        position_upper_bound=position_upper_bound,
    )
    modified_search = search_best_modified_s_s_q_policy(
        args=search_args,
        seed=search_seed,
        horizon=search_horizon,
        position_upper_bound=position_upper_bound,
        top_k_s_s_pairs=top_k_s_s_pairs,
        q_window=q_window,
        s_s_summary=s_s_summary,
    )

    evaluations = {
        "s_s": evaluate_policy_across_seeds(
            args=eval_args,
            policy_name="s_s",
            params=s_s_summary.best_result.params,
            num_seeds=eval_seeds,
            horizon=eval_horizon,
            track_demand=True,
        ),
        "s_nq": evaluate_policy_across_seeds(
            args=eval_args,
            policy_name="s_nq",
            params=s_nq_summary.best_result.params,
            num_seeds=eval_seeds,
            horizon=eval_horizon,
            track_demand=True,
        ),
        "modified_s_s_q": evaluate_policy_across_seeds(
            args=eval_args,
            policy_name="modified_s_s_q",
            params=modified_search["modified_policy"].best_result.params,
            num_seeds=eval_seeds,
            horizon=eval_horizon,
            track_demand=True,
        ),
    }

    payload = {
        "reference_instance": reference["name"],
        "description": reference["description"],
        "params": reference["params"],
        "search_config": {
            "search_horizon": search_horizon,
            "position_upper_bound": position_upper_bound,
            "search_seed": search_seed,
            "top_k_s_s_pairs": top_k_s_s_pairs,
            "q_window": q_window,
        },
        "search_results": {
            "s_s": s_s_summary.to_dict(),
            "s_nq": s_nq_summary.to_dict(),
            "modified_s_s_q": modified_search["modified_policy"].to_dict(),
        },
        "evaluation": evaluations,
        "ranking_check": {
            "modified_not_worse_than_s_s": evaluations["modified_s_s_q"]["mean_cost"] <= evaluations["s_s"]["mean_cost"] + 1e-9,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
