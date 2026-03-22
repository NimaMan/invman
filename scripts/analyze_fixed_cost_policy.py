import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from invman.nn.es_module import ESModule
from invman.env.lost_sales import build_env_from_args
from invman.problems.lost_sales_fixed_order_cost.heuristics import get_modified_s_s_q_order_quantity


def get_args():
    parser = argparse.ArgumentParser(description="Inspect a trained fixed-cost policy.")
    parser.add_argument("--model_dir", required=True, help="Directory containing model_config.json and model_params.torch.")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--horizon", default=50000, type=int)
    return parser.parse_args()


def build_reference_args(horizon: int):
    return SimpleNamespace(
        demand_rate=5.0,
        lead_time=4,
        horizon=horizon,
        max_order_size=50,
        inventory_upper_bound=200,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=5.0,
        demand_dist_name="Poisson",
        track_demand=True,
        warm_up_periods_ratio=0.2,
    )


def summarize_actions(actions):
    counts = Counter(actions)
    total = len(actions)
    return {
        "top_actions": counts.most_common(10),
        "unique_actions": len(counts),
        "zero_fraction": counts[0] / total if total else 0.0,
        "mean_action": float(np.mean(actions)) if actions else 0.0,
    }


def rollout_model(model, args, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = build_env_from_args(args, track_demand=True)
    state = env.norm_state
    actions = []
    done = False
    while not done:
        action = int(model(torch.as_tensor(state, dtype=torch.float32)))
        actions.append(action)
        state, _, done = env.step(action)
    return env.avg_total_cost, summarize_actions(actions)


def rollout_modified_s_s_q(args, seed: int, params=None):
    params = {"s": 22, "S": 29, "q": 8} if params is None else params
    np.random.seed(seed)
    env = build_env_from_args(args, track_demand=True)
    actions = []
    done = False
    while not done:
        action = get_modified_s_s_q_order_quantity(
            inventory_position=env.inventory_position,
            s=params["s"],
            S=params["S"],
            q=params["q"],
            max_order_size=env.max_order_size,
        )
        actions.append(action)
        _, _, done = env.step(action)
    return env.avg_total_cost, summarize_actions(actions)


def coarse_grid_action_histogram(model):
    vals = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    counts = Counter()
    for x0 in vals:
        for x1 in vals:
            for x2 in vals:
                for x3 in vals:
                    state = torch.tensor(np.array([x0, x1, x2, x3], dtype=np.float32) / 50.0)
                    counts[int(model(state))] += 1
    return counts.most_common(15), len(counts)


def main():
    args = get_args()
    model = ESModule.load(args.model_dir)
    ref_args = build_reference_args(args.horizon)

    model_cost, model_actions = rollout_model(model, ref_args, seed=args.seed)
    heuristic_cost, heuristic_actions = rollout_modified_s_s_q(ref_args, seed=args.seed)
    grid_top_actions, grid_unique_actions = coarse_grid_action_histogram(model)

    payload = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "seed": args.seed,
        "horizon": args.horizon,
        "model_rollout": {
            "avg_cost": model_cost,
            "action_summary": model_actions,
        },
        "modified_s_s_q_rollout": {
            "avg_cost": heuristic_cost,
            "action_summary": heuristic_actions,
        },
        "coarse_state_grid": {
            "top_actions": grid_top_actions,
            "unique_actions": grid_unique_actions,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
