import argparse
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import invman_rust

from invman.policy import Policy
from invman.rollout_fitness import get_model_fitness


def get_args():
    parser = argparse.ArgumentParser(description="Inspect a trained fixed-cost policy.")
    parser.add_argument("--model_dir", required=True, help="Directory containing policy_artifact.json and model_params.npy.")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--horizon", default=50000, type=int)
    parser.add_argument("--trace_horizon", default=200, type=int, help="Number of deterministic periods to trace for the Rust heuristic.")
    parser.add_argument("--trace_rows", default=20, type=int, help="Number of leading trace rows to include in the JSON payload.")
    parser.add_argument("--state_features", default="pipeline", help="State feature mode expected by the model.")
    parser.add_argument("--output_json", default=None, help="Optional path for persisting the diagnostic JSON payload.")
    return parser.parse_args()


def build_reference_args(horizon: int, state_features: str):
    return SimpleNamespace(
        problem="lost_sales_fixed_order_cost",
        demand_rate=5.0,
        lead_time=4,
        horizon=horizon,
        max_order_size=50,
        one_hot_inventory_upper_bound=200,
        holding_cost=1.0,
        shortage_cost=4.0,
        procurement_cost=0.0,
        fixed_order_cost=5.0,
        demand_dist_name="Poisson",
        track_demand=True,
        warm_up_periods_ratio=0.2,
        state_features=state_features,
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


def summarize_trace(trace_rows):
    all_actions = [int(row["order_quantity"]) for row in trace_rows]
    active_actions = [
        int(row["order_quantity"])
        for row in trace_rows
        if bool(row.get("active_after_warmup", True))
    ]
    return {
        "all_periods": summarize_actions(all_actions),
        "active_after_warmup": summarize_actions(active_actions),
    }


def _deterministic_trace_demands(args, trace_horizon):
    demand_value = max(0, int(round(float(args.demand_rate))))
    bounded_trace_horizon = max(1, min(int(trace_horizon), int(args.horizon)))
    return [demand_value] * bounded_trace_horizon, demand_value, bounded_trace_horizon


def _rust_policy_trace(model, args, *, trace_horizon: int, trace_rows: int):
    demands, demand_value, bounded_trace_horizon = _deterministic_trace_demands(
        args,
        trace_horizon,
    )
    common = dict(
        flat_params=model.get_model_flat_params().tolist(),
        input_dim=int(model.input_dim),
        current_inventory=0,
        lead_time_orders=[0] * int(args.lead_time),
        demands=demands,
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        procurement_cost=float(args.procurement_cost),
        fixed_order_cost=float(args.fixed_order_cost),
        warm_up_periods_ratio=float(args.warm_up_periods_ratio),
        state_normalizer=str(model.state_normalizer),
        state_scale=model.state_scale,
    )
    if model.backbone == "soft_tree":
        payload = invman_rust.lost_sales_soft_tree_trace_from_demands(
            depth=int(model.depth),
            temperature=float(model.temperature),
            split_type=str(model.split_type),
            leaf_type=str(model.leaf_type),
            policy_max_quantity=model.max_order_size,
            **common,
        )
    elif model.backbone == "linear":
        payload = invman_rust.lost_sales_linear_trace_from_demands(
            output_dim=int(model.output_dim),
            policy_max_quantity=model.max_order_size,
            policy_head=str(model.action_output_mode),
            **common,
        )
    elif model.backbone == "nn":
        payload = invman_rust.lost_sales_nn_trace_from_demands(
            hidden_dims=[int(width) for width in model.hidden_dim],
            output_dim=int(model.output_dim),
            policy_max_quantity=model.max_order_size,
            activation=str(model.activation_name),
            policy_head=str(model.action_output_mode),
            **common,
        )
    else:
        raise NotImplementedError(f"Unsupported policy backbone '{model.backbone}'")
    trace = list(payload["trace"])
    return {
        "note": "stochastic cost uses the Rust rollout protocol; trace sample uses a deterministic rounded-mean demand path",
        "trace_mean_cost": float(payload["mean_cost"]),
        "trace_demands": {
            "source": "deterministic_rounded_mean_demand",
            "horizon": bounded_trace_horizon,
            "value": demand_value,
        },
        "trace_summary": summarize_trace(trace),
        "trace_head": trace[: max(0, int(trace_rows))],
    }


def rollout_model(model, args, seed: int, *, trace_horizon: int, trace_rows: int):
    reward, _ = get_model_fitness(model, args, seed=seed)
    return -float(reward), _rust_policy_trace(
        model,
        args,
        trace_horizon=trace_horizon,
        trace_rows=trace_rows,
    )


def rollout_modified_s_s_q(args, seed: int, *, trace_horizon: int, trace_rows: int, params=None):
    search_kwargs = dict(
        demand_kind=args.demand_dist_name,
        demand_rate=float(args.demand_rate),
        demand_lambda_low=3.0,
        demand_lambda_high=7.0,
        demand_p00=0.9,
        demand_p11=0.9,
        lead_time=int(args.lead_time),
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        procurement_cost=float(args.procurement_cost),
        fixed_order_cost=float(args.fixed_order_cost),
        max_order_size=int(args.max_order_size),
        position_upper_bound=int(max(args.one_hot_inventory_upper_bound, args.lead_time * args.max_order_size)),
        horizon=int(args.horizon),
        seed=int(seed),
        warm_up_periods_ratio=float(args.warm_up_periods_ratio),
        top_k=1,
    )
    if hasattr(invman_rust, "lost_sales_fixed_heuristics_all_detailed"):
        heuristic_payload = invman_rust.lost_sales_fixed_heuristics_all_detailed(**search_kwargs)
        modified = dict(heuristic_payload["modified_s_s_q"])
        mean_cost = float(modified["mean_cost"])
        searched_params = [int(value) for value in modified["params"]]
    else:
        heuristic_costs = invman_rust.lost_sales_fixed_heuristics_all(
            args.demand_dist_name,
            float(args.demand_rate),
            3.0,
            7.0,
            0.9,
            0.9,
            int(args.lead_time),
            float(args.holding_cost),
            float(args.shortage_cost),
            float(args.procurement_cost),
            float(args.fixed_order_cost),
            int(args.max_order_size),
            int(max(args.one_hot_inventory_upper_bound, args.lead_time * args.max_order_size)),
            int(args.horizon),
            int(seed),
            float(args.warm_up_periods_ratio),
            1,
        )
        mean_cost = float(heuristic_costs["modified_s_s_q"])
        searched_params = [int(value) for value in params] if params is not None else None

    if searched_params is None:
        return mean_cost, {
            "note": "searched heuristic cost is available, but params are unavailable from the legacy binding",
            "params": None,
        }

    demands, demand_value, bounded_trace_horizon = _deterministic_trace_demands(
        args,
        trace_horizon,
    )
    trace_payload = invman_rust.lost_sales_fixed_policy_trace_from_demands(
        policy_name="modified_s_s_q",
        params=searched_params,
        current_inventory=0,
        lead_time_orders=[0] * int(args.lead_time),
        demands=demands,
        max_order_size=int(args.max_order_size),
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        procurement_cost=float(args.procurement_cost),
        fixed_order_cost=float(args.fixed_order_cost),
        warm_up_periods_ratio=float(args.warm_up_periods_ratio),
    )
    trace = list(trace_payload["trace"])
    return mean_cost, {
        "note": "searched mean cost uses the Rust stochastic demand protocol; trace sample uses a deterministic rounded-mean demand path",
        "params": searched_params,
        "trace_mean_cost": float(trace_payload["mean_cost"]),
        "trace_demands": {
            "source": "deterministic_rounded_mean_demand",
            "horizon": bounded_trace_horizon,
            "value": demand_value,
        },
        "trace_summary": summarize_trace(trace),
        "trace_head": trace[: max(0, int(trace_rows))],
    }


def build_policy_state(current_inventory, lead_time_orders, max_order_size, state_features):
    scale = float(max(1, max_order_size))
    pipeline_state = np.asarray(
        [current_inventory + lead_time_orders[0], *lead_time_orders[1:]],
        dtype=np.float32,
    ) / scale
    if state_features == "pipeline":
        return pipeline_state
    if state_features == "pipeline_plus_summary":
        position_scale = float(max(1, len(lead_time_orders) * max_order_size))
        summary = np.asarray(
            [
                current_inventory / scale,
                sum(lead_time_orders) / position_scale,
                (current_inventory + sum(lead_time_orders)) / position_scale,
            ],
            dtype=np.float32,
        )
        return np.concatenate([pipeline_state, summary]).astype(np.float32, copy=False)
    raise NotImplementedError(f"Unsupported state_features '{state_features}'")


def _rust_policy_action(model, state):
    flat_params = model.get_model_flat_params().tolist()
    if model.backbone == "soft_tree":
        action = invman_rust.soft_tree_action_vector_from_flat_params(
            state=state.tolist(),
            flat_params=flat_params,
            input_dim=int(model.input_dim),
            depth=int(model.depth),
            temperature=float(model.temperature),
            split_type=str(model.split_type),
            leaf_type=str(model.leaf_type),
            control_mode=str(model.control_mode),
            min_values=[int(value) for value in model.min_values],
            max_values=[int(value) for value in model.max_values],
            allowed_values=model.allowed_values,
        )
        if len(action) != 1:
            raise ValueError(f"expected scalar action for fixed-cost diagnostic, got {action}")
        return int(action[0])
    if model.backbone == "linear":
        return int(
            invman_rust.linear_policy_action_from_flat_params(
                state=state.tolist(),
                flat_params=flat_params,
                input_dim=int(model.input_dim),
                output_dim=int(model.output_dim),
                policy_head=str(model.action_output_mode),
                policy_max_quantity=model.max_order_size,
            )
        )
    if model.backbone == "nn":
        return int(
            invman_rust.nn_policy_action_from_flat_params(
                state=state.tolist(),
                flat_params=flat_params,
                input_dim=int(model.input_dim),
                hidden_dims=[int(width) for width in model.hidden_dim],
                output_dim=int(model.output_dim),
                activation=str(model.activation_name),
                policy_head=str(model.action_output_mode),
                policy_max_quantity=model.max_order_size,
            )
        )
    raise NotImplementedError(f"Unsupported policy backbone '{model.backbone}'")


def coarse_grid_action_histogram(model, state_features, max_order_size=50, lead_time=4):
    vals = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    counts = Counter()
    for current_inventory in vals:
        for lead_time_orders in product(vals, repeat=lead_time):
            state = build_policy_state(
                current_inventory=current_inventory,
                lead_time_orders=lead_time_orders,
                max_order_size=max_order_size,
                state_features=state_features,
            )
            counts[_rust_policy_action(model, state)] += 1
    return counts.most_common(15), len(counts)


def main():
    args = get_args()
    model = Policy.load(args.model_dir)
    ref_args = build_reference_args(args.horizon, args.state_features)

    model_cost, model_actions = rollout_model(
        model,
        ref_args,
        seed=args.seed,
        trace_horizon=args.trace_horizon,
        trace_rows=args.trace_rows,
    )
    heuristic_cost, heuristic_actions = rollout_modified_s_s_q(
        ref_args,
        seed=args.seed,
        trace_horizon=args.trace_horizon,
        trace_rows=args.trace_rows,
    )
    grid_top_actions, grid_unique_actions = coarse_grid_action_histogram(
        model,
        state_features=args.state_features,
        max_order_size=ref_args.max_order_size,
        lead_time=ref_args.lead_time,
    )

    payload = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "seed": args.seed,
        "horizon": args.horizon,
        "state_features": args.state_features,
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
    text = json.dumps(payload, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
