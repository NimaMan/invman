"""Fitness evaluation for CMA-ES: hand a `Policy` + problem args to the Rust rollout.

OBJECTIVE
---------
Python optimizes; Rust rolls out. This module is the single seam: it builds the
argument payload the `invman_rust` PyO3 rollout expects from a `Policy` (bounds +
architecture) and the CLI `args` (problem/env), calls Rust, and returns CMA-ES
fitness as (-cost, individual_index).

It exposes the two callables `es_mp.train` consumes:
  - get_model_fitness(model, args, model_params=None, seed=..., indiv_idx=..., ...)
      -> (fitness, indiv_idx)   [single candidate]
  - get_population_fitness(model, args, model_params_batch, seeds)
      -> [(fitness, indiv_idx), ...]   [whole population, batched in Rust]

Coverage matches the available Rust rollouts:
  - lost_sales / lost_sales_fixed_order_cost: soft_tree, linear, nn
  - dual_sourcing, multi_echelon: soft_tree only
"""

from __future__ import annotations

import numpy as np

DEFAULT_MMPP2_LAMBDA_LOW = 3.0
DEFAULT_MMPP2_LAMBDA_HIGH = 7.0
DEFAULT_MMPP2_POSITIVE_P00 = 0.9
DEFAULT_MMPP2_POSITIVE_P11 = 0.9

_BOUNDED_DENSE_HEADS = {
    "categorical_quantity",
    "capped_direct_quantity",
    "sigmoid_direct_quantity",
    "soft_gated_direct_quantity",
    "gated_sigmoid_direct_quantity",
    "hard_gated_direct_quantity",
    "soft_gated_ordinal_quantity",
    "hard_gated_ordinal_quantity",
}


def _flat(params) -> list:
    return np.asarray(params, dtype=np.float32).tolist()


def _demand_kwargs(args) -> dict:
    return {
        "demand_lambda_low": float(getattr(args, "demand_lambda_low", DEFAULT_MMPP2_LAMBDA_LOW)),
        "demand_lambda_high": float(getattr(args, "demand_lambda_high", DEFAULT_MMPP2_LAMBDA_HIGH)),
        "demand_p00": float(getattr(args, "demand_p00", DEFAULT_MMPP2_POSITIVE_P00)),
        "demand_p11": float(getattr(args, "demand_p11", DEFAULT_MMPP2_POSITIVE_P11)),
    }


def _state_normalization(policy):
    state_normalizer = str(getattr(policy, "state_normalizer", "identity"))
    state_scale = getattr(policy, "state_scale", None)
    if state_normalizer == "identity":
        return state_normalizer, None
    if state_scale is None:
        raise ValueError(f"{state_normalizer} requires an explicit state_scale")
    return state_normalizer, float(state_scale)


def _dense_policy_max_quantity(policy):
    if str(getattr(policy, "action_output_mode", "")) not in _BOUNDED_DENSE_HEADS:
        return None
    cap = getattr(policy, "max_order_size", None)
    if cap is None:
        raise ValueError(f"{policy.action_output_mode} requires a policy-side quantity cap")
    return int(cap)


def _soft_tree_policy_max_quantity(policy):
    if str(getattr(policy, "leaf_type", "constant")) != "sigmoid_linear":
        return None
    cap = getattr(policy, "max_order_size", None)
    if cap is None:
        raise ValueError("sigmoid_linear soft-tree leaves require a policy-side quantity cap")
    return int(cap)


# --- lost_sales / lost_sales_fixed_order_cost --------------------------------

def _lost_sales_single(invman_rust, policy, args, flat_params, seed):
    state_normalizer, state_scale = _state_normalization(policy)
    if policy.backbone == "soft_tree":
        return invman_rust.lost_sales_soft_tree_rollout(
            flat_params=_flat(flat_params),
            input_dim=int(policy.input_dim),
            depth=int(policy.depth),
            policy_max_quantity=_soft_tree_policy_max_quantity(policy),
            split_type=str(policy.split_type),
            leaf_type=str(policy.leaf_type),
            demand_rate=float(args.demand_rate),
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            seed=int(seed),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            temperature=float(policy.temperature),
            state_normalizer=state_normalizer,
            state_scale=state_scale,
            **_demand_kwargs(args),
        )
    if policy.backbone == "linear":
        return invman_rust.lost_sales_linear_rollout(
            flat_params=_flat(flat_params),
            input_dim=int(policy.input_dim),
            output_dim=int(policy.output_dim),
            policy_max_quantity=_dense_policy_max_quantity(policy),
            policy_head=str(policy.action_output_mode),
            demand_rate=float(args.demand_rate),
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            seed=int(seed),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            state_normalizer=state_normalizer,
            state_scale=state_scale,
            **_demand_kwargs(args),
        )
    # nn
    return invman_rust.lost_sales_nn_rollout(
        flat_params=_flat(flat_params),
        input_dim=int(policy.input_dim),
        hidden_dims=[int(w) for w in policy.hidden_dim],
        output_dim=int(policy.output_dim),
        policy_max_quantity=_dense_policy_max_quantity(policy),
        policy_head=str(policy.action_output_mode),
        activation=str(policy.activation_name),
        demand_rate=float(args.demand_rate),
        demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
        lead_time=int(args.lead_time),
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
        fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
        horizon=int(args.horizon),
        seed=int(seed),
        warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
        state_normalizer=state_normalizer,
        state_scale=state_scale,
        **_demand_kwargs(args),
    )


def _lost_sales_population(invman_rust, policy, args, params_batch, seeds):
    state_normalizer, state_scale = _state_normalization(policy)
    seeds = [int(s) for s in seeds]
    if policy.backbone == "soft_tree":
        return invman_rust.lost_sales_soft_tree_population_rollout(
            params_batch=params_batch,
            input_dim=int(policy.input_dim),
            depth=int(policy.depth),
            policy_max_quantity=_soft_tree_policy_max_quantity(policy),
            split_type=str(policy.split_type),
            leaf_type=str(policy.leaf_type),
            demand_rate=float(args.demand_rate),
            seeds=seeds,
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            temperature=float(policy.temperature),
            state_normalizer=state_normalizer,
            state_scale=state_scale,
            **_demand_kwargs(args),
        )
    if policy.backbone == "linear":
        return invman_rust.lost_sales_linear_population_rollout(
            params_batch=params_batch,
            input_dim=int(policy.input_dim),
            output_dim=int(policy.output_dim),
            policy_max_quantity=_dense_policy_max_quantity(policy),
            policy_head=str(policy.action_output_mode),
            demand_rate=float(args.demand_rate),
            seeds=seeds,
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            state_normalizer=state_normalizer,
            state_scale=state_scale,
            **_demand_kwargs(args),
        )
    return invman_rust.lost_sales_nn_population_rollout(
        params_batch=params_batch,
        input_dim=int(policy.input_dim),
        hidden_dims=[int(w) for w in policy.hidden_dim],
        output_dim=int(policy.output_dim),
        policy_max_quantity=_dense_policy_max_quantity(policy),
        policy_head=str(policy.action_output_mode),
        activation=str(policy.activation_name),
        demand_rate=float(args.demand_rate),
        seeds=seeds,
        demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
        lead_time=int(args.lead_time),
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
        fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
        horizon=int(args.horizon),
        warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
        state_normalizer=state_normalizer,
        state_scale=state_scale,
        **_demand_kwargs(args),
    )


# --- dual_sourcing (soft_tree only) ------------------------------------------

def _dual_sourcing_kwargs(policy, args):
    return dict(
        input_dim=int(policy.input_dim),
        depth=int(policy.depth),
        min_values=[int(v) for v in policy.min_values],
        max_values=[int(v) for v in policy.max_values],
        allowed_values=policy.allowed_values,
        split_type=str(policy.split_type),
        leaf_type=str(policy.leaf_type),
        action_mode=str(policy.control_mode),
        action_adapter=str(policy.action_adapter),
        regular_lead_time=int(args.regular_lead_time),
        regular_order_cost=float(args.regular_order_cost),
        expedited_order_cost=float(args.expedited_order_cost),
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        regular_max_order_size=int(args.regular_max_order_size),
        expedited_max_order_size=int(args.expedited_max_order_size),
        demand_low=int(args.dual_demand_low),
        demand_high=int(args.dual_demand_high),
        horizon=int(args.horizon),
        warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
        temperature=float(policy.temperature),
    )


# --- multi_echelon (soft_tree only) ------------------------------------------

def _multi_echelon_kwargs(policy, args):
    # lost-sales-style policy: feed the pure decision state and let the policy normalize it.
    state_normalizer, state_scale = _state_normalization(policy)
    return dict(
        input_dim=int(policy.input_dim),
        depth=int(policy.depth),
        min_values=[int(v) for v in policy.min_values],
        max_values=[int(v) for v in policy.max_values],
        allowed_values=policy.allowed_values,
        split_type=str(policy.split_type),
        leaf_type=str(policy.leaf_type),
        action_mode=str(policy.control_mode),
        policy_feature_mode="raw_decision_state",
        state_normalizer=state_normalizer,
        state_scale=state_scale,
        warehouse_lead_time=int(args.warehouse_lead_time),
        retailer_lead_time=int(args.retailer_lead_time),
        num_retailers=int(args.num_retailers),
        warehouse_holding_cost=float(args.warehouse_holding_cost),
        retailer_holding_cost=float(args.retailer_holding_cost),
        warehouse_expedited_cost=float(args.warehouse_expedited_cost),
        warehouse_lost_sale_cost=float(args.warehouse_lost_sale_cost),
        expedited_service_prob=float(args.expedited_service_prob),
        warehouse_capacity=int(args.warehouse_capacity),
        warehouse_inventory_cap=int(args.warehouse_inventory_cap),
        retailer_inventory_cap=int(args.retailer_inventory_cap),
        # Required positional args of multi_echelon_soft_tree_rollout. Default to the
        # paper-faithful gijs_2022 dynamics and the rounded-clipped-normal demand so the
        # CMA-ES fitness target constructs even when a caller forgets to set them.
        inventory_dynamics_mode=str(getattr(args, "inventory_dynamics_mode", "gijs_2022")),
        demand_distribution=str(getattr(args, "demand_distribution", "normal_rounded_clipped")),
        demand_mean=float(args.multi_demand_mean),
        demand_std=float(args.multi_demand_std),
        horizon=int(args.horizon),
        warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
        # Use the long-run average cost after warm-up so the learned-policy fitness is on
        # the same scale as the constant-base-stock benchmark (and is horizon-stable),
        # rather than the binding's raw cumulative-cost default.
        objective=str(getattr(args, "rollout_objective", "average_cost_after_warmup")),
        temperature=float(policy.temperature),
    )


# --- public dispatch ---------------------------------------------------------

def get_model_fitness(model, args, model_params=None, seed=1234, indiv_idx=-1,
                      return_env=False, track_demand=False, verbose=False):
    import invman_rust

    flat_params = model_params if model_params is not None else model.get_model_flat_params()
    problem = getattr(args, "problem", "lost_sales")
    if problem in ("lost_sales", "lost_sales_fixed_order_cost"):
        avg_cost = _lost_sales_single(invman_rust, model, args, flat_params, seed)
    elif problem == "dual_sourcing":
        avg_cost = invman_rust.dual_sourcing_soft_tree_rollout(
            flat_params=_flat(flat_params), seed=int(seed), **_dual_sourcing_kwargs(model, args)
        )
    elif problem == "multi_echelon":
        avg_cost = invman_rust.multi_echelon_soft_tree_rollout(
            flat_params=_flat(flat_params), seed=int(seed), **_multi_echelon_kwargs(model, args)
        )
    else:
        raise ValueError(f"Unknown problem '{problem}'")
    if verbose:
        print(f"Seed {seed}: avg cost {avg_cost:.4f}")
    return -float(avg_cost), indiv_idx


def get_population_fitness(model, args, model_params_batch, seeds):
    import invman_rust

    params_batch = [_flat(p) for p in model_params_batch]
    problem = getattr(args, "problem", "lost_sales")
    if problem in ("lost_sales", "lost_sales_fixed_order_cost"):
        costs = _lost_sales_population(invman_rust, model, args, params_batch, seeds)
    elif problem == "dual_sourcing":
        costs = invman_rust.dual_sourcing_soft_tree_population_rollout(
            params_batch=params_batch, seeds=[int(s) for s in seeds],
            **_dual_sourcing_kwargs(model, args)
        )
    elif problem == "multi_echelon":
        costs = invman_rust.multi_echelon_soft_tree_population_rollout(
            params_batch=params_batch, seeds=[int(s) for s in seeds],
            **_multi_echelon_kwargs(model, args)
        )
    else:
        raise ValueError(f"Unknown problem '{problem}'")
    return [(-float(cost), indiv_idx) for indiv_idx, cost in enumerate(costs)]
