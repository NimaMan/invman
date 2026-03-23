import numpy as np
import pytest
import torch

from invman.policies import SoftTreePolicy
from invman.problems.dual_sourcing import (
    get_benchmark_reference,
    build_reference_args,
    build_fixed_demand_path,
    get_reference_instance,
    search_best_capped_dual_index_policy,
    search_best_dual_index_policy,
    search_best_single_index_policy,
    search_best_tailored_base_surge_policy,
)
from invman.problems.dual_sourcing.env import DualSourcingEnv

invman_rust = pytest.importorskip("invman_rust")


def test_dual_sourcing_env_step_updates_reduced_state():
    env = DualSourcingEnv(
        regular_lead_time=3,
        regular_order_cost=100.0,
        expedited_order_cost=105.0,
        holding_cost=5.0,
        shortage_cost=495.0,
        regular_max_order_size=12,
        expedited_max_order_size=12,
        demand_low=0,
        demand_high=4,
        horizon=2,
        track_demand=True,
        warm_up_periods_ratio=0.0,
    )
    env.state = [8, 3, 1]
    env.horizon_demand = [4, 0]

    _, epoch_cost, done = env.step((2, 1))

    assert env.state == [8, 1, 2]
    assert epoch_cost == pytest.approx(100.0 * 2 + 105.0 * 1 + 5.0 * 5)
    assert done is False


def test_dual_sourcing_search_backends_match():
    args = build_reference_args("dual_l2_ce105")
    args.horizon = 200
    args.warm_up_periods_ratio = 0.0

    for search_fn in (
        search_best_single_index_policy,
        search_best_dual_index_policy,
        search_best_capped_dual_index_policy,
        search_best_tailored_base_surge_policy,
    ):
        python_result = search_fn(args, seed=321, horizon=200, backend="python")
        rust_result = search_fn(args, seed=321, horizon=200, backend="rust")
        assert rust_result.best_result.params == python_result.best_result.params
        assert rust_result.best_result.mean_cost == pytest.approx(python_result.best_result.mean_cost)


def test_dual_sourcing_reference_instances_include_literature_benchmark_metadata():
    benchmark = get_benchmark_reference()
    instance = get_reference_instance("dual_l4_ce110")

    assert "optimal_dp" in benchmark.benchmark_policies
    assert "capped_dual_index" in benchmark.benchmark_policies
    assert benchmark.published_values["a3c_optimality_gap_pct_upper"] == 2.0
    assert instance.expected_ranking[0] == "capped_dual_index"
    assert instance.literature_values["best_reported_heuristic_family"] == "capped_dual_index"
    assert instance.literature_values["has_exact_published_cost"] is False


def test_dual_sourcing_soft_tree_rust_matches_python_rollout():
    args = build_reference_args("dual_l2_ce105")
    args.horizon = 50
    args.warm_up_periods_ratio = 0.0
    fixed_path = build_fixed_demand_path(args, seed=77, horizon=50)
    env = DualSourcingEnv(
        regular_lead_time=args.regular_lead_time,
        regular_order_cost=args.regular_order_cost,
        expedited_order_cost=args.expedited_order_cost,
        holding_cost=args.holding_cost,
        shortage_cost=args.shortage_cost,
        regular_max_order_size=args.regular_max_order_size,
        expedited_max_order_size=args.expedited_max_order_size,
        demand_low=args.dual_demand_low,
        demand_high=args.dual_demand_high,
        horizon=args.horizon,
        track_demand=True,
        warm_up_periods_ratio=args.warm_up_periods_ratio,
    )
    model = SoftTreePolicy(
        input_dim=env.state_space_dim,
        action_spec=env.action_spec,
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="constant",
    )
    model.set_model_params(model.get_model_flat_params())

    env.state = list(fixed_path.state)
    env.current_epoch = 0
    env.done = False
    env.epoch_costs = []
    env.total_cost = 0.0
    env.horizon_demand = np.asarray(fixed_path.demands, dtype=np.int64)
    while not env.is_done():
        env.step(model(torch.as_tensor(env.policy_state, dtype=torch.float32)))

    rust_cost = invman_rust.dual_sourcing_soft_tree_rollout_from_demands(
        flat_params=model.get_model_flat_params().astype(np.float32).tolist(),
        input_dim=model.input_dim,
        depth=model.depth,
        min_values=model.min_values,
        max_values=model.max_values,
        action_mode=model.action_mode,
        state=list(fixed_path.state),
        demands=list(fixed_path.demands),
        regular_order_cost=args.regular_order_cost,
        expedited_order_cost=args.expedited_order_cost,
        holding_cost=args.holding_cost,
        shortage_cost=args.shortage_cost,
        regular_max_order_size=args.regular_max_order_size,
        expedited_max_order_size=args.expedited_max_order_size,
        warm_up_periods_ratio=0.0,
        temperature=model.temperature,
        split_type=model.split_type,
        leaf_type=model.leaf_type,
        allowed_values=model.action_spec["allowed_values"],
    )
    assert rust_cost == pytest.approx(env.avg_total_cost)
