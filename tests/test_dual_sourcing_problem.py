from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from invman.policies import apply_policy_name, make_soft_tree_policy_name
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
    solve_bounded_dp,
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
    assert benchmark.published_values["published_metric"] == "optimality_gap_pct"
    assert instance.expected_ranking == (
        "capped_dual_index",
        "dual_index",
        "tailored_base_surge",
        "single_index",
    )
    assert instance.literature_values["best_reported_heuristic_family"] == "capped_dual_index"
    assert instance.literature_values["published_optimality_gap_pct"] == {
        "capped_dual_index": 0.11,
        "dual_index": 0.49,
        "single_index": 2.44,
        "tailored_base_surge": 0.58,
        "a3c": 1.33,
    }
    assert instance.literature_values["has_exact_published_cost"] is False


def test_dual_sourcing_bounded_dp_reports_average_cost_of_extracted_policy():
    args = SimpleNamespace(
        regular_lead_time=2,
        regular_order_cost=10.0,
        expedited_order_cost=12.0,
        holding_cost=1.0,
        shortage_cost=20.0,
        regular_max_order_size=4,
        expedited_max_order_size=4,
        dual_demand_low=0,
        dual_demand_high=2,
    )
    inventory_lower = -8
    inventory_upper = 10

    dp_result = solve_bounded_dp(
        args,
        inventory_lower=inventory_lower,
        inventory_upper=inventory_upper,
        tolerance=1e-10,
        max_iterations=500,
    )

    demand_values = list(range(args.dual_demand_low, args.dual_demand_high + 1))
    demand_prob = 1.0 / len(demand_values)
    state_space = list(
        product(
            range(inventory_lower, inventory_upper + 1),
            range(args.regular_max_order_size + 1),
        )
    )
    state_to_idx = {state: idx for idx, state in enumerate(state_space)}
    transition = np.zeros((len(state_space), len(state_space)), dtype=np.float64)
    expected_cost = np.zeros(len(state_space), dtype=np.float64)

    for state_idx, state in enumerate(state_space):
        regular_order, expedited_order = dp_result.best_action_by_state[state]
        for demand in demand_values:
            end_inventory = int(state[0]) + int(expedited_order) - int(demand)
            next_state = (
                max(inventory_lower, min(inventory_upper, end_inventory + int(state[1]))),
                max(0, min(args.regular_max_order_size, int(regular_order))),
            )
            transition[state_idx, state_to_idx[next_state]] += demand_prob
            expected_cost[state_idx] += demand_prob * (
                args.regular_order_cost * int(regular_order)
                + args.expedited_order_cost * int(expedited_order)
                + args.holding_cost * max(end_inventory, 0)
                + args.shortage_cost * max(-end_inventory, 0)
            )

    initial_state = (
        int(round((args.regular_lead_time + 1) * 0.5 * (args.dual_demand_low + args.dual_demand_high))),
        0,
    )
    distribution = np.zeros(len(state_space), dtype=np.float64)
    distribution[state_to_idx[initial_state]] = 1.0
    for _ in range(10_000):
        next_distribution = distribution @ transition
        if np.max(np.abs(next_distribution - distribution)) < 1e-14:
            distribution = next_distribution
            break
        distribution = next_distribution

    stationary_cost = float(distribution @ expected_cost)
    assert dp_result.average_cost == pytest.approx(stationary_cost, abs=1e-8)


def test_dual_sourcing_rust_bounded_average_cost_matches_python_dp():
    args = SimpleNamespace(
        regular_lead_time=2,
        regular_order_cost=10.0,
        expedited_order_cost=12.0,
        holding_cost=1.0,
        shortage_cost=20.0,
        regular_max_order_size=4,
        expedited_max_order_size=4,
        dual_demand_low=0,
        dual_demand_high=2,
    )

    python_summary = solve_bounded_dp(
        args,
        inventory_lower=-8,
        inventory_upper=10,
        tolerance=1e-10,
        max_iterations=500,
    )
    rust_summary = dict(
        invman_rust.dual_sourcing_bounded_average_cost_optimal_summary(
            regular_lead_time=args.regular_lead_time,
            regular_order_cost=args.regular_order_cost,
            expedited_order_cost=args.expedited_order_cost,
            holding_cost=args.holding_cost,
            shortage_cost=args.shortage_cost,
            regular_max_order_size=args.regular_max_order_size,
            expedited_max_order_size=args.expedited_max_order_size,
            demand_low=args.dual_demand_low,
            demand_high=args.dual_demand_high,
            inventory_lower=-8,
            inventory_upper=10,
            tolerance=1e-10,
            max_iterations=500,
        )
    )

    initial_state = tuple(rust_summary["initial_state"])
    assert rust_summary["average_cost"] == pytest.approx(python_summary.average_cost, abs=1e-10)
    assert tuple(rust_summary["first_action"]) == python_summary.best_action_by_state[initial_state]
    assert rust_summary["iterations"] == python_summary.iterations


def test_dual_sourcing_rust_reference_benchmark_matches_figure9_gaps_for_l2_ce105():
    report = dict(
        invman_rust.dual_sourcing_reference_benchmark_summary(
            reference_instance_name="dual_l2_ce105",
            inventory_lower=-12,
            inventory_upper=24,
            tolerance=1e-8,
            max_iterations=250,
            search_seed=123,
            search_horizon=6000,
            warm_up_periods_ratio=0.2,
        )
    )

    assert report["reference_name"] == "dual_l2_ce105"
    assert report["optimal"]["average_cost"] > 0.0

    heuristics = {entry["policy_name"]: dict(entry) for entry in report["heuristics"]}
    assert list(heuristics) == [
        "capped_dual_index",
        "tailored_base_surge",
        "dual_index",
        "single_index",
    ]

    assert heuristics["capped_dual_index"]["published_optimality_gap_pct"] == pytest.approx(0.00)
    assert heuristics["tailored_base_surge"]["published_optimality_gap_pct"] == pytest.approx(0.06)
    assert heuristics["dual_index"]["published_optimality_gap_pct"] == pytest.approx(0.11)
    assert heuristics["single_index"]["published_optimality_gap_pct"] == pytest.approx(0.56)

    for entry in heuristics.values():
        assert entry["average_cost"] >= report["optimal"]["average_cost"] - 1e-8
        assert entry["optimality_gap_pct"] == pytest.approx(
            entry["published_optimality_gap_pct"],
            abs=0.01,
        )


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
        env.step(model(env.policy_state))

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


def test_dual_sourcing_structured_tree_rust_matches_python_rollout():
    args = build_reference_args("dual_l2_ce105")
    args.horizon = 50
    args.warm_up_periods_ratio = 0.0
    args.problem = "dual_sourcing"
    args.policy_name = make_soft_tree_policy_name(
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
        action_adapter="capped_dual_index_targets",
    )
    apply_policy_name(args)
    fixed_path = build_fixed_demand_path(args, seed=91, horizon=50)
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
    from invman.policies.factory import build_policy

    model = build_policy(args, env)
    model.set_model_params(model.get_model_flat_params())

    env.state = list(fixed_path.state)
    env.current_epoch = 0
    env.done = False
    env.epoch_costs = []
    env.total_cost = 0.0
    env.horizon_demand = np.asarray(fixed_path.demands, dtype=np.int64)
    while not env.is_done():
        env.step(model(env.policy_state))

    rust_cost = invman_rust.dual_sourcing_soft_tree_rollout_from_demands(
        flat_params=model.get_model_flat_params().astype(np.float32).tolist(),
        input_dim=model.input_dim,
        depth=model.depth,
        min_values=model.min_values,
        max_values=model.max_values,
        action_mode=model.control_mode,
        action_adapter=model.action_adapter,
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
        allowed_values=model.control_spec["allowed_values"],
    )
    assert rust_cost == pytest.approx(env.avg_total_cost)
