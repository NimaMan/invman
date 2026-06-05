import math
from types import SimpleNamespace

import numpy as np
import pytest

import invman_rust
from invman.policy_build import build_policy
from invman.policy_registry import apply_policy_name, make_soft_tree_policy_name
from invman.rollout_fitness import get_model_fitness, get_population_fitness
from scripts.dual_sourcing import dual_sourcing_benchmark_lib as lib


def _dual_args(reference_name="dual_l2_ce105", *, policy_name="soft_tree_identity"):
    args = lib.build_reference_args(reference_name)
    args.problem = "dual_sourcing"
    args.policy_name = policy_name
    args.horizon = 80
    args.warm_up_periods_ratio = 0.0
    apply_policy_name(args)
    return args


def _rollout_from_demands(policy, args, *, state, demands):
    return invman_rust.dual_sourcing_soft_tree_rollout_from_demands(
        flat_params=policy.get_model_flat_params().astype(np.float32).tolist(),
        input_dim=policy.input_dim,
        depth=policy.depth,
        min_values=list(policy.min_values),
        max_values=list(policy.max_values),
        action_mode=policy.control_mode,
        action_adapter=policy.action_adapter,
        state=list(state),
        demands=list(demands),
        regular_order_cost=args.regular_order_cost,
        expedited_order_cost=args.expedited_order_cost,
        holding_cost=args.holding_cost,
        shortage_cost=args.shortage_cost,
        regular_max_order_size=args.regular_max_order_size,
        expedited_max_order_size=args.expedited_max_order_size,
        warm_up_periods_ratio=args.warm_up_periods_ratio,
        temperature=policy.temperature,
        split_type=policy.split_type,
        leaf_type=policy.leaf_type,
        allowed_values=policy.allowed_values,
    )


def _step_state(state, regular_order, expedited_order, demand):
    if len(state) == 1:
        return [state[0] + expedited_order - demand + regular_order]
    end_inventory = state[0] + expedited_order - demand
    return [end_inventory + state[1], *state[2:], regular_order]


def _epoch_cost(state, regular_order, expedited_order, demand, args):
    end_inventory = state[0] + expedited_order - demand
    return (
        args.regular_order_cost * regular_order
        + args.expedited_order_cost * expedited_order
        + args.holding_cost * max(end_inventory, 0)
        + args.shortage_cost * max(-end_inventory, 0)
    )


def _mean_after_warmup(costs, warm_up_periods_ratio):
    warmup = min(math.floor(warm_up_periods_ratio * len(costs)), len(costs))
    active = costs[warmup:] if warmup < len(costs) else costs
    return sum(active) / len(active)


def _constant_action_oracle(args, *, state, demands, action):
    state = list(state)
    costs = []
    for demand in demands:
        regular_order, expedited_order = action
        costs.append(_epoch_cost(state, regular_order, expedited_order, demand, args))
        state = _step_state(state, regular_order, expedited_order, demand)
    return _mean_after_warmup(costs, args.warm_up_periods_ratio)


def test_dual_sourcing_reference_instances_include_literature_benchmark_metadata():
    names = [dict(item)["name"] for item in invman_rust.dual_sourcing_list_reference_instances()]
    instance = dict(invman_rust.dual_sourcing_get_reference_instance("dual_l4_ce110"))
    published = dict(instance["published_optimality_gap_pct"])

    assert names == [
        "dual_l2_ce105",
        "dual_l2_ce110",
        "dual_l3_ce105",
        "dual_l3_ce110",
        "dual_l4_ce105",
        "dual_l4_ce110",
    ]
    assert invman_rust.dual_sourcing_primary_reference_instance_name() == "dual_l4_ce110"
    assert instance["source"].startswith("Gijsbrechts")
    assert instance["regular_lead_time"] == 4
    assert instance["initial_state"] == [10, 0, 0, 0]
    assert published["capped_dual_index"] == pytest.approx(0.11)
    assert published["dual_index"] == pytest.approx(0.49)
    assert published["single_index"] == pytest.approx(2.44)
    assert published["tailored_base_surge"] == pytest.approx(0.58)
    assert published["a3c"] == pytest.approx(1.33)


def test_dual_sourcing_gijs_experiment_grid_has_expected_shape():
    grid = lib.get_benchmark_grid()
    instances = lib.build_grid_instances()

    assert grid["name"] == lib.GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME
    assert grid["reference_instance_names"] == [
        "dual_l2_ce105",
        "dual_l2_ce110",
        "dual_l3_ce105",
        "dual_l3_ce110",
        "dual_l4_ce105",
        "dual_l4_ce110",
    ]
    assert grid["regular_lead_times"] == [2, 3, 4]
    assert grid["expedited_order_costs"] == [105.0, 110.0]

    assert len(instances) == 6
    assert instances[0]["name"] == "dual_l2_ce105"
    assert instances[-1]["name"] == "dual_l4_ce110"
    assert instances[-1]["params"]["regular_lead_time"] == 4
    assert instances[-1]["search"]["inventory_lower"] == -12
    assert instances[-1]["search"]["inventory_upper"] == 24
    assert instances[-1]["literature_metadata"]["literature_verified"] is True
    assert (
        instances[-1]["literature_metadata"]["literature_verification_metric"]
        == "published_relative_optimality_gap_pct"
    )


def test_dual_sourcing_default_heuristics_are_rust_searched_and_ranked():
    args = lib.build_reference_args("dual_l2_ce105")
    args.horizon = 50
    args.warm_up_periods_ratio = 0.2

    heuristics = lib.evaluate_default_heuristics(args, seed=123, horizon=50, top_k=2)
    best_name, best_cost = lib.best_heuristic(heuristics)

    assert set(heuristics) == {
        "single_index",
        "dual_index",
        "capped_dual_index",
        "tailored_base_surge",
    }
    assert all(item["available"] for item in heuristics.values())
    assert all(item["source"] == "rust_search_from_demands" for item in heuristics.values())
    assert best_name == "capped_dual_index"
    assert best_cost == pytest.approx(heuristics["capped_dual_index"]["mean_cost"])


@pytest.mark.parametrize(
    ("search_fn", "param_count"),
    [
        (invman_rust.dual_sourcing_single_index_search_from_demands, 2),
        (invman_rust.dual_sourcing_dual_index_search_from_demands, 2),
        (invman_rust.dual_sourcing_capped_dual_index_search_from_demands, 3),
        (invman_rust.dual_sourcing_tailored_base_surge_search_from_demands, 2),
    ],
)
def test_dual_sourcing_direct_search_bindings_return_sorted_top_results(search_fn, param_count):
    common = dict(
        state=[6, 0],
        demands=[0, 1, 2, 3, 4, 2, 1, 0],
        regular_max_order_size=4,
        expedited_max_order_size=4,
        regular_order_cost=10.0,
        expedited_order_cost=12.0,
        holding_cost=1.0,
        shortage_cost=20.0,
        warm_up_periods_ratio=0.0,
        target_upper_bound=8,
        top_k=4,
    )

    best, top = search_fn(**common)

    assert tuple(best) == tuple(top[0])
    assert len(best) == param_count + 1
    assert len(top) <= 4
    assert [row[-1] for row in top] == sorted(row[-1] for row in top)


def test_dual_sourcing_bounded_dp_summary_exposes_stable_policy_metadata():
    summary = dict(
        invman_rust.dual_sourcing_bounded_average_cost_optimal_summary(
            regular_lead_time=2,
            regular_order_cost=10.0,
            expedited_order_cost=12.0,
            holding_cost=1.0,
            shortage_cost=20.0,
            regular_max_order_size=4,
            expedited_max_order_size=4,
            demand_low=0,
            demand_high=2,
            inventory_lower=-8,
            inventory_upper=10,
            tolerance=1e-10,
            max_iterations=500,
        )
    )

    assert summary["initial_state"] == [3, 0]
    assert summary["inventory_bounds"] == [-8, 10]
    assert len(summary["first_action"]) == 2
    assert summary["average_cost"] > 0.0
    assert summary["iterations"] > 0


def test_dual_sourcing_reference_benchmark_matches_figure9_gaps_for_l2_ce105():
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


def test_dual_sourcing_soft_tree_rollout_from_demands_matches_step_oracle_for_identity_head():
    args = _dual_args("dual_l2_ce105", policy_name="soft_tree_identity")
    policy = build_policy(args)
    state = [8, 3]
    demands = [4, 0, 3, 1]

    rust_cost = _rollout_from_demands(policy, args, state=state, demands=demands)
    expected = _constant_action_oracle(args, state=state, demands=demands, action=(1, 1))

    assert policy.leaf_type == "linear"
    assert policy.control_mode == "vector_quantity"
    assert rust_cost == pytest.approx(expected)


@pytest.mark.parametrize(
    "policy_name",
    [
        make_soft_tree_policy_name(
            depth=2,
            temperature=0.25,
            split_type="oblique",
            leaf_type="linear",
            action_adapter="capped_dual_index_targets",
        ),
        "soft_tree_dual_index_delta_targets",
        "soft_tree_capped_dual_index_delta_targets",
        "soft_tree_capped_dual_index_delta_smallcap_targets",
    ],
)
def test_dual_sourcing_structured_soft_tree_rollout_from_demands_is_finite(policy_name):
    args = _dual_args("dual_l2_ce105", policy_name=policy_name)
    policy = build_policy(args)

    cost = _rollout_from_demands(
        policy,
        args,
        state=[6, 0],
        demands=[0, 1, 2, 3, 4, 2],
    )

    assert policy.action_adapter.startswith("dual_sourcing_")
    assert np.isfinite(cost)


def test_dual_sourcing_population_fitness_matches_single_model_fitness():
    args = _dual_args(
        "dual_l2_ce105",
        policy_name="soft_tree_capped_dual_index_delta_smallcap_targets",
    )
    args.warm_up_periods_ratio = 0.2
    args.horizon = 60
    policy = build_policy(args)
    base = policy.get_model_flat_params()
    shifted = np.linspace(-0.2, 0.2, policy.num_params, dtype=np.float32)
    batch = [base, shifted]
    seeds = [11, 22]

    population = get_population_fitness(policy, args, batch, seeds)
    singles = [
        get_model_fitness(policy, args, model_params=params, seed=seed, indiv_idx=idx)
        for idx, (params, seed) in enumerate(zip(batch, seeds))
    ]

    assert [idx for _, idx in population] == [0, 1]
    assert [score for score, _ in population] == pytest.approx([score for score, _ in singles])


def test_dual_sourcing_configure_run_args_preserves_reference_and_policy_protocol(tmp_path):
    parsed = SimpleNamespace(
        budget="screening",
        seed=777,
        same_seed=True,
        mp_num_processors=2,
        eval_horizon=123,
        eval_seeds=2,
        training_episodes=5,
        training_horizon=60,
        run_tag="unit_dual",
    )
    spec = lib.EXPERIMENT_SPECS[0]

    args = lib.configure_run_args(parsed, spec, tmp_path, "dual_l2_ce105")

    assert args.problem == "dual_sourcing"
    assert args.reference_instance == "dual_l2_ce105"
    assert args.policy_name == spec["policy_name"]
    assert args.rollout_backend == "rust"
    assert args.eval_horizon == 123
    assert args.eval_seeds == 2
    assert args.training_episodes == 5
    assert args.horizon == 60
    assert args.results_dir == str(tmp_path / "results")
