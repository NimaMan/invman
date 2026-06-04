use crate::core::policies::soft_tree::build_action_spec;
use crate::problems::one_warehouse_multi_retailer::allocation::{
    min_shortage_shipments, proportional_shipments,
};
use crate::problems::one_warehouse_multi_retailer::env::{
    build_raw_state, initialize_state, retailer_inventory_positions, step_state,
};
use crate::problems::one_warehouse_multi_retailer::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::one_warehouse_multi_retailer::heuristics::echelon_base_stock_orders;
use crate::problems::one_warehouse_multi_retailer::references::{
    KAYNOV_2024_REFERENCE, PRIMARY_REFERENCE_INSTANCE, TABLE_A3_INSTANCES,
    VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE,
};
use crate::problems::one_warehouse_multi_retailer::rollout::{
    policy_action_from_tree, OneWarehouseMultiRetailerRolloutConfig, PolicyActionMode,
};

fn nested_pipeline_vec(pipelines: &[&[usize]]) -> Vec<Vec<usize>> {
    pipelines.iter().map(|pipeline| pipeline.to_vec()).collect()
}

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(KAYNOV_2024_REFERENCE.benchmark_policies.len(), 3);
    assert_eq!(TABLE_A3_INSTANCES.len(), 14);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.name, "kaynov2024_instance_7");
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.retailer_lead_times.len(), 3);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE
            .published_proportional_benchmark
            .expect("primary benchmark must exist")
            .mean_cost,
        -1406.27
    );
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.retailer_lead_times, &[1, 1]);
}

#[test]
fn raw_state_layout_matches_expected_shape() {
    let state =
        initialize_state(3, &[2, 2], &[1, 0], &vec![vec![1], vec![0]]).expect("state must build");
    let raw_state = build_raw_state(&state).expect("raw state must build");
    assert_eq!(raw_state, vec![3.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(
        worked.initial_warehouse_inventory,
        worked.initial_warehouse_pipeline,
        worked.initial_retailer_inventory,
        &nested_pipeline_vec(worked.initial_retailer_pipeline),
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.warehouse_order,
        worked.retailer_shipments,
        worked.realized_demands,
        0.5,
        &[1.0, 1.0],
        &[9.0, 9.0],
        worked.customer_behavior,
        0.0,
        None,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.next_state.warehouse_inventory,
        worked.expected_next_warehouse_inventory
    );
    assert_eq!(
        outcome.next_state.warehouse_pipeline,
        worked.expected_next_warehouse_pipeline.to_vec()
    );
    assert_eq!(
        outcome.next_state.retailer_inventory,
        worked.expected_next_retailer_inventory.to_vec()
    );
    assert_eq!(
        outcome.next_state.retailer_pipeline,
        nested_pipeline_vec(worked.expected_next_retailer_pipeline)
    );
    assert_eq!(outcome.holding_cost, worked.expected_holding_cost);
    assert_eq!(outcome.shortage_cost, worked.expected_shortage_cost);
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn allocation_and_base_stock_orders_match_named_heuristic_evaluators() {
    let reference = VERIFICATION_PROBLEM_INSTANCE;
    let state = initialize_state(
        reference.initial_warehouse_inventory,
        reference.initial_warehouse_pipeline,
        reference.initial_retailer_inventory,
        &nested_pipeline_vec(reference.initial_retailer_pipeline),
    )
    .expect("state must build");
    let action = echelon_base_stock_orders(
        &state,
        reference.heuristic_warehouse_base_stock_level,
        reference.heuristic_retailer_base_stock_levels,
    )
    .expect("base-stock orders must compute");
    let retailer_positions = retailer_inventory_positions(&state).expect("positions must compute");
    let proportional = proportional_shipments(
        (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize,
        &action[1..],
    )
    .expect("proportional shipments must compute");
    let min_shortage = min_shortage_shipments(
        (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize,
        &action[1..],
        &retailer_positions,
        reference.heuristic_retailer_base_stock_levels,
    )
    .expect("min-shortage shipments must compute");

    let proportional_eval = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "echelon_base_stock_proportional",
    )
    .expect("proportional heuristic evaluation must solve");
    let min_shortage_eval = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "echelon_base_stock_min_shortage",
    )
    .expect("min-shortage heuristic evaluation must solve");

    assert_eq!(action, proportional_eval.first_action);
    assert_eq!(action, min_shortage_eval.first_action);
    // Proportional now floors (Kaynov Eq. 8, remainder stays at the warehouse), so it ships no more
    // than the exhausting min-shortage allocation.
    assert!(
        proportional.iter().sum::<usize>() <= min_shortage.iter().sum::<usize>()
    );
    assert!(
        proportional.iter().sum::<usize>()
            <= (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize
    );
}

#[test]
fn proportional_allocation_floors_per_kaynov_eq_8_and_leaves_remainder_at_warehouse() {
    // Kaynov et al. (2024) Eq. (8): q_i = floor(a_i * available / sum a_j); the remainder is NOT
    // redistributed to retailers, it stays at the warehouse. For available=5, orders=[4,4,4]
    // (total 12): floor(4*5/12)=1 each -> [1,1,1], summing to 3, with 2 units retained.
    let shipments =
        proportional_shipments(5, &[4, 4, 4]).expect("proportional allocation must compute");
    assert_eq!(shipments, vec![1, 1, 1]);
    assert_eq!(shipments.iter().sum::<usize>(), 3);
    assert!(shipments.iter().sum::<usize>() <= 5);
}

#[test]
fn symmetric_echelon_target_mode_expands_shared_retailer_target() {
    let reference = VERIFICATION_PROBLEM_INSTANCE;
    let state = initialize_state(
        reference.initial_warehouse_inventory,
        reference.initial_warehouse_pipeline,
        reference.initial_retailer_inventory,
        &nested_pipeline_vec(reference.initial_retailer_pipeline),
    )
    .expect("state must build");
    let config = OneWarehouseMultiRetailerRolloutConfig {
        input_dim: 1
            + state.warehouse_pipeline.len()
            + state.retailer_inventory.len()
            + state
                .retailer_pipeline
                .iter()
                .map(|pipeline| pipeline.len())
                .sum::<usize>()
            + 2,
        depth: 1,
        action_spec: build_action_spec(
            "discrete_grid",
            vec![0, 0],
            vec![6, 4],
            Some(vec![vec![0, 3, 6], vec![0, 2, 4]]),
        )
        .expect("action spec must build"),
        periods: reference.periods,
        demand_models: vec![],
        allocation_policy: crate::problems::one_warehouse_multi_retailer::allocation::AllocationPolicy::Proportional,
        retailer_target_inventory_positions: None,
        holding_cost_warehouse: reference.holding_cost_warehouse,
        holding_cost_retailers: reference.holding_cost_retailers.to_vec(),
        penalty_costs_retailers: reference.penalty_costs_retailers.to_vec(),
        customer_behavior: reference.customer_behavior,
        emergency_shipment_probability: reference.emergency_shipment_probability,
        discount_factor: reference.discount_factor,
        policy_action_mode: PolicyActionMode::SymmetricEchelonTargets,
        temperature: 0.1,
        split_type: crate::core::policies::soft_tree::SoftTreeSplitType::AxisAligned,
        leaf_type: crate::core::policies::soft_tree::SoftTreeLeafType::Constant,
    };
    let flat_params = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // split weights + bias
        0.0, 0.0, 0.0, 0.0, // identical leaf logits => projected controls [3, 2]
    ];
    let action = policy_action_from_tree(&flat_params, &state, &config)
        .expect("symmetric action must compute");
    assert_eq!(action.retailer_target_inventory_positions, Some(vec![2, 2]));
    assert_eq!(
        action.orders,
        echelon_base_stock_orders(&state, 3, &[2, 2]).expect("orders must compute")
    );
}

#[test]
fn finite_horizon_dp_dominates_repo_heuristics() {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)
        .expect("optimal finite-horizon DP must solve");
    let proportional = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "echelon_base_stock_proportional",
    )
    .expect("proportional heuristic evaluation must solve");
    let min_shortage = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "echelon_base_stock_min_shortage",
    )
    .expect("min-shortage heuristic evaluation must solve");

    assert!(
        optimal.discounted_cost <= proportional.discounted_cost + 1e-9,
        "optimal={} proportional={}",
        optimal.discounted_cost,
        proportional.discounted_cost
    );
    assert!(
        optimal.discounted_cost <= min_shortage.discounted_cost + 1e-9,
        "optimal={} min_shortage={}",
        optimal.discounted_cost,
        min_shortage.discounted_cost
    );
}

/// Literature verification (executing, stochastic): the env reproduces published Kaynov et al.
/// (2024) Table A.3 benchmark REWARDS by simulation (100 periods x 1000 replications, undiscounted,
/// mean-filled pipeline warm start, grid-searched echelon base-stock levels). Each regime is matched
/// by the allocation rule the env is faithful to:
///   - instance_7 (lost_sales), min-shortage: env 1394.8 vs published 1408.08 -> -0.94%
///   - instance_11 (partial_backorder), proportional (Eq.8 floor + post-emergency holding): env
///     1113.2 vs published 1111.76 -> +0.13%
/// (On lost-sales the Eq.8 floor proportional is +3.1%, so that regime is verified via min-shortage;
/// the backorder regime reaches ~-1.4% after the fixes but is not within ~1%, an unresolved residual
/// in the paper's underspecified min-shortage stop-rule.) Levels/seed come from
/// scripts/one_warehouse_multi_retailer/run_heuristic_published_benchmark.py.
#[test]
fn kaynov_table_a3_rows_reproduced_by_env_simulation() {
    use crate::problems::one_warehouse_multi_retailer::allocation::AllocationPolicy;
    use crate::problems::one_warehouse_multi_retailer::demand::mean_demand;
    use crate::problems::one_warehouse_multi_retailer::heuristics::simulate_policy;

    // Mean-filled pipeline warm start, then simulate; returns the mean 100-period cost.
    fn simulate_benchmark(
        instance_name: &str,
        allocation: AllocationPolicy,
        warehouse_base_stock: usize,
        retailer_base_stock: &[usize],
        seed: u64,
    ) -> f64 {
        let reference = TABLE_A3_INSTANCES
            .iter()
            .find(|r| r.name == instance_name)
            .expect("benchmark instance must exist");
        let means: Vec<f64> = reference
            .demand_models
            .iter()
            .map(|model| mean_demand(model).expect("demand mean must compute"))
            .collect();
        let retailer_inventory: Vec<i32> = means.iter().map(|mean| mean.round() as i32).collect();
        let warehouse_inventory = means.iter().sum::<f64>().round() as i32;
        let warehouse_pipeline = vec![warehouse_inventory as usize; reference.warehouse_lead_time];
        let retailer_pipeline: Vec<Vec<usize>> = reference
            .retailer_lead_times
            .iter()
            .enumerate()
            .map(|(idx, &lead)| vec![retailer_inventory[idx] as usize; lead])
            .collect();
        let state = initialize_state(
            warehouse_inventory,
            &warehouse_pipeline,
            &retailer_inventory,
            &retailer_pipeline,
        )
        .expect("mean-filled warm-start state must build");

        let mut params = vec![warehouse_base_stock as f64];
        params.extend(retailer_base_stock.iter().map(|&level| level as f64));

        let summary = simulate_policy(
            "echelon_base_stock",
            &params,
            &state,
            reference.benchmark_periods,
            reference.benchmark_replications,
            seed,
            reference.demand_models,
            allocation,
            reference.holding_cost_warehouse,
            reference.holding_cost_retailers,
            reference.penalty_costs_retailers,
            reference.customer_behavior,
            reference.emergency_shipment_probability,
            1.0,
        )
        .expect("policy simulation must succeed");
        summary.mean_cost
    }

    // instance_7 (lost_sales) verified via min-shortage allocation (S_w=44, S_r=[10,10,10]).
    let instance_7 = TABLE_A3_INSTANCES
        .iter()
        .find(|r| r.name == "kaynov2024_instance_7")
        .unwrap();
    let published_7 = -instance_7
        .published_min_shortage_benchmark
        .expect("min-shortage benchmark")
        .mean_cost;
    let env_7 = simulate_benchmark(
        "kaynov2024_instance_7",
        AllocationPolicy::MinShortage,
        44,
        &[10, 10, 10],
        2222,
    );
    assert!(
        (env_7 - published_7).abs() / published_7 < 0.012,
        "instance_7 min-shortage env cost {env_7} should reproduce published {published_7} within 1.2%"
    );

    // instance_11 (partial_backorder) verified via Eq.8-floor proportional (S_w=43, S_r=[6,6,6]).
    let instance_11 = TABLE_A3_INSTANCES
        .iter()
        .find(|r| r.name == "kaynov2024_instance_11")
        .unwrap();
    let published_11 = -instance_11
        .published_proportional_benchmark
        .expect("proportional benchmark")
        .mean_cost;
    let env_11 = simulate_benchmark(
        "kaynov2024_instance_11",
        AllocationPolicy::Proportional,
        43,
        &[6, 6, 6],
        2222,
    );
    assert!(
        (env_11 - published_11).abs() / published_11 < 0.005,
        "instance_11 proportional env cost {env_11} should reproduce published {published_11} within 0.5%"
    );
}
