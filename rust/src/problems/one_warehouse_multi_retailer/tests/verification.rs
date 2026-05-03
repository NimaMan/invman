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
    assert_eq!(
        proportional.iter().sum::<usize>(),
        min_shortage.iter().sum::<usize>()
    );
    assert!(
        proportional.iter().sum::<usize>()
            <= (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize
    );
}

#[test]
fn proportional_allocation_uses_all_available_inventory_when_orders_exceed_supply() {
    let shipments =
        proportional_shipments(5, &[4, 4, 4]).expect("proportional allocation must compute");
    assert_eq!(shipments.iter().sum::<usize>(), 5);
    assert_eq!(shipments, vec![2, 2, 1]);
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
