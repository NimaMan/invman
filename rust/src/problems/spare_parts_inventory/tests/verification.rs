use crate::problems::spare_parts_inventory::env::{build_raw_state, initialize_state, step_state};
use crate::problems::spare_parts_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::spare_parts_inventory::heuristics::{
    base_stock_order_quantity, lead_time_mean_cover_order_quantity,
};
use crate::problems::spare_parts_inventory::literature::kranenburg_lateral_transshipment::{
    compare_to_published_table, evaluate_reference_instance, KRANENBURG_TABLE_ROUNDING_TOLERANCE,
};
use crate::problems::spare_parts_inventory::references::{
    KRANENBURG_2006_REFERENCE, KRANENBURG_2006_TABLE_5_2_ROWS, PRIMARY_REFERENCE_INSTANCE,
    SPARE_PARTS_REVIEW_REFERENCE, VAN_DER_HAAR_2025_REFERENCE, VAN_OERS_2024_REFERENCE,
    VAN_OERS_2024_TABLE_1_SCENARIOS, VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE,
    ZHOU_2024_REFERENCE,
};

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(SPARE_PARTS_REVIEW_REFERENCE.benchmark_policies.len(), 3);
    assert!(!SPARE_PARTS_REVIEW_REFERENCE.reported_numbers_available);
    assert_eq!(
        ZHOU_2024_REFERENCE.benchmark_policies,
        &["marl", "multi-echelon spare-parts baselines"]
    );
    assert!(!ZHOU_2024_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(
        VAN_DER_HAAR_2025_REFERENCE.benchmark_policies,
        &[
            "drl",
            "distance-based transshipment and expediting heuristics"
        ]
    );
    assert!(KRANENBURG_2006_REFERENCE.reported_numbers_available);
    assert!(KRANENBURG_2006_REFERENCE.numbers_anchor_repo_assertions);
    assert!(VAN_OERS_2024_REFERENCE.reported_numbers_available);
    assert!(VAN_OERS_2024_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.installed_base, 12);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.procurement_lead_time, 3);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.installed_base, 3);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_order_quantity, 4);
}

#[test]
fn kranenburg_table_5_2_rows_are_reproduced_within_table_rounding() {
    assert_eq!(KRANENBURG_2006_TABLE_5_2_ROWS.len(), 35);

    for reference in KRANENBURG_2006_TABLE_5_2_ROWS {
        let evaluation =
            evaluate_reference_instance(reference).expect("Kranenburg row must evaluate");
        let comparison =
            compare_to_published_table(reference, &evaluation, KRANENBURG_TABLE_ROUNDING_TOLERANCE);
        assert!(
            comparison.all_within_tolerance,
            "reference {} deviated too far from the published table: {:?}",
            reference.name, comparison
        );
    }
}

#[test]
fn van_oers_2024_table_is_recorded_exactly() {
    assert_eq!(VAN_OERS_2024_TABLE_1_SCENARIOS.len(), 3);

    let no_am = &VAN_OERS_2024_TABLE_1_SCENARIOS[0];
    assert_eq!(no_am.name, "van_oers2024_table1_no_am");
    assert!(no_am.literature_verified);
    assert_eq!(
        no_am.verification_source,
        "published_benchmark_table_from_literature"
    );
    assert_eq!(no_am.published_policy_results[0].base_stock_levels, &[8, 4]);
    assert_eq!(no_am.published_policy_results[0].reported_cost_value, 100.0);
    assert_eq!(no_am.published_policy_results[2].base_stock_levels, &[6, 6]);

    let upstream_am = &VAN_OERS_2024_TABLE_1_SCENARIOS[1];
    assert_eq!(upstream_am.am_location, "upstream");
    assert_eq!(upstream_am.am_lead_time_hours, Some(6.42));
    assert_eq!(
        upstream_am.published_policy_results[1].base_stock_levels,
        &[5, 0]
    );
    assert_eq!(
        upstream_am.published_policy_results[2].reported_readiness_percent,
        98.81
    );

    let downstream_am = &VAN_OERS_2024_TABLE_1_SCENARIOS[2];
    assert_eq!(downstream_am.am_location, "downstream");
    assert_eq!(
        downstream_am.published_policy_results[0].base_stock_levels,
        &[5, 0]
    );
    assert_eq!(
        downstream_am.published_policy_results[0].reported_cost_value,
        71.98
    );
    assert_eq!(
        downstream_am.published_policy_results[2].reported_cost_value,
        72.01
    );
}

#[test]
fn raw_state_layout_matches_expected_shape() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        VERIFICATION_PROBLEM_INSTANCE.initial_procurement_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.initial_repair_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.installed_base,
    )
    .expect("state must build");
    let raw_state = build_raw_state(&state).expect("raw state must build");

    assert_eq!(raw_state, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(
        worked.initial_on_hand_inventory,
        worked.initial_backlog,
        worked.initial_procurement_pipeline,
        worked.initial_repair_pipeline,
        worked.installed_base,
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.action,
        worked.realized_failures,
        worked.installed_base,
        worked.holding_cost,
        worked.downtime_cost,
        worked.procurement_cost,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.procurement_arrival,
        worked.expected_procurement_arrival
    );
    assert_eq!(outcome.repair_return, worked.expected_repair_return);
    assert_eq!(
        outcome.post_failure_on_hand_inventory,
        worked.expected_post_failure_on_hand_inventory
    );
    assert_eq!(
        outcome.post_failure_backlog,
        worked.expected_post_failure_backlog
    );
    assert_eq!(
        outcome.next_state.on_hand_inventory,
        worked.expected_next_on_hand_inventory
    );
    assert_eq!(outcome.next_state.backlog, worked.expected_next_backlog);
    assert_eq!(
        outcome.next_state.procurement_pipeline,
        worked.expected_next_procurement_pipeline.to_vec()
    );
    assert_eq!(
        outcome.next_state.repair_pipeline,
        worked.expected_next_repair_pipeline.to_vec()
    );
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn heuristic_first_actions_match_named_heuristic_evaluators() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        VERIFICATION_PROBLEM_INSTANCE.initial_procurement_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.initial_repair_pipeline,
        VERIFICATION_PROBLEM_INSTANCE.installed_base,
    )
    .expect("state must build");
    let base_stock =
        base_stock_order_quantity(&state, VERIFICATION_PROBLEM_INSTANCE.base_stock_level)
            .expect("base-stock must compute");
    let mean_cover = lead_time_mean_cover_order_quantity(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.installed_base,
        VERIFICATION_PROBLEM_INSTANCE.failure_probability,
        VERIFICATION_PROBLEM_INSTANCE.lead_time_mean_cover_safety_buffer,
    )
    .expect("lead-time mean-cover must compute");

    let base_stock_eval = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "base_stock")
        .expect("base-stock evaluation must solve");
    let mean_cover_eval =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "lead_time_mean_cover")
            .expect("lead-time mean-cover evaluation must solve");

    assert_eq!(base_stock, base_stock_eval.first_action);
    assert_eq!(mean_cover, mean_cover_eval.first_action);
}

#[test]
fn exact_dp_dominates_repo_heuristics() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let base_stock = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "base_stock")
        .expect("base-stock evaluation must solve");
    let mean_cover =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "lead_time_mean_cover")
            .expect("lead-time mean-cover evaluation must solve");

    assert!(
        optimal.discounted_cost <= base_stock.discounted_cost + 1e-9,
        "optimal={} base_stock={}",
        optimal.discounted_cost,
        base_stock.discounted_cost
    );
    assert!(
        optimal.discounted_cost <= mean_cover.discounted_cost + 1e-9,
        "optimal={} lead_time_mean_cover={}",
        optimal.discounted_cost,
        mean_cover.discounted_cost
    );
}
