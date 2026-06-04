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
    // Kranenburg Table 5.2 is the only executable literature reproduction in this
    // family, so its numbers genuinely anchor a re-running assertion.
    assert!(KRANENBURG_2006_REFERENCE.reported_numbers_available);
    assert!(KRANENBURG_2006_REFERENCE.numbers_anchor_repo_assertions);
    // van Oers (2024) Table 1 numbers are recorded only. They are reported in the
    // paper but no env/solver re-runs them here, so they must NOT anchor a
    // verified assertion: numbers_anchor_repo_assertions stays false.
    assert!(VAN_OERS_2024_REFERENCE.reported_numbers_available);
    assert!(!VAN_OERS_2024_REFERENCE.numbers_anchor_repo_assertions);
    // The trainable env.rs instances are honestly flagged NOT literature-verified
    // in references.rs itself (not only at the binding layer).
    assert!(!PRIMARY_REFERENCE_INSTANCE.literature_verified);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE.verification_source,
        "repo_native_periodic_review_env_not_verified_against_literature"
    );
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.installed_base, 12);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.procurement_lead_time, 3);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.installed_base, 3);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_order_quantity, 4);
}

// LITERATURE VERIFICATION (the only one in this family).
//
// This test RE-RUNS the analytical lateral-transshipment solver in
// literature/kranenburg_lateral_transshipment.rs and asserts that the freshly
// computed optimal randomized stock R* and total cost for Situation 1 (separate
// stock points) and Situation 3 (lateral transshipment) reproduce every printed
// row of Table 5.2 in Kranenburg (2006), "Spare parts inventory control under
// system availability constraints", PhD thesis, TU/e, Chapter 5, p.107, within
// table-rounding tolerance 0.02 (e.g. base case R1*=9.09 C1=91.90, R3*=6.10
// C3=63.00, ratio 1.46). This is a CONTINUOUS-REVIEW, METRIC-style multi-location
// model and is STRUCTURALLY DIFFERENT from the trainable env.rs; its verification
// covers the analytical module ONLY and says nothing about env.rs.
#[test]
fn kranenburg_table_5_2_rows_are_reproduced_within_table_rounding() {
    assert_eq!(KRANENBURG_2006_TABLE_5_2_ROWS.len(), 35);
    // The analytical Kranenburg subfamily is the only literature-verified anchor.
    assert!(KRANENBURG_2006_TABLE_5_2_ROWS[0].literature_verified);
    assert_eq!(
        KRANENBURG_2006_TABLE_5_2_ROWS[0].verification_source,
        "published_exact_table_reproduced_from_literature"
    );

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

// DRIFT GUARD ONLY -- NOT LITERATURE VERIFICATION.
//
// This test is a frozen snapshot: it asserts that the carried van Oers (2024)
// Table 1 constants equal themselves so they cannot silently drift. Per the repo
// rule (rust/README.md "What counts as literature-verified") a frozen snapshot is
// explicitly NOT verification, because nothing re-runs an env/solver to reproduce
// the numbers. The scenarios are therefore flagged literature_verified = false,
// which this test now pins.
#[test]
fn van_oers_2024_table_is_recorded_but_not_literature_verified() {
    assert_eq!(VAN_OERS_2024_TABLE_1_SCENARIOS.len(), 3);

    let no_am = &VAN_OERS_2024_TABLE_1_SCENARIOS[0];
    assert_eq!(no_am.name, "van_oers2024_table1_no_am");
    assert!(!no_am.literature_verified);
    assert_eq!(
        no_am.verification_source,
        "recorded_published_table_no_executing_reproduction"
    );
    assert_eq!(no_am.published_policy_results[0].base_stock_levels, &[8, 4]);
    assert_eq!(no_am.published_policy_results[0].reported_cost_value, 100.0);
    assert_eq!(no_am.published_policy_results[2].base_stock_levels, &[6, 6]);

    let upstream_am = &VAN_OERS_2024_TABLE_1_SCENARIOS[1];
    assert!(!upstream_am.literature_verified);
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
    assert!(!downstream_am.literature_verified);
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

// SELF-CONSISTENCY ONLY -- NOT LITERATURE VERIFICATION.
//
// This proves env.rs + finite_horizon_dp.rs are internally consistent: the exact
// finite-horizon DP optimum cannot cost more than the carried base-stock and
// lead-time-mean-cover heuristics on the reduced verification instance. It
// reproduces no paper-printed number, so it does not literature-verify env.rs.
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

// CHARACTERIZATION / DRIFT GUARD ONLY -- NOT LITERATURE VERIFICATION.
//
// Pins the repo-native PERIODIC-REVIEW env.rs dynamics over a full multi-period
// trajectory (the worked-transition test only pins a single step). It drives a
// fixed action sequence against a fixed failure path and asserts the per-period
// costs, the running balance after the deterministic repair return (a failed unit
// re-enters the pipeline and arrives exactly repair_lead_time periods later), and
// the total undiscounted cost. These numbers are computed by hand from env.rs's
// own accounting; no paper publishes this trajectory, so this test only protects
// env.rs against silent behavioral drift -- it does NOT literature-verify it.
#[test]
fn env_periodic_review_trajectory_is_pinned_characterization_not_literature() {
    // Reduced repairable instance: installed_base = 3, both lead times = 2.
    let installed_base = 3usize;
    let holding_cost = 0.5;
    let downtime_cost = 6.0;
    let procurement_cost = 2.0;

    let mut state = initialize_state(1, 0, &[0, 0], &[0, 0], installed_base)
        .expect("initial state must build");

    // (order_quantity, realized_failures, expected period cost) for each period.
    let plan: [(usize, usize, f64); 4] = [
        (2, 1, 4.0),
        (1, 2, 14.0),
        (0, 0, 12.0),
        (0, 1, 0.0),
    ];

    let mut total_cost = 0.0;
    for (period, (order, failures, expected_cost)) in plan.iter().enumerate() {
        let outcome = step_state(
            &state,
            *order,
            *failures,
            installed_base,
            holding_cost,
            downtime_cost,
            procurement_cost,
        )
        .expect("env step must succeed");
        assert!(
            (outcome.period_cost - *expected_cost).abs() < 1e-9,
            "period {} cost {} != expected {}",
            period,
            outcome.period_cost,
            expected_cost
        );
        total_cost += outcome.period_cost;
        state = outcome.next_state;
    }

    // Deterministic repair returns and procurement arrivals settle the backlog by
    // the final period; on-hand equals the installed base, backlog is cleared.
    assert_eq!(state.on_hand_inventory, 3);
    assert_eq!(state.backlog, 0);
    assert_eq!(state.procurement_pipeline, vec![0, 0]);
    assert_eq!(state.repair_pipeline, vec![0, 1]);
    assert!(
        (total_cost - 30.0).abs() < 1e-9,
        "total env trajectory cost {} != expected 30.0",
        total_cost
    );
}
