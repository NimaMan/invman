use crate::problems::dual_sourcing::bounded_dp::{benchmark_reference_instance, BoundedDpConfig};
use crate::problems::dual_sourcing::env::{epoch_cost, step_state};
use crate::problems::dual_sourcing::references::{
    get_figure_9_gap_reference, get_primary_reference_instance, get_reference_instance,
    list_reference_instances, FIGURE_9_GAP_REFERENCES, GIJSBRECHTS_2022_REFERENCE,
    PRIMARY_REFERENCE_INSTANCE, SHEOPURI_2010_REFERENCE, VEERARAGHAVAN_2008_REFERENCE,
    VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE,
};

#[test]
fn reference_set_has_expected_shape() {
    let instances = list_reference_instances();
    let primary = get_primary_reference_instance();

    assert_eq!(instances.len(), 6);
    assert_eq!(GIJSBRECHTS_2022_REFERENCE.benchmark_policies.len(), 7);
    assert_eq!(
        VEERARAGHAVAN_2008_REFERENCE.benchmark_policies,
        &["optimal_dp", "dual_index", "single_sourcing"]
    );
    assert_eq!(
        SHEOPURI_2010_REFERENCE.benchmark_policies,
        &[
            "single_index",
            "dual_index",
            "best_weighted_bounds",
            "tailored_base_surge"
        ]
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.name, "dual_l4_ce110");
    assert_eq!(primary.regular_lead_time, 4);
    assert_eq!(primary.expedited_order_cost, 110.0);
}

#[test]
fn benchmark_rows_match_gijsbrechts_small_scale_family() {
    let dual_l2_ce105 = get_reference_instance("dual_l2_ce105").expect("instance must exist");
    let dual_l4_ce110 = get_reference_instance("dual_l4_ce110").expect("instance must exist");

    assert_eq!(dual_l2_ce105.regular_lead_time, 2);
    assert_eq!(dual_l2_ce105.expedited_lead_time, 0);
    assert_eq!(dual_l2_ce105.regular_order_cost, 100.0);
    assert_eq!(dual_l2_ce105.expedited_order_cost, 105.0);
    assert_eq!(dual_l2_ce105.holding_cost, 5.0);
    assert_eq!(dual_l2_ce105.shortage_cost, 495.0);
    assert_eq!(dual_l2_ce105.demand_low, 0);
    assert_eq!(dual_l2_ce105.demand_high, 4);

    assert_eq!(dual_l4_ce110.regular_lead_time, 4);
    assert_eq!(dual_l4_ce110.expedited_order_cost, 110.0);
}

#[test]
fn figure_9_gap_labels_are_frozen() {
    let dual_l2_ce105 = get_figure_9_gap_reference("dual_l2_ce105").expect("gap row must exist");
    let dual_l4_ce110 = get_figure_9_gap_reference("dual_l4_ce110").expect("gap row must exist");

    assert_eq!(FIGURE_9_GAP_REFERENCES.len(), 6);
    assert_eq!(dual_l2_ce105.capped_dual_index_gap_pct, 0.00);
    assert_eq!(dual_l2_ce105.dual_index_gap_pct, 0.11);
    assert_eq!(dual_l2_ce105.single_index_gap_pct, 0.56);
    assert_eq!(dual_l2_ce105.tailored_base_surge_gap_pct, 0.06);
    assert_eq!(dual_l2_ce105.a3c_gap_pct, 0.52);

    assert_eq!(dual_l4_ce110.capped_dual_index_gap_pct, 0.11);
    assert_eq!(dual_l4_ce110.dual_index_gap_pct, 0.49);
    assert_eq!(dual_l4_ce110.single_index_gap_pct, 2.44);
    assert_eq!(dual_l4_ce110.tailored_base_surge_gap_pct, 0.58);
    assert_eq!(dual_l4_ce110.a3c_gap_pct, 1.33);
}

#[test]
fn worked_transition_matches_reduced_state_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let next_state = step_state(
        worked.initial_reduced_state,
        worked.regular_order,
        worked.expedited_order,
        worked.realized_demand,
    );
    let period_cost = epoch_cost(
        worked.initial_reduced_state,
        worked.regular_order,
        worked.expedited_order,
        worked.realized_demand,
        worked.regular_order_cost,
        worked.expedited_order_cost,
        worked.holding_cost,
        worked.shortage_cost,
    );

    assert_eq!(next_state, worked.expected_next_reduced_state.to_vec());
    assert_eq!(period_cost, worked.expected_period_cost);
}

#[test]
fn single_verification_instance_matches_repo_and_literature_tolerances() {
    let verification = VERIFICATION_PROBLEM_INSTANCE;
    let config = BoundedDpConfig {
        inventory_lower: verification.inventory_lower,
        inventory_upper: verification.inventory_upper,
        tolerance: verification.solver_tolerance,
        max_iterations: verification.max_iterations,
    };
    let report = benchmark_reference_instance(
        verification.reference_instance_name,
        &config,
        verification.search_seed,
        verification.search_horizon,
        verification.warm_up_periods_ratio,
    )
    .expect("benchmark report must build");
    let published = get_figure_9_gap_reference(verification.reference_instance_name)
        .expect("published gap row must exist");

    assert!(report.optimal.average_cost.is_finite());
    assert!(report.optimal.average_cost > 0.0);

    for heuristic in report.heuristics.iter() {
        let published_gap = match heuristic.policy_name {
            "capped_dual_index" => published.capped_dual_index_gap_pct,
            "dual_index" => published.dual_index_gap_pct,
            "single_index" => published.single_index_gap_pct,
            "tailored_base_surge" => published.tailored_base_surge_gap_pct,
            other => panic!("unexpected heuristic {other}"),
        };
        assert!(
            (heuristic.optimality_gap_pct - published_gap).abs()
                <= verification.literature_gap_abs_tolerance_pct
        );
    }
}
