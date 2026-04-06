use crate::problems::perishable_inventory::env::{build_raw_state, PerishableState};
use crate::problems::perishable_inventory::references::{
    build_lifetime_sweep_instances, get_primary_reference_instance, get_reference_instance,
    list_reference_instances, BENCHMARK_POLICIES, DE_MOOR_2022_REFERENCE,
    FARRINGTON_2025_REFERENCE, VERIFICATION_PROBLEM_INSTANCE, VERIFICATION_PROBLEM_INSTANCES,
};
use crate::problems::perishable_inventory::value_iteration_mdp::{
    best_base_stock_level_by_expected_return, build_exact_mdp, build_policy_table_9x9,
    expected_discounted_return_from_zero_state, value_iteration_best_action_values,
};

#[test]
fn scenario_a_reference_set_has_expected_shape() {
    let primary = get_primary_reference_instance();
    let instances = list_reference_instances();
    let lifetime_sweep = build_lifetime_sweep_instances(&[3, 4, 5]);

    assert_eq!(instances.len(), 32);
    assert_eq!(primary.name, "de_moor2022_m2_exp2_l1_cp7_fifo");
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.reference_instance_name,
        "de_moor2022_m2_exp2_l1_cp7_fifo"
    );
    assert!(instances
        .iter()
        .all(|instance| instance.max_order_size == 10));
    assert!(instances.iter().all(|instance| instance.demand_mean == 4.0));
    assert_eq!(
        DE_MOOR_2022_REFERENCE.benchmark_policies,
        BENCHMARK_POLICIES
    );
    assert_eq!(
        FARRINGTON_2025_REFERENCE.benchmark_policies,
        &["value_iteration", "base_stock"]
    );
    assert_eq!(lifetime_sweep.len(), 24);
    assert_eq!(
        primary
            .published_scenario_a_returns
            .expect("primary Scenario A metrics must exist")
            .value_iteration_mean_return,
        -1457
    );
    assert_eq!(
        primary
            .published_figure3_verification
            .expect("primary figure verification must exist")
            .published_base_stock_level,
        7
    );
}

#[test]
fn m2_base_stock_levels_match_de_moor_figure_3() {
    for verification in VERIFICATION_PROBLEM_INSTANCES {
        let mdp = build_exact_mdp(verification.reference_instance_name);
        let best_level =
            best_base_stock_level_by_expected_return(verification.reference_instance_name, &mdp);
        assert_eq!(best_level, verification.published_base_stock_level);
    }
}

#[test]
fn m2_optimal_policies_match_de_moor_figure_3_and_2025_returns() {
    for verification in VERIFICATION_PROBLEM_INSTANCES {
        let instance = get_reference_instance(verification.reference_instance_name)
            .expect("verification instance must exist");
        let mdp = build_exact_mdp(verification.reference_instance_name);
        let (policy, _) = value_iteration_best_action_values(&mdp, 0.99);
        let expected_return = expected_discounted_return_from_zero_state(
            verification.reference_instance_name,
            &mdp,
            &policy,
        );
        let policy_table = build_policy_table_9x9(&policy, &mdp);

        assert_eq!(policy_table, *verification.published_optimal_policy);
        assert_eq!(
            expected_return.round() as i32,
            verification.published_value_iteration_mean_return
        );

        let figure = instance
            .published_figure3_verification
            .expect("figure 3 verification must exist");
        assert_eq!(
            figure.published_optimal_policy,
            verification.published_optimal_policy
        );
    }
}

#[test]
fn policy_state_order_matches_official_observation_layout() {
    let state = PerishableState {
        pipeline_orders: vec![5, 3],
        on_hand: vec![4, 2, 1],
    };
    let policy_state = build_raw_state(&state);

    assert_eq!(policy_state, vec![5.0, 3.0, 4.0, 2.0, 1.0]);
}
